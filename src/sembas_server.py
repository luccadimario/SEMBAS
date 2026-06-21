#!/usr/bin/env python3
"""
sembas_server.py — Adaptive boundary scanner for the CARLoS scenario space.

Implements the SEMBAS socket protocol (server side) on 127.0.0.1:2000 and
drives a 3-phase adaptive boundary scan of [lane_width, num_obstacles, speed].

Protocol (server side):
  1. Accept TCP connection from carlos_sembas_client.py
  2. Receive ndim as big-endian int64 from client
  3. Send "OK\\n"
  4. Loop:
     a. Send ndim × float64 scenario vector x ∈ [0,1]^ndim  (NATIVE byte order)
     b. Receive 1-byte bool from client: 0x01=pass, 0x00=fail

NOTE: The float64 vector uses native byte order to match the client's
      struct.unpack(f"{ndim}d", ...) call in sembas_api.receive_request().

Algorithm:
  Phase 1 — Latin Hypercube Sampling (20 pts): find initial pass/fail examples.
  Phase 2 — Bisection: binary-search each (pass, fail) pair for a boundary pt.
  Phase 3 — Boundary Walk: perturb known boundary pts tangentially, bisect each
             straddle pair that is discovered to expand boundary coverage.

Run this BEFORE starting carlos_sembas_client.py (or use run_boundary_scan.sh).
"""

import json
import math
import os
import random
import socket
import struct
from dataclasses import dataclass, field
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────

HOST = "127.0.0.1"
PORT = 2000
NDIM = 3

INITIAL_SAMPLES = 15       # Phase 1 Latin Hypercube points
TARGET_BOUNDARY_PTS = 35   # stop after collecting this many boundary points
BISECTION_MAX_ITERS = 12   # max bisection steps per pair
BISECTION_TOL = 0.02       # stop bisecting when interval length < this (L2)
WALK_STEP = 0.06           # tangential walk step size in [0,1]^3
WALK_ATTEMPTS = 3          # random directions tried per boundary seed point

OUTPUT_FILE = "boundary_scan_results.json"

# ── Latin Hypercube Sampler ───────────────────────────────────────────────────

def latin_hypercube(n: int, d: int, seed: int = 0) -> list[list[float]]:
    rng = random.Random(seed)
    intervals = [[i / n, (i + 1) / n] for i in range(n)]
    cols = []
    for _ in range(d):
        perm = intervals[:]
        rng.shuffle(perm)
        cols.append([rng.uniform(lo, hi) for lo, hi in perm])
    return [[cols[dim][i] for dim in range(d)] for i in range(n)]


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


# ── Socket / protocol helpers ─────────────────────────────────────────────────

def send_query(conn: socket.socket, x: list[float]) -> bool:
    """Send scenario vector (native byte order) and receive 1-byte pass/fail."""
    conn.sendall(struct.pack(f"{NDIM}d", *x))   # native endian matches client's unpack
    return conn.recv(1) == b'\x01'


def handshake(server_sock: socket.socket) -> Optional[socket.socket]:
    print(f"Listening on {HOST}:{PORT} — launch carlos_sembas_client.py now…")
    conn, addr = server_sock.accept()
    print(f"Client connected: {addr}")

    raw = conn.recv(8)
    client_ndim = struct.unpack("!q", raw)[0]   # big-endian as per sembas_api.py
    if client_ndim != NDIM:
        print(f"ERROR: expected ndim={NDIM}, got {client_ndim}")
        conn.close()
        return None

    conn.sendall(b"OK\n")
    print(f"Handshake OK (ndim={NDIM})\n")
    return conn


# ── Scan state ────────────────────────────────────────────────────────────────

@dataclass
class ScanState:
    pass_pts: list[list[float]] = field(default_factory=list)
    fail_pts: list[list[float]] = field(default_factory=list)
    boundary_pts: list[list[float]] = field(default_factory=list)
    n_queries: int = 0


def _log(n: int, x: list[float], label: str) -> None:
    lw  = round(10 + x[0] * 4)        # 10–14 ft
    obs = round(4  + x[1] * 6)        # 4–10
    spd = 10.0 + x[2] * 65.0          # 10–75 mph
    print(f"  [{n:>4}] lane={lw}ft  obs={obs}  spd={spd:5.1f}mph  →  {label}")


def query_and_record(conn: socket.socket, state: ScanState, x: list[float]) -> bool:
    result = send_query(conn, x)
    state.n_queries += 1
    _log(state.n_queries, x, "PASS" if result else "FAIL")
    (state.pass_pts if result else state.fail_pts).append(x)
    return result


# ── Phase 1: Latin Hypercube Sampling ────────────────────────────────────────

def phase1_lhs(conn: socket.socket, state: ScanState) -> None:
    print(f"=== Phase 1: Latin Hypercube Sampling ({INITIAL_SAMPLES} pts) ===")
    # 5 guaranteed low-speed seeds (x[2] ∈ [0.0, 0.03] → speed ≤ ~12 mph).
    # The PASS zone is only the bottom ~2% of the speed axis, so pure LHS never
    # lands there.  Pinning speed near zero while varying lane_width and
    # num_obstacles ensures Phase 2 always gets at least one pass/fail pair.
    low_speed_seeds = [
        [1.0, 0.0,  0.000],  # 14 ft / 4 obs / 10.0 mph — confirmed PASS
        [0.5, 0.0,  0.010],  # 12 ft / 4 obs / 10.7 mph — likely PASS
        [1.0, 1.0,  0.015],  # 14 ft / 10 obs / 11.0 mph — uncertain
        [0.0, 0.5,  0.020],  # 10 ft / 7 obs / 11.3 mph — likely FAIL
        [0.5, 1.0,  0.030],  # 12 ft / 10 obs / 12.0 mph — likely FAIL
    ]
    for x in low_speed_seeds:
        query_and_record(conn, state, x)
    for x in latin_hypercube(INITIAL_SAMPLES - len(low_speed_seeds), NDIM, seed=0):
        query_and_record(conn, state, x)
    print(f"  → {len(state.pass_pts)} pass, {len(state.fail_pts)} fail\n")


# ── Phase 2: Bisection ────────────────────────────────────────────────────────

def bisect(
    conn: socket.socket,
    state: ScanState,
    p_pass: list[float],
    p_fail: list[float],
) -> Optional[list[float]]:
    """Binary-search between p_pass and p_fail; append the boundary pt to state."""
    lo, hi = list(p_pass), list(p_fail)
    for _ in range(BISECTION_MAX_ITERS):
        mid = [clamp01((lo[i] + hi[i]) / 2) for i in range(NDIM)]
        result = send_query(conn, mid)
        state.n_queries += 1
        _log(state.n_queries, mid, "PASS" if result else "FAIL")
        if result:
            lo = mid
            state.pass_pts.append(mid)
        else:
            hi = mid
            state.fail_pts.append(mid)
        if math.sqrt(sum((hi[i] - lo[i]) ** 2 for i in range(NDIM))) < BISECTION_TOL:
            break

    bp = [(lo[i] + hi[i]) / 2 for i in range(NDIM)]
    state.boundary_pts.append(bp)
    print(f"    ↳ boundary pt #{len(state.boundary_pts)}: {[f'{v:.3f}' for v in bp]}")
    return bp


def phase2_bisect(conn: socket.socket, state: ScanState, max_bdry: int) -> None:
    print(f"=== Phase 2: Bisection (up to {max_bdry} boundary pts) ===")
    rng = random.Random(1)
    pairs = [(pp, fp) for pp in state.pass_pts for fp in state.fail_pts]
    rng.shuffle(pairs)
    for pp, fp in pairs:
        if len(state.boundary_pts) >= max_bdry:
            break
        bisect(conn, state, pp, fp)
    print(f"  → {len(state.boundary_pts)} boundary pts after Phase 2\n")


# ── Phase 3: Boundary Walk ────────────────────────────────────────────────────

def walk_from(
    conn: socket.socket,
    state: ScanState,
    seed_pt: list[float],
    rng: random.Random,
) -> None:
    """Try WALK_ATTEMPTS tangential perturbations from seed_pt.

    For each perturbation direction d, query seed±step*d. If the two results
    differ, bisect between them to get a new boundary point and continue the
    walk from there.
    """
    current = seed_pt
    for _ in range(WALK_ATTEMPTS):
        if len(state.boundary_pts) >= TARGET_BOUNDARY_PTS:
            return

        d = [rng.gauss(0, 1) for _ in range(NDIM)]
        norm = math.sqrt(sum(v ** 2 for v in d))
        if norm < 1e-9:
            continue
        d = [v / norm for v in d]

        p_plus  = [clamp01(current[i] + d[i] * WALK_STEP) for i in range(NDIM)]
        p_minus = [clamp01(current[i] - d[i] * WALK_STEP) for i in range(NDIM)]

        r_plus = send_query(conn, p_plus)
        state.n_queries += 1
        _log(state.n_queries, p_plus, "PASS" if r_plus else "FAIL")
        (state.pass_pts if r_plus else state.fail_pts).append(p_plus)

        r_minus = send_query(conn, p_minus)
        state.n_queries += 1
        _log(state.n_queries, p_minus, "PASS" if r_minus else "FAIL")
        (state.pass_pts if r_minus else state.fail_pts).append(p_minus)

        if r_plus != r_minus:
            p_p = p_plus  if r_plus  else p_minus
            p_f = p_minus if r_plus  else p_plus
            new_bp = bisect(conn, state, p_p, p_f)
            if new_bp:
                current = new_bp


def phase3_walk(conn: socket.socket, state: ScanState) -> None:
    print(f"=== Phase 3: Boundary Walk (target: {TARGET_BOUNDARY_PTS} boundary pts) ===")
    rng = random.Random(2)
    seeds = list(state.boundary_pts)
    rng.shuffle(seeds)
    for bp in seeds:
        if len(state.boundary_pts) >= TARGET_BOUNDARY_PTS:
            break
        walk_from(conn, state, bp, rng)
    print(f"  → {len(state.boundary_pts)} boundary pts after Phase 3\n")


# ── Results ───────────────────────────────────────────────────────────────────

def save_results(state: ScanState) -> None:
    out = {
        "ndim": NDIM,
        "dimensions": ["lane_width_ft", "num_obstacles", "speed_mph"],
        "ranges": {
            "lane_width_ft":  [10, 14],
            "num_obstacles":  [4, 10],
            "speed_mph":      [10, 75],
        },
        "n_queries": state.n_queries,
        "boundary_points_normalized": state.boundary_pts,
        "pass_points_normalized":     state.pass_pts,
        "fail_points_normalized":     state.fail_pts,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), OUTPUT_FILE)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run() -> None:
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, PORT))
    server_sock.listen(1)

    conn = handshake(server_sock)
    if conn is None:
        server_sock.close()
        return

    state = ScanState()
    try:
        phase1_lhs(conn, state)

        if not state.pass_pts or not state.fail_pts:
            print("Cannot proceed: Phase 1 must find at least one PASS and one FAIL.")
            return

        phase2_bisect(conn, state, max_bdry=20)
        phase3_walk(conn, state)

    except (ConnectionResetError, BrokenPipeError, OSError, struct.error) as exc:
        print(f"\nConnection lost: {exc}")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        conn.close()
        server_sock.close()

    print("=== Scan complete ===")
    print(f"  Total queries:   {state.n_queries}")
    print(f"  Boundary points: {len(state.boundary_pts)}")
    print(f"  Pass / Fail:     {len(state.pass_pts)} / {len(state.fail_pts)}")
    save_results(state)


if __name__ == "__main__":
    run()
