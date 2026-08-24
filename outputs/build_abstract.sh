#!/usr/bin/env bash
# build_abstract.sh — compile iros_vas_abstract.tex to PDF from WSL
# Usage: bash build_abstract.sh
# Prerequisites: texlive-publishers (provides IEEEtran.cls)

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building IROS VAS Abstract ==="

# Install IEEEtran if not available
if ! kpsewhich IEEEtran.cls > /dev/null 2>&1; then
    echo "IEEEtran.cls not found. Installing texlive-publishers..."
    sudo apt-get install -y texlive-publishers
fi

# Compile (twice for references)
pdflatex -interaction=nonstopmode iros_vas_abstract.tex
pdflatex -interaction=nonstopmode iros_vas_abstract.tex

echo ""
echo "Done! Output: $SCRIPT_DIR/iros_vas_abstract.pdf"
