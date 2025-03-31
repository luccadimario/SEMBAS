import math


class Point:
    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return "Point(" + str(self.x) + ", " + str(self.y) + ")"

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def norm(self, p: int = 2) -> float:
        return (self.x**p + self.y**p) ** (1.0 / p)

    def dot(self, other: "Point") -> float:
        return self.x * other.x + self.y * other.y

    def __mul__(self, other: float) -> "Point":
        return Point(other * self.x, other * self.y)

    def __rmul__(self, other: float) -> "Point":
        return self.__mul__(other)

    def __truediv__(self, other: float) -> "Point":
        return self.__mul__(1.0 / other)

    def values(self):
        return [self.x, self.y]

    def distanceTo(self, other: "Point") -> float:
        return (self - other).norm(p=2)
    
    