import numpy as np


class EquilibriumSet:
    def __init__(self, center, radius):
        self.center = np.array(center)
        self.radius = radius

        if not isinstance(radius, (int, float)) or radius <= 0:
            raise ValueError("Radius must be a positive number")

    def contains_point(self, point):
        point = np.array(point)

        if point.shape != self.center.shape:
            raise ValueError(
                f"Point must have the same dimensions as the center. Point: {point.shape} Center: {self.center.shape}")

        distance = np.linalg.norm(point - self.center)
        return distance <= self.radius
