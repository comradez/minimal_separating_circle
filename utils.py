import numpy as np
import numpy.typing as npt


class Line:
    """
    Represents a line in the form of ax + by + c = 0
    """
    def __init__(self, a: np.float32, b: np.float32, c: np.float32):
        self.a = a
        self.b = b
        self.c = c
    
    @staticmethod
    def from_points(point1: npt.NDArray[np.float32], point2: npt.NDArray[np.float32]) -> 'Line':
        a = point2[1] - point1[1]
        b = point1[0] - point2[0]
        c = point1[1] * point2[0] - point1[0] * point2[1]
        return Line(a, b, c)

    def distance_with_sign(self, point: npt.NDArray[np.float32]) -> np.float32:
        return (self.a * point[0] + self.b * point[1] + self.c) / np.sqrt(self.a ** 2 + self.b ** 2)


class Circle:
    """
    Represents a circle in the form of (x - center[0])^2 + (y - center[1])^2 = radius^2
    """
    def __init__(self, center: npt.NDArray[np.float32], radius: np.float32):
        self.center = center
        self.radius = radius
        
    @staticmethod
    def from_triplet(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32], c: npt.NDArray[np.float32]):    
        m = (a + b) / 2
        n = (b + c) / 2
        
        if np.allclose(m, n):
            return Circle(m, np.linalg.norm(a - m))
        
        k_m = np.array([-a[1] + b[1], a[0] - b[0]])
        k_n = np.array([-b[1] + c[1], b[0] - c[0]])
        
        k_diff = np.array([k_m[0] * k_n[0], k_m[1] * k_n[0] - k_m[0] * k_n[1]])
        b_diff = np.array([k_m[0] * k_n[0], k_m[1] * k_n[0] * m[0] - k_m[0] * k_n[1] * n[0] + k_m[0] * k_n[0] * (n[1] - m[1])])

        x_coord = b_diff[1] / k_diff[1]
        if np.allclose(k_m[0], 0):
            y_coord = k_n[1] / k_n[0] * (x_coord - n[0]) + n[1]
        else:
            y_coord = k_m[1] / k_m[0] * (x_coord - m[0]) + m[1]
            
        center = np.array([x_coord, y_coord])
        radius = np.linalg.norm(center - a)
        
        return Circle(center, radius)
    

def angle_from_triplet(a: npt.NDArray[np.float32], b: npt.NDArray[np.float32], c: npt.NDArray[np.float32]) -> np.float32:
    v1 = a - b
    v2 = c - b
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
