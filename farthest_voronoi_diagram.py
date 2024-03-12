import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull
from utils import Circle, angle_from_triplet


class CHVertex:
    def __init__(self, point: npt.NDArray[np.float32], idx: int):
        self.point = point
        self.idx = idx


class DCELVertex:
    def __init__(self, point: npt.NDArray[np.float32], circle_vertices: list[int], faraway=False):
        self.point = point
        self.circle_vertices = circle_vertices
        self.faraway = faraway


class DCELEdge:
    def __init__(self, origin_vert: DCELVertex, dest_vert: DCELVertex, origin_idx: int, dest_idx: int):
        self.origin_vert = origin_vert
        self.dest_vert = dest_vert
        self.origin_idx = origin_idx
        self.dest_idx = dest_idx
        self.neighbor = list(set(origin_vert.circle_vertices) & set(dest_vert.circle_vertices))


class FarthestVoronoiDiagram:
    def __init__(self, vertices: list[DCELVertex], edges: list[DCELEdge], points: list[int]):
        self.vertices = vertices
        self.edges = edges
        self.points = points
        
    @staticmethod
    def from_points(points: npt.NDArray[np.float32]) -> 'FarthestVoronoiDiagram':
        ch = ConvexHull(points)
        ch_vertices = [CHVertex(point=points[ch.vertices[i]], idx=i) for i in range(len(ch.vertices))]
        h = len(ch.vertices)
        
        fvd_edges: list[DCELEdge] = []
        fvd_points = list(range(h))
        fvd_vertices: list[DCELVertex] = []
        for i in range(h):
            next_point = points[ch.vertices[(i + 1) % h]]
            this_point = points[ch.vertices[i]]
            
            _i = this_point[1] - next_point[1]
            _j = next_point[0] - this_point[0]
            x = (next_point[1] + this_point[1]) / 2 + _i * 100
            y = (next_point[0] + this_point[0]) / 2 + _j * 100
            fvd_vertices.append(DCELVertex(
                point=np.array([x, y]),
                circle_vertices=[ch_vertices[i].idx, ch_vertices[(i + 1) % h].idx],
                faraway=True
            ))       

        while len(ch_vertices) > 2:
            h = len(ch_vertices)
            
            idx, circle = maximal_circle(ch_vertices)
            prev_idx = (h + idx - 1) % h
            this_vert = ch_vertices[idx]
            prev_vert = ch_vertices[prev_idx]
            next_vert = ch_vertices[(idx + 1) % h]
            
            center_idx = len(fvd_vertices)
            center_wrapped = DCELVertex(
                point=circle.center,
                circle_vertices=[prev_vert.idx, this_vert.idx, next_vert.idx]
            )
            
            fvd_edges.append(DCELEdge(
                origin_vert=center_wrapped,
                dest_vert=fvd_vertices[prev_vert.idx],
                origin_idx=center_idx,
                dest_idx=prev_vert.idx))
            fvd_edges.append(DCELEdge(
                origin_vert=center_wrapped,
                dest_vert=fvd_vertices[this_vert.idx],
                origin_idx=center_idx,
                dest_idx=this_vert.idx))
            
            fvd_vertices.append(center_wrapped)
            fvd_points[ch_vertices[idx].idx] = center_idx
            fvd_vertices[prev_vert.idx] = center_wrapped
            ch_vertices.pop(idx)

        h = len(ch_vertices)
        prev_idx = (h + idx - 1) % h
        this_idx = (prev_idx + 1) % h

        prev_vert = ch_vertices[prev_idx]
        this_vert = ch_vertices[this_idx]
        
        fvd_edges.append(DCELEdge(
            origin_vert=fvd_vertices[prev_vert.idx],
            dest_vert=fvd_vertices[this_vert.idx],
            origin_idx=prev_vert.idx,
            dest_idx=this_vert.idx))
            
        fvd_vertices.append(fvd_vertices[-1])
        fvd_points[prev_idx] = len(fvd_vertices) - 1
        fvd_vertices.append(fvd_vertices[-1])
        fvd_points[this_idx] = len(fvd_vertices) - 1
            
        return FarthestVoronoiDiagram(fvd_vertices, fvd_edges, fvd_points)


def maximal_circle(ch_vertices: list[CHVertex]) -> tuple[int, Circle]:
    max_circle = Circle(np.array([0, 0]), 0)
    max_angle = np.float32(0.)
    h = len(ch_vertices)
    
    for i in range(h):
        p_prev = ch_vertices[i - 1].point
        p_curr = ch_vertices[i].point
        p_next = ch_vertices[(i + 1) % h].point
        
        circle = Circle.from_triplet(p_prev, p_curr, p_next)
        angle = angle_from_triplet(p_prev, p_curr, p_next)
         
        if i == 0 or circle.radius >= max_circle.radius:
            if circle.radius != max_circle.radius or angle > max_angle:
                max_circle = circle
                max_angle = angle
                max_index = i
            
    return max_index, max_circle
    