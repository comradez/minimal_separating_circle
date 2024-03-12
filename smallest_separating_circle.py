import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull
from farthest_voronoi_diagram import FarthestVoronoiDiagram, CHVertex, maximal_circle
from utils import Circle, Line


def find_maximal_surrounding_circle(points: npt.NDArray[np.float32]) -> Circle:
    convex_hull = ConvexHull(points)
    return maximal_circle(
        [
            CHVertex(point=points[convex_hull.vertices[i]], idx=i)
            for i in range(len(convex_hull.vertices))
        ]
    )[1]


def find_minimal_separating_circle(
    red_points: npt.NDArray[np.float32], blue_points: npt.NDArray[np.float32]
) -> tuple[Circle, list[npt.NDArray[np.float32]]]:

    def inv_lerp(
        a: npt.NDArray[np.float32],
        b: npt.NDArray[np.float32],
        p: npt.NDArray[np.float32],
    ) -> np.float32:
        t = (p - a) / (b - a)
        return t[0] if not np.isnan(t[0]) else t[1]

    red_hull = ConvexHull(red_points)
    red_fvd = FarthestVoronoiDiagram.from_points(red_points)

    smallest_sep_circle_center: npt.NDArray[np.float32] = None
    smallest_sep_circle_radius = float("inf")
    smallest_sep_circle_blue_point_num = 0x7FFFFFFF
    
    all_event_points = []

    for edge in red_fvd.edges:
        # the two vertices of the edge
        ei, ej = edge.origin_vert.point, edge.dest_vert.point
        # the two vertices on the convex hull, where ei ej is the bisector of vi vj
        vi, vj = (
            red_points[red_hull.vertices[edge.neighbor[0]]],
            red_points[red_hull.vertices[edge.neighbor[1]]],
        )

        # make sure that ei is closer to vi than ej
        if np.linalg.norm(ei - vi) > np.linalg.norm(ej - vi):
            ei, ej = ej, ei

        # all blue points inside the circle
        blue_points_in_circle = np.empty((0, 2), dtype=np.float32)

        # all event points on ei -> ej
        event_points: list[tuple[np.float32, npt.NDArray[np.float32], npt.NDArray[np.float32]]] = []

        for blue_point in blue_points:
            if np.linalg.norm(blue_point - ei) < np.linalg.norm(vi - ei):
                blue_points_in_circle = np.vstack([blue_points_in_circle, blue_point])

            circle = Circle.from_triplet(vi, vj, blue_point)

            t = inv_lerp(ei, ej, circle.center)
            if t > 0 and t < 1:
                # This event point is on the segment ei -> ej
                event_points.append((t, blue_point, circle.center))

        # sort the event points from ei to ej
        event_points.sort(key=lambda x: x[0])
        
        all_event_points.extend([event_point for _, _, event_point in event_points])

        # The circle with ei already separates the points
        if len(blue_points_in_circle) <= smallest_sep_circle_blue_point_num:
            center, radius = ei, np.linalg.norm(ei - vi)
            if (
                len(blue_points_in_circle) != smallest_sep_circle_blue_point_num
                or radius < smallest_sep_circle_radius
            ):
                smallest_sep_circle_center = center
                smallest_sep_circle_radius = radius
                smallest_sep_circle_blue_point_num = len(blue_points_in_circle)

        if smallest_sep_circle_blue_point_num == 0:
            continue

        # scan from ei to ej for each event point, maintain the set of blue points inside the circle
        for _, blue_point, event_point in event_points:
            line = Line.from_points(vi, vj)
            blue_point_idx = np.where(
                np.all(blue_points_in_circle == blue_point, axis=1)
            )

            # same side of the line
            if (
                line.distance_with_sign(event_point)
                * line.distance_with_sign(blue_point)
                > 0
            ):
                blue_points_in_circle = np.vstack([blue_points_in_circle, blue_point])
            else:
                blue_points_in_circle = np.delete(
                    blue_points_in_circle, blue_point_idx, axis=0
                )

            if len(blue_points_in_circle) <= smallest_sep_circle_blue_point_num:
                center, radius = event_point, np.linalg.norm(event_point - vi)
                if (
                    len(blue_points_in_circle) != smallest_sep_circle_blue_point_num
                    or radius < smallest_sep_circle_radius
                ):
                    smallest_sep_circle_center = center
                    smallest_sep_circle_radius = radius
                    smallest_sep_circle_blue_point_num = len(blue_points_in_circle)

                if smallest_sep_circle_blue_point_num == 0:
                    break

    return Circle(smallest_sep_circle_center, smallest_sep_circle_radius), all_event_points
