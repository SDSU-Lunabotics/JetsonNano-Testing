import numpy as np


def plane_params(plane):
    # Support multiple ZED SDK Python API versions.
    if hasattr(plane, "normal"):
        n = plane.normal
        a0, b0, c0 = n.x, n.y, n.z
        d0 = plane.distance if hasattr(plane, "distance") else plane.get_distance()
        return a0, b0, c0, d0
    if hasattr(plane, "get_normal"):
        n = plane.get_normal()
        # n can be a struct with x/y/z or a sequence
        if hasattr(n, "x"):
            a0, b0, c0 = n.x, n.y, n.z
        else:
            a0, b0, c0 = n[0], n[1], n[2]
        if hasattr(plane, "get_distance"):
            d0 = plane.get_distance()
        else:
            eq = plane.get_plane_equation()
            d0 = eq[3]
        return a0, b0, c0, d0
    if hasattr(plane, "get_plane_equation"):
        eq = plane.get_plane_equation()
        return eq[0], eq[1], eq[2], eq[3]
    raise AttributeError("Unsupported Plane API: missing normal/normal getter")


def normalize_plane(a, b, c, d):
    # Ensure the plane normal points "up" (positive Y) so signed distance
    # is positive above the ground plane.
    if b < 0:
        return -a, -b, -c, -d
    return a, b, c, d


def classify_points(xyz, a, b, c, d, ground_thresh=0.10):
    # Distance to plane (signed)
    denom = np.sqrt(a * a + b * b + c * c)
    dist = (a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d) / denom

    # Ground threshold: within 10 cm of plane
    ground_mask = np.abs(dist) < ground_thresh
    # Wall/obstacle: above ground by > 10 cm
    obstacle_mask = dist > ground_thresh
    return dist, ground_mask, obstacle_mask
