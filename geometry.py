"""
===============================================================================
                    GEOMETRIC & TOPOLOGICAL CORE (GEOMETRY)
===============================================================================

  "Where points align and borders meet,
   A quiet test ensures complete."

  Provides FAST feature extraction, bounding triangle T0 generation, zero-weight
  boundary anchor evaluation, boundary-aware Shannon entropy calculation,
  and least-squares luminance plane fitting.

===============================================================================
"""

import cv2
import numpy as np


def extract_fast_anchors(img_gray: np.ndarray, target_anchors: int = 300) -> np.ndarray:
    """Extracts top-K FAST feature anchors using Non-Maximum Suppression (NMS)."""
    fast = cv2.FastFeatureDetector_create(threshold=5, nonmaxSuppression=True)
    keypoints = fast.detect(img_gray, None)

    if not keypoints:
        return np.empty((0, 2), dtype=np.float32)

    keypoints = sorted(keypoints, key=lambda kp: kp.response, reverse=True)[:target_anchors]
    return np.array([kp.pt for kp in keypoints], dtype=np.float32)


def compute_bounding_triangle_t0(width: int, height: int) -> np.ndarray:
    """Constructs outer bounding triangle T0 enclosing the domain [0, W] x [0, H]."""
    w, h = float(width), float(height)
    v1 = np.array([-w, -h], dtype=np.float32)
    v2 = np.array([3.0 * w, -h], dtype=np.float32)
    v3 = np.array([w / 2.0, 3.0 * h], dtype=np.float32)
    return np.array([v1, v2, v3], dtype=np.float32)


def evaluate_sub_triangle_counts_with_boundary(
    pts: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    v3: np.ndarray,
    t1: float,
    t2: float,
    t3: float,
    eps: float = 1e-3
) -> tuple:
    """
    Evaluates point distribution across 4 sub-triangles, isolating boundary points.

    Points lying on cutting lines (|Ax + By + C| < eps) receive zero weight.
    """
    if len(pts) == 0:
        return np.zeros(4, dtype=np.int32), 0

    p1 = v1 + t1 * (v2 - v1)
    p2 = v2 + t2 * (v3 - v2)
    p3 = v3 + t3 * (v1 - v3)

    a1, b1 = p1[1] - p2[1], p2[0] - p1[0]
    c1 = p1[0] * p2[1] - p2[0] * p1[1]

    a2, b2 = p2[1] - p3[1], p3[0] - p2[0]
    c2 = p2[0] * p3[1] - p3[0] * p2[1]

    a3, b3 = p3[1] - p1[1], p1[0] - p3[0]
    c3 = p3[0] * p1[1] - p1[0] * p3[1]

    lines = np.array([[a1, b1], [a2, b2], [a3, b3]], dtype=np.float32)
    consts = np.array([c1, c2, c3], dtype=np.float32)

    distances = pts @ lines.T + consts
    is_boundary = np.any(np.abs(distances) < eps, axis=1)
    boundary_count = int(np.sum(is_boundary))

    internal_pts = pts[~is_boundary]
    if len(internal_pts) == 0:
        return np.zeros(4, dtype=np.int32), boundary_count

    internal_distances = distances[~is_boundary]
    signs = internal_distances > 0.0

    weights = np.array([4, 2, 1], dtype=np.int32)
    region_ids = signs @ weights

    lut = np.array([0, 1, 2, 3, -1, -1, -1, -1], dtype=np.int32)
    valid_ids = lut[region_ids]

    counts = np.bincount(valid_ids[valid_ids >= 0], minlength=4).astype(np.int32)
    return counts, boundary_count


def calculate_boundary_aware_entropy(counts: np.ndarray, boundary_count: int, total_pts: int) -> float:
    """Computes Shannon entropy penalized and discounted by boundary alignment."""
    if total_pts == 0:
        return 0.0

    internal_sum = np.sum(counts)
    if internal_sum == 0:
        return 0.0

    probs = counts[counts > 0] / float(internal_sum)
    base_entropy = float(-np.sum(probs * np.log2(probs)))
    boundary_discount = (total_pts - boundary_count) / float(total_pts)
    return base_entropy * boundary_discount


def fit_luminance_plane(
    img: np.ndarray,
    v1: tuple,
    v2: tuple,
    v3: tuple
) -> (float, float, float):
    """
    Fits continuous luminance plane I(x, y) = a*x + b*y + c over triangular region
    using Ordinary Least Squares (OLS).
    """
    h, w = img.shape
    pts = np.array([v1, v2, v3], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, pts, 1)

    y_coords, x_coords = np.where(mask > 0)

    if len(x_coords) == 0:
        cx = float(np.clip((v1[0] + v2[0] + v3[0]) / 3.0, 0, w - 1))
        cy = float(np.clip((v1[1] + v2[1] + v3[1]) / 3.0, 0, h - 1))
        lum = float(img[int(cy), int(cx)])
        return 0.0, 0.0, lum

    z = img[y_coords, x_coords].astype(np.float64)
    A = np.column_stack([x_coords, y_coords, np.ones_like(x_coords, dtype=np.float64)])

    sol, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return float(sol[0]), float(sol[1]), float(sol[2])


# =============================================================================
#                                 TEST SUITE
# =============================================================================

def test_compute_bounding_triangle_t0():
    """Tests outer bounding triangle T0 enclosure over rectangular domain."""
    w, h = 512, 512
    t0 = compute_bounding_triangle_t0(w, h)
    assert t0.shape == (3, 2), "T0 must have shape (3, 2)"
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    v1, v2, v3 = t0[0], t0[1], t0[2]
    denom = (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1])
    for c in corners:
        w1 = ((v2[1] - v3[1]) * (c[0] - v3[0]) + (v3[0] - v2[0]) * (c[1] - v3[1])) / denom
        w2 = ((v3[1] - v1[1]) * (c[0] - v3[0]) + (v1[0] - v3[0]) * (c[1] - v3[1])) / denom
        w3 = 1.0 - w1 - w2
        assert w1 >= 0 and w2 >= 0 and w3 >= 0, f"Corner {c} lies outside T0"


def test_evaluate_sub_triangle_counts_with_boundary():
    """Tests boundary points identification and zero-weight filtering on cutting lines."""
    w, h = 512, 512
    t0 = compute_bounding_triangle_t0(w, h)
    v1, v2, v3 = t0[0], t0[1], t0[2]

    p1 = v1 + 0.5 * (v2 - v1)
    p2 = v2 + 0.5 * (v3 - v2)
    boundary_pt = p1 + 0.3 * (p2 - p1)
    pts = np.array([boundary_pt], dtype=np.float32)

    counts, b_count = evaluate_sub_triangle_counts_with_boundary(
        pts, v1, v2, v3, 0.5, 0.5, 0.5, eps=1e-2
    )
    assert b_count == 1, "Point on split line must be identified as boundary"
    assert np.sum(counts) == 0, "Boundary point must have zero count in sub-triangles"


def test_calculate_boundary_aware_entropy():
    """Tests entropy discounting when points align with boundaries."""
    counts = np.zeros(4, dtype=np.int32)
    ent = calculate_boundary_aware_entropy(counts, boundary_count=1, total_pts=1)
    assert ent == 0.0, "Entropy must be 0 when all points lie on boundaries"


def test_extract_fast_anchors():
    """Tests FAST feature extractor output shape and keypoint detection on a smooth circular edge."""
    synthetic_img = np.zeros((256, 256), dtype=np.uint8)
    
    cv2.circle(synthetic_img, (128, 128), 60, 255, -1, lineType=cv2.LINE_AA)

    anchors = extract_fast_anchors(synthetic_img, target_anchors=50)
    assert len(anchors) > 0, "FAST detector should find edge points sliding along the circle boundary"
    assert anchors.shape[1] == 2, "Anchors array must have 2 columns (x, y)"


def test_fit_luminance_plane():
    """Tests Ordinary Least Squares plane fitting over synthetic gradient surface."""
    y, x = np.ogrid[:100, :100]
    A, B, C = 1.0, 1.2, 10.
    linear_surface = (1.0 * x + 1.2 * y + 10.0).astype(np.float32)    
    a, b, c = fit_luminance_plane(linear_surface, (10, 10), (90, 10), (50, 90))
    
    assert abs(a - A) < 1e-1 and abs(b - B) < 1e-1 and abs(c - C), f"Plane fitting failed: got a={a}, b={b}"

# =============================================================================
#                                MODULE TESTS
# =============================================================================

if __name__ == "__main__":
    print("[TEST] Running geometry.py module tests...")

    test_compute_bounding_triangle_t0()
    test_evaluate_sub_triangle_counts_with_boundary()
    test_calculate_boundary_aware_entropy()
    test_extract_fast_anchors()
    test_fit_luminance_plane()

    print("[SUCCESS] All geometry.py tests passed cleanly!")