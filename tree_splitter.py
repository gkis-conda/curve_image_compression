"""
===============================================================================
               PARAMETRIC L-GRAMMAR TREE SPLITTER (TREE_SPLITTER)
===============================================================================

  Combines 125-point Golden Section FAST boundary entropy with gradient-based
  refinement and depth-adaptive bit-quantization for t_i.

  Strictly compliant with Python 3.6 ASCII standard.
===============================================================================
"""

import math
import numpy as np
from geometry import (
    evaluate_sub_triangle_counts_with_boundary,
    calculate_boundary_aware_entropy
)

# 5-point Golden Ratio Grid: 5 x 5 x 5 = 125 entropy evaluations
GOLDEN_GRID_5 = np.array([0.146, 0.382, 0.500, 0.618, 0.854], dtype=np.float32)


class LSymbol:
    """Parametric L-Grammar Symbol (S or F)."""
    def __init__(self, symbol_type, params=None):
        self.symbol_type = symbol_type  # 'S' or 'F'
        self.params = params if params is not None else {}

    def __repr__(self):
        if self.symbol_type == 'F':
            return "F()"
        splits = self.params.get('splits', [])
        formatted = ", ".join(
            ["(t_q={}/{}, t={:.3f}, L={:.1f}, R={:.1f})".format(q, max_q, t, dL, dR)
             for q, max_q, t, dL, dR in splits]
        )
        return "S[depth=%d, bits=%d: %s]" % (
            self.params.get('depth', 0),
            self.params.get('bits_per_t', 7),
            formatted
        )


def get_bits_for_depth(depth):
    """
    Law of diminishing precision: reduces bit-budget for t_i as depth increases.
    """
    if depth == 0:
        return 7  # 128 levels
    elif depth == 1:
        return 6  # 64 levels
    elif depth == 2:
        return 5  # 32 levels
    else:
        return 4  # 16 levels


def quantize_t_adaptive(t, bits):
    """Quantizes continuous t in (0, 1) to integers based on depth bit-budget."""
    max_val = (1 << bits) - 1
    q = int(round(np.clip(t, 0.0, 1.0) * max_val))
    t_dequantized = float(q) / float(max_val)
    return q, max_val, t_dequantized


def sample_image_intensity(img, x, y):
    """Safely samples image intensity at float coordinates."""
    h, w = img.shape[:2]
    ix = int(round(np.clip(x, 0, w - 1)))
    iy = int(round(np.clip(y, 0, h - 1)))
    return float(img[iy, ix])


def evaluate_edge_step_deltas(img, vA, vB, iA, iB, t):
    """
    Computes split point P_i and dual-sided left/right deltas (dI_L, dI_R)
    relative to baseline linear interpolation across edge vA -> vB.
    """
    Px = (1.0 - t) * vA[0] + t * vB[0]
    Py = (1.0 - t) * vA[1] + t * vB[1]
    P = np.array([Px, Py], dtype=np.float32)

    i_interp = (1.0 - t) * iA + t * iB

    dx = vB[0] - vA[0]
    dy = vB[1] - vA[1]
    length = math.sqrt(dx * dx + dy * dy) + 1e-6
    nx, ny = -dy / length, dx / length

    offset = 1.0
    val_L = sample_image_intensity(img, Px - nx * offset, Py - ny * offset)
    val_R = sample_image_intensity(img, Px + nx * offset, Py + ny * offset)

    dI_L = val_L - i_interp
    dI_R = val_R - i_interp

    return P, dI_L, dI_R, val_L, val_R


def find_fast_split_parameters(img, anchors, v1, v2, v3, i1, i2, i3):
    """
    Stage 1: Global evaluation over 125 Golden Ratio grid points (5x5x5).
    Stage 2: Local gradient jump maximization around candidate t_i.
    """
    total_pts = len(anchors)
    
    if total_pts == 0:
        return (0.5, 0.5, 0.5)

    best_entropy = float("inf")
    best_t = (0.5, 0.5, 0.5)

    # 1. Stage 1: 125 Golden Ratio Entropy Evaluations
    for t1 in GOLDEN_GRID_5:
        for t2 in GOLDEN_GRID_5:
            for t3 in GOLDEN_GRID_5:
                counts, b_count = evaluate_sub_triangle_counts_with_boundary(
                    anchors, v1, v2, v3, t1, t2, t3
                )
                ent = calculate_boundary_aware_entropy(counts, b_count, total_pts)
                if ent < best_entropy:
                    best_entropy = ent
                    best_t = (t1, t2, t3)

    # 2. Stage 2: Gradient Jump Maximization (|dL - dR|)
    edges = [(v1, v2, i1, i2), (v2, v3, i2, i3), (v3, v1, i3, i1)]
    refined_t = []

    for idx, (vA, vB, iA, iB) in enumerate(edges):
        t_rough = best_t[idx]
        best_grad = -1.0
        t_fine = t_rough

        # Search fine window around t_rough
        for dt in np.linspace(-0.05, 0.05, 5):
            t_cand = float(np.clip(t_rough + dt, 0.01, 0.99))
            _, dL, dR, _, _ = evaluate_edge_step_deltas(img, vA, vB, iA, iB, t_cand)
            grad = abs(dL - dR)
            if grad > best_grad:
                best_grad = grad
                t_fine = t_cand

        refined_t.append(t_fine)

    return tuple(refined_t)


def split_facet_dfs(img, anchors, v1, v2, v3, i1, i2, i3, depth=0, max_depth=3, error_threshold=5.0):
    """
    DFS Tree subdivision using 125-entropy + gradient search and adaptive quantization.
    """
    cx = (v1[0] + v2[0] + v3[0]) / 3.0
    cy = (v1[1] + v2[1] + v3[1]) / 3.0
    i_actual_center = sample_image_intensity(img, cx, cy)
    i_interp_center = (i1 + i2 + i3) / 3.0
    error = abs(i_actual_center - i_interp_center)

    if depth >= max_depth or error <= error_threshold:
        return [LSymbol('F')]

    # 1. Fast split search (125 entropy grid + local gradient refinement)
    t1_raw, t2_raw, t3_raw = find_fast_split_parameters(img, anchors, v1, v2, v3, i1, i2, i3)

    # 2. Adaptive quantization of t_i based on depth
    bits = get_bits_for_depth(depth)
    q1, max_q1, t1 = quantize_t_adaptive(t1_raw, bits)
    q2, max_q2, t2 = quantize_t_adaptive(t2_raw, bits)
    q3, max_q3, t3 = quantize_t_adaptive(t3_raw, bits)

    # 3. Evaluate deltas at quantized split points
    p1, dL1, dR1, iL1, iR1 = evaluate_edge_step_deltas(img, v1, v2, i1, i2, t1)
    p2, dL2, dR2, iL2, iR2 = evaluate_edge_step_deltas(img, v2, v3, i2, i3, t2)
    p3, dL3, dR3, iL3, iR3 = evaluate_edge_step_deltas(img, v3, v1, i3, i1, t3)

    split_symbol = LSymbol('S', {
        'depth': depth,
        'bits_per_t': bits,
        'splits': [
            (q1, max_q1, t1, dL1, dR1),
            (q2, max_q2, t2, dL2, dR2),
            (q3, max_q3, t3, dL3, dR3)
        ]
    })

    stream = [split_symbol]

    # Subdivide into 4 child triangles
    stream.extend(split_facet_dfs(img, anchors, v1, p1, p3, i1, iL1, iL3, depth + 1, max_depth, error_threshold))
    stream.extend(split_facet_dfs(img, anchors, p1, v2, p2, iR1, i2, iR2, depth + 1, max_depth, error_threshold))
    stream.extend(split_facet_dfs(img, anchors, p3, p2, v3, iR3, iL2, i3, depth + 1, max_depth, error_threshold))
    stream.extend(split_facet_dfs(img, anchors, p1, p2, p3, iL1, iR2, iL3, depth + 1, max_depth, error_threshold))

    return stream


# =============================================================================
#                               PURE ASSERT TESTS
# =============================================================================

def run_tests():
    print("[TEST] Running updated tree_splitter.py assertions...")

    # Test 1: Depth adaptive quantization check
    assert get_bits_for_depth(0) == 7, "Depth 0 must use 7 bits"
    assert get_bits_for_depth(1) == 6, "Depth 1 must use 6 bits"
    assert get_bits_for_depth(3) == 4, "Depth 3+ must use 4 bits"

    q, max_q, t_deq = quantize_t_adaptive(0.618, bits=7)
    assert max_q == 127, "7 bits max value must be 127"
    assert abs(t_deq - 0.618) < 0.01, "Quantization error must be small"

    # Test 2: Split execution with adaptive bits
    edge_img = np.zeros((64, 64), dtype=np.float32)
    edge_img[32:, :] = 200.0
    anchors = np.array([[32.0, 32.0], [30.0, 32.0]], dtype=np.float32)
    v1, v2, v3 = np.array([0, 0]), np.array([63, 0]), np.array([0, 63])

    stream = split_facet_dfs(edge_img, anchors, v1, v2, v3, 0.0, 200.0, 0.0, max_depth=2, error_threshold=1.0)
    assert stream[0].symbol_type == 'S', "Root symbol must be S"
    assert stream[0].params['bits_per_t'] == 7, "Root split must use 7 bits"

    print("[SUCCESS] All updated tree_splitter.py tests passed!")


def test_square_split_behavior():
    print("[RUN] Running Square FAST-Anchor Split Test...")

    # 1. Create test square image 64x64:
    # Top-left triangle = 0.0 (black)
    # Bottom-right triangle = 255.0 (white)
    # Diagonal boundary from (0, 63) to (63, 0)
    img = np.zeros((64, 64), dtype=np.float32)
    for y in range(64):
        for x in range(64):
            if x + y >= 63:
                img[y, x] = 255.0

    # 2. Base facet covering the square:
    # V1 = (0, 0)   [Black area]
    # V2 = (63, 0)  [Boundary transition]
    # V3 = (0, 63)  [Boundary transition]
    v1 = np.array([0.0, 0.0], dtype=np.float32)
    v2 = np.array([63.0, 0.0], dtype=np.float32)
    v3 = np.array([0.0, 63.0], dtype=np.float32)

    # 3. Define 3 FAST anchor points directly on edge V2-V3:
    # Point 1: Center (31.5, 31.5) -> t2 approx 0.5
    # Point 2: Near V2 (47.25, 15.75) -> t2 approx 0.25
    # Point 3: Near V3 (15.75, 47.25) -> t2 approx 0.75
    anchors = np.array([
        [31.5, 31.5],
        [47.25, 15.75],
        [15.75, 47.25]
    ], dtype=np.float32)

    # 4. Evaluate split parameters
    t1_raw, t2_raw, t3_raw = find_fast_split_parameters(
        img, anchors, v1, v2, v3, i1=0.0, i2=255.0, i3=255.0
    )

    print("\n--- Split Results for Square Boundary ---")
    print("t1 (Edge V1->V2): {:.3f}".format(t1_raw))
    print("t2 (Edge V2->V3, with 3 anchors): {:.3f}".format(t2_raw))
    print("t3 (Edge V3->V1): {:.3f}".format(t3_raw))

    # 5. Execute full L-grammar DFS stream generation
    stream = split_facet_dfs(
        img, anchors, v1, v2, v3,
        i1=0.0, i2=255.0, i3=255.0,
        depth=0, max_depth=1, error_threshold=1.0
    )

    root_symbol = stream[0]
    print("\nRoot L-Symbol:")
    print(root_symbol)

    # 6. Strict ASCII assertions
    assert root_symbol.symbol_type == 'S', "Root must be split symbol S"
    assert len(stream) == 5, "Should produce S + 4 child leaves F()"
    
    # Check delta jump across edge V2->V3
    splits = root_symbol.params['splits']
    _, _, _, dL2, dR2 = splits[1]  # Edge V2->V3
    assert abs(dL2 - dR2) > 50.0, "Delta jump across edge V2->V3 must be significant"

    print("\n[SUCCESS] Square split test passed completely!")



if __name__ == "__main__":
    run_tests()
    test_square_split_behavior()
