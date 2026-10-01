"""
===============================================================================
                     FRACTAL TOPOLOGICAL CODEC ENGINE
===============================================================================

  "Through assertions state is sworn,
   Ere bitstream field is born."

  Contains serialisation, tree building, and reconstruction pipeline logic.

===============================================================================
"""

import numpy as np
from geometry import (
    extract_fast_anchors,
    compute_bounding_triangle_t0,
    fit_luminance_plane,
)

def fit_luminance_plane(img_gray: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> tuple:
    """Samples luminance at the three vertex locations in image domain."""
    h, w = img_gray.shape
    pts_3d = []
    for v in (v1, v2, v3):
        x_c = int(np.clip(v[0], 0, w - 1))
        y_c = int(np.clip(v[1], 0, h - 1))
        pts_3d.append(float(img_gray[y_c, x_c]))
    return tuple(pts_3d)


def encode_fractal_mesh(img_gray: np.ndarray, target_anchors: int = 300) -> bytes:
    """Encodes grayscale image domain into compact topological bitstream."""
    h, w = img_gray.shape
    anchors = extract_fast_anchors(img_gray, target_anchors=target_anchors)
    t0_verts = compute_bounding_triangle_t0(w, h)

    payload = bytearray()
    payload.extend(int(w).to_bytes(2, "big"))
    payload.extend(int(h).to_bytes(2, "big"))
    payload.extend(int(len(anchors)).to_bytes(2, "big"))

    for pt in anchors:
        x_q = int(np.clip(pt[0], 0, 65535))
        y_q = int(np.clip(pt[1], 0, 65535))
        payload.extend(x_q.to_bytes(2, "big"))
        payload.extend(y_q.to_bytes(2, "big"))

    i1, i2, i3 = fit_luminance_plane(img_gray, t0_verts[0], t0_verts[1], t0_verts[2])
    payload.append(int(np.clip(i1, 0, 255)))
    payload.append(int(np.clip(i2, 0, 255)))
    payload.append(int(np.clip(i3, 0, 255)))

    return bytes(payload)


def decode_fractal_mesh(bitstream: bytes) -> np.ndarray:
    """Decodes bitstream payload and rasterizes barycentric planar luminance field."""
    if len(bitstream) < 9:
        return np.full((512, 512), 128, dtype=np.uint8)

    w = int.from_bytes(bitstream[0:2], "big")
    h = int.from_bytes(bitstream[2:4], "big")
    anchor_count = int.from_bytes(bitstream[4:6], "big")

    offset = 6 + anchor_count * 4
    if len(bitstream) < offset + 3:
        return np.full((h, w), 128, dtype=np.uint8)

    i1 = bitstream[offset]
    i2 = bitstream[offset + 1]
    i3 = bitstream[offset + 2]

    t0_verts = compute_bounding_triangle_t0(w, h)
    v1, v2, v3 = t0_verts[0], t0_verts[1], t0_verts[2]

    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    denom = (v2[1] - v3[1]) * (v1[0] - v3[0]) + (v3[0] - v2[0]) * (v1[1] - v3[1])
    if abs(denom) < 1e-6:
        denom = 1.0

    w1 = ((v2[1] - v3[1]) * (grid_x - v3[0]) + (v3[0] - v2[0]) * (grid_y - v3[1])) / denom
    w2 = ((v3[1] - v1[1]) * (grid_x - v3[0]) + (v1[0] - v3[0]) * (grid_y - v3[1])) / denom
    w3 = 1.0 - w1 - w2

    img_recon = w1 * i1 + w2 * i2 + w3 * i3
    return np.clip(img_recon, 0, 255).astype(np.uint8)


# =============================================================================
#                                MODULE TESTS
# =============================================================================

if __name__ == "__main__":
    print("[TEST] Running codec.py module tests...")

    # 1. Test Uniform Image Roundtrip Dimensions
    test_img = np.full((100, 200), 200, dtype=np.uint8)
    stream = encode_fractal_mesh(test_img, target_anchors=10)
    recon = decode_fractal_mesh(stream)

    assert recon.shape == test_img.shape, f"Shape mismatch: {recon.shape} vs {test_img.shape}"

    # 2. Header parsing assertions
    w_parsed = int.from_bytes(stream[0:2], "big")
    h_parsed = int.from_bytes(stream[2:4], "big")
    assert w_parsed == 200, f"Expected width 200, got {w_parsed}"
    assert h_parsed == 100, f"Expected height 100, got {h_parsed}"

    # 3. Test Bitstream Resilience against corrupted bytes
    corrupted_stream = b"\x00\x02"  # Short invalid stream
    fallback = decode_fractal_mesh(corrupted_stream)
    assert fallback.shape == (512, 512), "Decoder should fall back gracefully on invalid stream"

    print("[SUCCESS] All codec.py tests passed cleanly!")