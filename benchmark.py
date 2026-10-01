"""
===============================================================================
                    BENCHMARK & PERFORMANCE EVALUATION
===============================================================================

  "We weigh the shadow against the sun,
   To measure what the mesh has won."

  A standalone harness that loads image manifolds, measures encoding and
  decoding execution latency, calculates fidelity metrics (PSNR, SSIM),
  and computes bitstream compression efficiency.

===============================================================================
"""
"""
===============================================================================
              FRACTAL CODEC INTEGRATION BENCHMARK
===============================================================================

  "Our revels now are ended. These our actors,
   As I foretold you, were all spirits..."
  -- William Shakespeare, The Tempest

  Complete pipeline: image loading, encoding, decoding, and benchmarking.
  Imports encode_fractal_mesh and decode_fractal_mesh from implementation.
  Calculates PSNR, SSIM, BPP, and latency, logging status via dramatic logger.

===============================================================================
"""

import os
import sys
import time
import math
import numpy as np
import cv2

from dramatic_logger import log_event

# -----------------------------------------------------------------------------
# DYNAMIC IMPLEMENTATION IMPORT FROM CODEC / STUB MODULE
# -----------------------------------------------------------------------------
try:
    from codec import encode_fractal_mesh, decode_fractal_mesh
except ImportError:
    try:
        from codec_stub import encode_fractal_mesh, decode_fractal_mesh
    except ImportError:
        encode_fractal_mesh = None
        decode_fractal_mesh = None


def calculate_psnr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) between images."""
    mse = np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def calculate_ssim(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Calculates Structural Similarity Index Measure (SSIM) between images."""
    img1 = original.astype(np.float64)
    img2 = reconstructed.astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))

    mu1 = cv2.filter2D(img1, -1, kernel)
    mu2 = cv2.filter2D(img2, -1, kernel)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, kernel) - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, kernel) - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, kernel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(np.mean(ssim_map))


def psnr_and_ssim_to_pathos(psnr: float, ssim: float) -> int:
    """Maps combined PSNR and SSIM quality metrics to Pathos level (-5 to +5)."""
    if math.isinf(psnr) or (psnr >= 38.0 and ssim >= 0.98):
        return 5
    elif psnr >= 32.0 and ssim >= 0.92:
        return 3
    elif psnr >= 28.0 and ssim >= 0.85:
        return 1
    elif psnr >= 24.0 and ssim >= 0.75:
        return 0
    elif psnr >= 18.0 and ssim >= 0.60:
        return -1
    elif psnr >= 12.0 and ssim >= 0.40:
        return -3
    else:
        return -5


def run_benchmark(image_path: str, target_anchors: int = 300):
    """
    Executes complete pipeline: loading, encoding, decoding, and benchmarking.
    """
    # Guard check for missing implementation
    if encode_fractal_mesh is None or decode_fractal_mesh is None:
        log_event("Codec Implementation Import Failure", topic="file_io", pathos=-5)
        return

    # Guard check for missing file
    if not os.path.exists(image_path):
        log_event(f"File Non-Existent: '{image_path}'", topic="file_io", pathos=-5)
        return

    # Load input manifold as single-channel grayscale
    img_orig = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_orig is None:
        log_event(f"Canvas Decode Failure: '{image_path}'", topic="file_io", pathos=-5)
        return

    h, w = img_orig.shape
    raw_size_bytes = img_orig.nbytes

    log_event(
        f"Processing Canvas: {os.path.basename(image_path)} ({w}x{h} px) | "
        f"Raw: {raw_size_bytes / 1024.0:.2f} KB | Anchors Target: {target_anchors}",
        topic="file_io",
        pathos=1,
    )

    # Encode domain surface into fractal stream
    t0 = time.perf_counter()
    bitstream = encode_fractal_mesh(img_orig, target_anchors=target_anchors)
    t_enc = (time.perf_counter() - t0) * 1000.0

    compressed_size_bytes = len(bitstream)
    bpp = (compressed_size_bytes * 8.0) / (w * h)
    compression_ratio = (
        raw_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else 0.0
    )

    # Decode fractal stream back into spatial manifold
    t0 = time.perf_counter()
    img_recon = decode_fractal_mesh(bitstream)
    t_dec = (time.perf_counter() - t0) * 1000.0

    # Compute reconstruction fidelity metrics
    psnr_val = calculate_psnr(img_orig, img_recon)
    ssim_val = calculate_ssim(img_orig, img_recon)
    pathos_val = psnr_and_ssim_to_pathos(psnr_val, ssim_val)

    log_event(
        f"Compression Completed: {compressed_size_bytes} B ({compressed_size_bytes / 1024.0:.2f} KB) | "
        f"Ratio: {compression_ratio:.2f}:1 | Density: {bpp:.4f} bpp",
        topic="bitstream",
        pathos=0,
    )

    log_event(
        f"Benchmark Metrics: PSNR={psnr_val:.2f} dB | SSIM={ssim_val:.4f} | "
        f"Enc Latency={t_enc:.2f} ms | Dec Latency={t_dec:.2f} ms",
        topic="psnr_bench",
        pathos=pathos_val,
    )


if __name__ == "__main__":
    # Check CLI arguments or fall back to synthetic canvas
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
    else:
        test_path = "test_input.png"
        if not os.path.exists(test_path):
            synthetic = np.zeros((512, 512), dtype=np.uint8)
            cv2.circle(synthetic, (256, 256), 128, 255, -1)
            cv2.imwrite(test_path, synthetic)
            log_event(f"Synthetic Target Created: '{test_path}'", topic="file_io", pathos=0)

    run_benchmark(test_path, target_anchors=300)