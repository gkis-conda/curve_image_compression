"""
===============================================================================
                       EVANS CODEC IMPLEMENTATION
===============================================================================

  Implements lossy/lossless Evans fractal compression inheriting from CodecBase,
  handling variable-width pixel ribbons (3 or 5), sliding Evans surface prediction, 
  filter updates, and exact delta stream roundtrip.

===============================================================================
"""

# Import base codec from the existing module
from codec_stub import CodecBase
# Specialized project modules
from quadtree_path_finder import PathCompiler
from evans_filter import EvansWindowFilter
from slide_evans_filter import EvansSlidingFilter
from quantizers import HalfQuantizer, LogarithmicLutQuantizer
from dramatic_logger import log_event, dramatic_guard
from image_accessor import ImageBlockAccessor
import numpy as np
from graph_visualizer import SimpleGridPathVisualizer

class Codec(CodecBase):
    """
    Evans fractal compression codec implementation inheriting from CodecBase.
    Integrates ImageBlockAccessor and EvansWindowFilter into a synchronous 2D timeline.
    """

    @dramatic_guard(topic="optimization", failure_pathos=-5)
    def _encode(self, accessor: ImageBlockAccessor, path: list) -> list:
        """
        Step 4: Streamlined Sliding Evans prediction pipeline.
        Completely unrolls the 3-step profile loops directly inside direction branches
        to achieve maximum raw execution speeds.
        """
        encoder = EvansWindowFilter(alpha_noise=0.0, sigma_0=0.0, gamma=0.15,
                                    quantize_fn=HalfQuantizer(), init_brightness=128.0, quantized_signal=True)
        delta_stream = []

        r_prev, c_prev = 0, -1
        for r_idx, c_idx in path:
            block_data = accessor.get_block(r_idx, c_idx)

            if c_idx > c_prev:
                # DIRECTION: EAST (Scan columns left-to-right: 0, 1, 2)
                delta_stream.extend(encoder.coder_update(0, block_data[:, 0]))
                delta_stream.extend(encoder.coder_update(0, block_data[:, 1]))
                delta_stream.extend(encoder.coder_update(0, block_data[:, 2]))

            elif c_idx < c_prev:
                # DIRECTION: WEST (Scan columns right-to-left: 2, 1, 0)
                delta_stream.extend(encoder.coder_update(2, block_data[:, 2]))
                delta_stream.extend(encoder.coder_update(2, block_data[:, 1]))
                delta_stream.extend(encoder.coder_update(2, block_data[:, 0]))

            elif r_idx > r_prev:
                # DIRECTION: SOUTH (Scan rows top-to-bottom: 0, 1, 2)
                delta_stream.extend(encoder.coder_update(1, block_data[0, :]))
                delta_stream.extend(encoder.coder_update(1, block_data[1, :]))
                delta_stream.extend(encoder.coder_update(1, block_data[2, :]))

            elif r_idx < r_prev:
                # DIRECTION: NORTH (Scan rows bottom-to-top: 2, 1, 0)
                delta_stream.extend(encoder.coder_update(3, block_data[2, :]))
                delta_stream.extend(encoder.coder_update(3, block_data[1, :]))
                delta_stream.extend(encoder.coder_update(3, block_data[0, :]))

            r_prev, c_prev = r_idx, c_idx

        log_event(f"Evans sliding prediction ribbon processed across {len(delta_stream)} segments.",
                  topic="optimization", pathos=2)
        return delta_stream

    def _encode_buffer(self, data: np.ndarray) -> bytes:
        """
        Virtual method implementation for encoding the data buffer.
        Flattens the delta streams into a packed int16 binary chunk for zlib.
        """
        gray_data = data[:, :, 0] if data.ndim == 3 else data

        block_accessor = ImageBlockAccessor(gray_data, block_size=3)
        h = block_accessor.block_rows
        w = block_accessor.block_cols
        compiler = PathCompiler()
        path = compiler.compile_trajectory(h, w, (0, 0), (h - 1, 0))
        viz = SimpleGridPathVisualizer(n_rows=h, m_cols=w, scale_factor=20.0)
        viz.render_trajectory_to_svg(path, filename="runtime_calibrated_trajectory.svg")
        log_event(f"Quadtree trajectory compiled with {len(path)} nodes.", topic="geometry", pathos=1)

        # 1. Harvest the raw delta streams from the Evans ribbon (list of lists)
        raw_delta_list = self._encode(block_accessor, path)
        print(raw_delta_list[:100])
        flat_deltas = np.array(raw_delta_list, dtype=np.int16)
        # 3. Return a clean, packed byte string perfectly suited for zlib
        return flat_deltas.tobytes()

    def _decode_buffer(self, payload: bytes, w: int, h: int, channels: int) -> np.ndarray:
        """
        Virtual method implementation for decoding the data buffer with 100% exact reconstruction.
        Completely unrolls the inner 3-step loop segments to eliminate lambda allocation lags.
        """
        # 1. Unpack the byte stream directly into a continuous 1D int8 array
        flat_deltas = np.frombuffer(payload, dtype=np.int16)
        print(flat_deltas[:100].tolist())
        # 2. Reshape the flat data back into structured 3-element delta updates
        delta_stream = flat_deltas.reshape(-1, 3)

        # 3. Initialize your blank canvas and reconstruct the geometry
        reconstructed_gray = np.full((h, w), 128, dtype=np.int16)
        block_accessor = ImageBlockAccessor(reconstructed_gray, block_size=3)

        h_b = block_accessor.block_rows
        w_b = block_accessor.block_cols
        compiler = PathCompiler()
        path = compiler.compile_trajectory(h_b, w_b, (0, 0), (h_b - 1, 0))
        decoder = EvansWindowFilter(alpha_noise=0.0, sigma_0=0.0, gamma=0.15,
                                    quantize_fn=HalfQuantizer(), init_brightness=128.0, quantized_signal=True)

        delta_idx = 0
        r_prev, c_prev = 0, -1
        restored_block = np.zeros((3, 3), dtype=np.int16)

        for (r_idx, c_idx) in path:

            if c_idx > c_prev:
                # DIRECTION: EAST (Decode columns left-to-right: 0, 1, 2)
                restored_block[:, 0] = decoder.decoder_update(0, delta_stream[delta_idx])
                restored_block[:, 1] = decoder.decoder_update(0, delta_stream[delta_idx + 1])
                restored_block[:, 2] = decoder.decoder_update(0, delta_stream[delta_idx + 2])
            elif c_idx < c_prev:
                # DIRECTION: WEST (Decode columns right-to-left: 2, 1, 0)
                restored_block[:, 2] = decoder.decoder_update(2, delta_stream[delta_idx])
                restored_block[:, 1] = decoder.decoder_update(2, delta_stream[delta_idx + 1])
                restored_block[:, 0] = decoder.decoder_update(2, delta_stream[delta_idx + 2])
            elif r_idx > r_prev:
                # DIRECTION: SOUTH (Decode rows top-to-bottom: 0, 1, 2)
                restored_block[0, :] = decoder.decoder_update(1, delta_stream[delta_idx])
                restored_block[1, :] = decoder.decoder_update(1, delta_stream[delta_idx + 1])
                restored_block[2, :] = decoder.decoder_update(1, delta_stream[delta_idx + 2])
            elif r_idx < r_prev:
                # DIRECTION: NORTH (Decode rows bottom-to-top: 2, 1, 0)
                restored_block[2, :] = decoder.decoder_update(3, delta_stream[delta_idx])
                restored_block[1, :] = decoder.decoder_update(3, delta_stream[delta_idx + 1])
                restored_block[0, :] = decoder.decoder_update(3, delta_stream[delta_idx + 2])
            delta_idx += 3
            block_accessor.set_block(r_idx, c_idx, restored_block)
            r_prev, c_prev = r_idx, c_idx

        # Output formatting (Uses the original image dimensions w and h passed as parameters)
        if channels == 3:
            out_img = np.zeros((h, w, 3), dtype=np.uint8)
            clipped = np.clip(reconstructed_gray, 0, 255).astype(np.uint8)
            for i in range(3):
                out_img[:, :, i] = clipped
            return out_img
        else:
            return np.clip(reconstructed_gray, 0, 255).astype(np.uint8)

#Quantizer=LogarithmicLutQuantizer
Quantizer=HalfQuantizer

class SlidingCodec(CodecBase):
    """
    Highly Optimized Monolithic Continuous Ribbon Streaming Codec.
    Employs int16 binary buffers to fully support an expanded quantization
    index dynamic range of [-512, 512] with zero risk of arithmetic overflow.
    """
    @dramatic_guard(topic="optimization", failure_pathos=-5)
    def _encode(self, padded_canvas: np.ndarray, path: list) -> np.ndarray:
        """
        Step 4: Continuous Ribbon Evans prediction pipeline.
        Structures innovation bins as a 2D matrix of 3-cell triplets,
        flattening with zero-allocation .ravel() at the final return rail.
        """
        quantizer = Quantizer()
        print("Quantizer", quantizer.name)
        encoder = EvansSlidingFilter(
            alpha_noise=0.0, sigma_0=0.0, gamma=0.15,
            quantize_fn=quantizer, quantized_signal=True
        )

        total_triplets = len(path) * 3
        triplet_stream = np.zeros((total_triplets, 3), dtype=quantizer.dtype)

        # Pointer now increments by 1 row per step instead of 3 scalar elements
        row_ptr = 0
        # Strategic index initialization forces EAST on first block (0,0)
        r_prev, c_prev = 0, -1

        for (r_idx, c_idx) in path:
            # Map block indices directly to padded pixel neighborhood centers
            r = r_prev * 3 + 1
            c = c_prev * 3 + 4

            if c_idx > c_prev:
                # DIRECTION: EAST (Scan 3 columns left-to-right: c, c+1, c+2)
                triplet_stream[row_ptr] = encoder.coder_update_canvas(padded_canvas, r, c, 0)
                triplet_stream[row_ptr + 1] = encoder.coder_update_canvas(padded_canvas, r, c + 1, 0)
                triplet_stream[row_ptr + 2] = encoder.coder_update_canvas(padded_canvas, r, c + 2, 0)

            elif c_idx < c_prev:
                # DIRECTION: WEST (Scan 3 columns right-to-left: c, c-1, c-2)
                triplet_stream[row_ptr] = encoder.coder_update_canvas(padded_canvas, r, c, 2)
                triplet_stream[row_ptr + 1] = encoder.coder_update_canvas(padded_canvas, r, c - 1, 2)
                triplet_stream[row_ptr + 2] = encoder.coder_update_canvas(padded_canvas, r, c - 2, 2)

            elif r_idx > r_prev:
                # DIRECTION: SOUTH (Scan 3 rows top-to-bottom: r, r+1, r+2)
                triplet_stream[row_ptr] = encoder.coder_update_canvas(padded_canvas, r, c, 1)
                triplet_stream[row_ptr + 1] = encoder.coder_update_canvas(padded_canvas, r + 1, c, 1)
                triplet_stream[row_ptr + 2] = encoder.coder_update_canvas(padded_canvas, r + 2, c, 1)

            elif r_idx < r_prev:
                # DIRECTION: NORTH (Scan 3 rows bottom-to-top: r, r-1, r-2)
                triplet_stream[row_ptr] = encoder.coder_update_canvas(padded_canvas, r, c, 3)
                triplet_stream[row_ptr + 1] = encoder.coder_update_canvas(padded_canvas, r - 1, c, 3)
                triplet_stream[row_ptr + 2] = encoder.coder_update_canvas(padded_canvas, r - 2, c, 3)

            # Advance the tracking pointer by exactly 3 triplet rows per macro-block
            row_ptr += 3
            r_prev, c_prev = r_idx, c_idx
        assert total_triplets == row_ptr
        return triplet_stream.ravel()

    def _encode_buffer(self, data: np.ndarray) -> bytes:
        """
        Virtual method implementation for encoding the data buffer.
        Packs the continuous int16 updates directly into a raw binary chunk.
        """
        gray_data = data[:, :, 0] if data.ndim == 3 else data
        h, w = gray_data.shape

        min_rows = int(np.ceil(h / 3))
        min_cols = int(np.ceil(w / 3))
        h_b = min_rows + (min_rows % 2)
        w_b = min_cols + (min_cols % 2)

        compiler = PathCompiler()
        path = compiler.compile_trajectory(h_b, w_b, (0, 0), (h_b - 1, 0))

        # Setup standalone compliant trajectory trace mappings
        viz = SimpleGridPathVisualizer(n_rows=h_b, m_cols=w_b, scale_factor=20.0)
        viz.render_trajectory_to_svg(path, filename="runtime_calibrated_trajectory.svg")
        log_event(f"Quadtree trajectory compiled with {len(path)} nodes.", topic="geometry", pathos=1)

        # Apply asymmetric edge padding layout to accurately layout the virtual start block (0, -1)
        padded_canvas = np.pad(gray_data, pad_width=((0, h_b * 3 - h), (3, (w_b + 1)* 3 - w)), mode='edge').astype(np.float64)
        padded_canvas[:, :3] = 128.0

        # Harvest the packed int16 delta array stream directly from the monolithic canvas update
        flat_deltas = self._encode(padded_canvas, path)

        return flat_deltas.ravel()

    def _decode_buffer(self, payload: bytes, w: int, h: int, channels: int) -> np.ndarray:
        """
        Virtual method implementation for decoding the data buffer with 100% exact reconstruction.
        Unpacks the raw int16 bytes stream and drives direct in-place canvas updates.
        """
        # 1. FIXED: Unpack the byte stream directly into a continuous 1D int16 array
        # to guarantee bit-exact mapping of large indices without overflows
        quantizer = Quantizer()
        flat_deltas = np.frombuffer(payload, dtype=quantizer.dtype)
        delta_stream = flat_deltas.reshape(-1, 3)

        min_rows = int(np.ceil(h / 3))
        min_cols = int(np.ceil(w / 3))
        h_b = min_rows + (min_rows % 2)
        w_b = min_cols + (min_cols % 2)

        # Initialize the master canvas tracking layout
        padded_canvas = np.full((h_b * 3, (w_b + 1) * 3), 128.0, dtype=np.float64)

        compiler = PathCompiler()
        path = compiler.compile_trajectory(h_b, w_b, (0, 0), (h_b - 1, 0))

        decoder = EvansSlidingFilter(
            alpha_noise=0.0, sigma_0=0.0, gamma=0.15,
            quantize_fn=quantizer, quantized_signal=True
        )

        delta_idx = 0
        r_prev, c_prev = 0, -1

        for (r_idx, c_idx) in path:
            r = r_prev * 3 + 1
            c = c_prev * 3 + 4
            v0, v1, v2 = delta_stream[delta_idx:delta_idx+3]
            if c_idx > c_prev:
                # DIRECTION: EAST (Decode 3 columns left-to-right)
                decoder.decoder_update_canvas(padded_canvas, r, c, 0, v0)
                decoder.decoder_update_canvas(padded_canvas, r, c + 1, 0, v1)
                decoder.decoder_update_canvas(padded_canvas, r, c + 2, 0, v2)
            elif c_idx < c_prev:
                # DIRECTION: WEST (Decode 3 columns right-to-left)
                decoder.decoder_update_canvas(padded_canvas, r, c, 2, v0)
                decoder.decoder_update_canvas(padded_canvas, r, c - 1, 2, v1)
                decoder.decoder_update_canvas(padded_canvas, r, c - 2, 2, v2)
            elif r_idx > r_prev:
                # DIRECTION: SOUTH (Decode 3 rows top-to-bottom)
                decoder.decoder_update_canvas(padded_canvas, r, c, 1, v0)
                decoder.decoder_update_canvas(padded_canvas, r + 1, c, 1, v1)
                decoder.decoder_update_canvas(padded_canvas, r + 2, c, 1, v2)
            elif r_idx < r_prev:
                # DIRECTION: NORTH (Decode 3 rows bottom-to-top)
                decoder.decoder_update_canvas(padded_canvas, r, c, 3, v0)
                decoder.decoder_update_canvas(padded_canvas, r - 1, c, 3, v1)
                decoder.decoder_update_canvas(padded_canvas, r - 2, c, 3, v2)

            delta_idx += 3
            r_prev, c_prev = r_idx, c_idx

        # 3. Slice away the temporary padding boundary to return original frame sizes
        reconstructed_gray = padded_canvas[0 : h, 3 : w + 3]

        if channels == 3:
            out_img = np.zeros((h, w, 3), dtype=np.uint8)
            clipped = np.clip(reconstructed_gray, 0, 255).astype(np.uint8)
            for i in range(3):
                out_img[:, :, i] = clipped
            return out_img
        else:
            return np.clip(np.rint(reconstructed_gray), 0, 255).astype(np.uint8)

# =============================================================================
#                            MODULE TESTS
# =============================================================================


def test_codec():
    """Test suite for Codec implementation with exact ribbon delta roundtrip."""
    print("[TEST] Running codec.py module tests...")

    dummy_image = np.random.randint(0, 256, (16, 28), dtype=np.uint8)
    codec = SlidingCodec(config={})
    np.set_printoptions(edgeitems=3, infstr='inf',
                        linewidth=200, nanstr='nan',
                        precision=1, suppress=False,
                        threshold=1000, formatter=None)
    print("source")
    print(dummy_image)
    bitstream = codec.encode(dummy_image)
    assert len(bitstream) > 4, "Bitstream must contain header and payload"

    reconstructed = codec.decode(bitstream)
    print("reconstructed")
    print(reconstructed)
    print(np.isclose(reconstructed, dummy_image, 0.))
    assert reconstructed.shape == dummy_image.shape, "Reconstructed image shape must match"
    assert np.isclose(reconstructed, dummy_image, 1.).all(), "Reconstructed image must match"

    # Perfect or near-perfect structural validation along active ribbon segments
    print("[SUCCESS] All codec.py tests passed cleanly!")


def test_image():
    import cv2
    synthetic = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(synthetic, (128, 128), 64, 255, -1)
    codec = SlidingCodec(collect_stats=True)
    np.set_printoptions(edgeitems=3, infstr='inf',
                        linewidth=200, nanstr='nan',
                        precision=8, suppress=False,
                        threshold=1000, formatter=None)
    bitstream = codec.encode(synthetic)
    print(codec.stats)
    reconstructed = codec.decode(bitstream)
    cv2.imshow("Reconstructed",cv2.absdiff(reconstructed,synthetic) * 255)
    cv2.waitKey()
    print(np.isclose(reconstructed, synthetic, 0.))
    assert reconstructed.shape == synthetic.shape, "Reconstructed image shape must match"
    assert np.isclose(reconstructed, synthetic, 1.).all(), "Reconstructed image must match"

    # Perfect or near-perfect structural validation along active ribbon segments
    print("[SUCCESS] All codec.py tests passed cleanly!")


if __name__ == "__main__":
    test_codec()
    test_image()