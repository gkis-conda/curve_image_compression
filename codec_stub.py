"""
===============================================================================
                    FRACTAL TOPOLOGICAL CODEC (INTERFACE)
===============================================================================

  "Where raw pixels dissolve into light,
   And facets sculpt geometry out of night."

  This module implements a polymorphic self-describing codec architecture 
  using the Template Method pattern for extensible buffer encoding/decoding.

===============================================================================
"""

import json
import zlib
import numpy as np
from GolombRice_codes import GolombRiceCodec, ExpGolombCodec
from residuals_splitter import ResidualSplitter, UniformTailEstimator


def encode_stream(residuals: np.ndarray, config: dict) -> tuple:
    """
    Factory function: selects and executes the optimal entropy coder
    based on the configuration profile produced by the UniformTailEstimator.
    """
    recommended_coder = config.get("recommended_coder", "ExpGolomb")
    k = config.get("golomb_k", 0)
    if recommended_coder == "GolombRice":
        return GolombRiceCodec.encode_stream(residuals, k=k)
    elif recommended_coder == "ExpGolomb":
        return ExpGolombCodec.encode_stream(residuals, k=k)
    elif recommended_coder == "Raw":
        encoded_bytes = residuals.tobytes()
        total_bits = residuals.nbytes * 8
        return encoded_bytes, total_bits
    else:
        raise Exception("Unknown codec")


def decode_stream(residuals: np.ndarray, config: dict) -> tuple:
    """
    Factory function: selects and executes the optimal entropy coder
    based on the configuration profile produced by the UniformTailEstimator.
    """
    recommended_coder = config.get("recommended_coder", "ExpGolomb")
    dtype = np.dtype(config.get("residuals_dtype", "int16"))
    k = config.get("golomb_k", 0)
    if recommended_coder == "GolombRice":
        return GolombRiceCodec.decode_stream(residuals, k=k, dtype=dtype)
    elif recommended_coder == "ExpGolomb":
        return ExpGolombCodec.decode_stream(residuals, k=k, dtype=dtype)
    elif recommended_coder == "Raw":
        return GolombRiceCodec.decode_stream(residuals, k=0, dtype=dtype)
    else:
        raise Exception("Unknown codec")


class CodecBase:

    def __init__(self, **config):
        """
        Initializes the codec with base configuration including spatial dims,
        channels, and metrics collection flag, updated by user-provided parameters.
        """
        self.config = {
            "channels": 1,
            "collect_stats": False,  # Flag to enable/disable statistics collection
        }
        self.config.update(config)
        self.stats = {}  # Dictionary to store collected performance metrics

    def update_config(self, **kwargs):
        """
        Dynamically updates codec tuning parameters on the fly.
        """
        self.config.update(kwargs)

    def encode(self, img_data: np.ndarray) -> bytes:
        """
        Template method: extracts dimensions and channels, serializes config,
        invokes buffer encoder, applies EM-based UniformTailEstimator ('microscope'),
        and routes through the optimal entropy compression path.
        """
        if not isinstance(img_data, np.ndarray):
            img_data = np.array(img_data, dtype=np.uint8)

        if img_data.ndim == 2:
            h, w = img_data.shape
            channels = 1
        else:
            h, w, channels = img_data.shape

        self.config["w"] = w
        self.config["h"] = h
        self.config["channels"] = channels

        # 1. Obtain residuals from the subclass buffer encoder
        residuals = self._encode_buffer(img_data)
        self.config["residuals_dtype"] = str(residuals.dtype)

        # 2. Use our Microscope (UniformTailEstimator) to analyze residuals distribution
        threshold = self.config.get("splitter_threshold", -1)

        if threshold <= 0:
            # Instantiate and run the EM-based MLE estimator
            estimator = UniformTailEstimator(tail_min=self.config.get("tail_min", 32))
            estimator.fit(residuals)
            threshold = estimator.threshold
            factory_config = estimator.get_codec_factory_config()
            self.config.update(factory_config)

        # Check if splitting should be active (threshold > 0 and distribution calls for a mixture tail)
        use_splitter = self.config.get("distribution_type") == "mixture_tail"
        self.config["use_splitter"] = use_splitter

        if use_splitter:
            # Instantiate our ResidualSplitter wrapper with the EM-calculated threshold
            splitter = ResidualSplitter(threshold)
            # Treat predictions as zeros for pure residual splitting stage
            reset_mask, smooth_residuals, reset_residuals = splitter.split(residuals)

            # Pack all 3 substreams into a container payload structure:
            # [4 bytes: mask_len] + [4 bytes: mask_bytes_len] + [mask_bytes] +
            # [4 bytes: amp_bytes_len] + [amp_bytes] + [smooth_bytes]
            payload_container = bytearray()
            # Reset mask encoded compactly (packed bits into bytes)
            mask_packed = np.packbits(reset_mask)
            mask_bits = len(mask_packed) * 8
            payload_container.extend(int(len(reset_mask)).to_bytes(4, 'little', signed=False))
            payload_container.extend(int(len(mask_packed)).to_bytes(4, 'little', signed=False))
            payload_container.extend(mask_packed)
            # reset_amplitudes
            reset_amplitudes_bytes = reset_residuals.tobytes()
            amp_bits = len(reset_amplitudes_bytes) * 8
            payload_container.extend(int(len(reset_amplitudes_bytes)).to_bytes(4, 'little', signed=False))
            payload_container.extend(reset_amplitudes_bytes)
            # smooth
            smooth_bytes, smooth_bits = encode_stream(smooth_residuals, config=self.config)
            payload_container.extend(smooth_bytes)

            encoded_bytes = bytes(payload_container)
            total_bits = smooth_bits + amp_bits + mask_bits
        else:
            # Standard single stream mode (Pure Laplacian Core or Bypass)
            encoded_bytes, total_bits = GolombRiceCodec.encode_stream(residuals, k=self.config["golomb_k"])

        metadata = {
            "config": self.config
        }
        meta_json = json.dumps(metadata)
        header = f"FRACTAL_MESH_V3\n{meta_json}\n"
        header_bytes = header.encode("ascii")

        # 3. Final compression using zlib
        compressed_payload_bytes = zlib.compress(encoded_bytes, level=9)

        # 4. Collect performance statistics and residuals histogram if enabled
        if self.config.get("collect_stats", False):
            # --- STATISTICS: THEORETICAL (SHANNON) VS ACTUAL COMPRESSION ---
            orig_flat = img_data.flatten()
            orig_symbols = orig_flat.size
            _, orig_counts = np.unique(orig_flat, return_counts=True)
            orig_probs = orig_counts.astype(np.float64) / max(1, orig_symbols)
            original_entropy = -np.sum(orig_probs * np.log2(orig_probs + 1e-12))

            num_symbols = residuals.size
            bins_count = self.config.get("bins", 64)
            hist_counts, hist_bins = np.histogram(residuals, bins=bins_count)
            probs = hist_counts / max(1, num_symbols)
            # bin width enthropy correction
            bin_width = (hist_bins[-1] - hist_bins[0]) / len(hist_counts)
            shannon_entropy = -np.sum(probs * np.log2(probs + 1e-12)) #+ np.log2(max(1.0, bin_width))

            theoretical_bits = shannon_entropy * num_symbols
            theoretical_bytes = int(np.ceil(theoretical_bits / 8.0))

            original_bytes = num_symbols * img_data.itemsize
            actual_payload_bytes = len(compressed_payload_bytes)
            total_compressed_size = len(header_bytes) + actual_payload_bytes
            actual_bits_per_symbol = (actual_payload_bytes * 8.0) / max(1, num_symbols)

            self.stats = {
                "total_symbols": int(num_symbols),

                "original_entropy": original_entropy * orig_symbols,
                "original_bits_per_symbol": original_entropy,

                "theoretical_entropy": float(shannon_entropy),
                "theoretical_payload_bytes": int(theoretical_bytes),
                "theoretical_compression_ratio": float(original_bytes / max(1, theoretical_bytes)),

                # --- ACTUAL PERFORMANCE ---
                "actual_bits_per_symbol": float(actual_bits_per_symbol),
                "original_bytes": int(original_bytes),
                "compressed_payload_bytes": int(actual_payload_bytes),
                "total_stream_bytes": int(total_compressed_size),
                "compression_ratio": float(original_bytes / max(1, total_compressed_size)),

                "residuals_histogram": {
                    "counts": hist_counts.tolist(),
                    "bin_width": bin_width
                }
            }
        return header_bytes + compressed_payload_bytes

    def decode(self, bitstream: bytes) -> np.ndarray:
        """
        Template method: parses stream header, extracts configuration,
        applies settings, and invokes the virtual buffer decoder.
        """
        first_newline = bitstream.find(b"\n")
        if first_newline == -1:
            raise ValueError("Error: Malformed bitstream structure - missing version line.")

        second_newline = bitstream.find(b"\n", first_newline + 1)
        if second_newline == -1:
            raise ValueError("Error: Malformed bitstream structure - missing metadata line.")

        version_line = bitstream[:first_newline].decode("ascii")
        meta_json = bitstream[first_newline + 1: second_newline].decode("ascii")

        compressed_payload_bytes = bitstream[second_newline + 1:]

        if version_line == "FRACTAL_MESH_V3":
            metadata = json.loads(meta_json)
            stream_config = metadata.get("config", {})
            self.update_config(**stream_config)
        else:
            raise ValueError(f"Unsupported bitstream version: {version_line}")

        w = self.config.get("w")
        h = self.config.get("h")
        channels = self.config.get("channels", 1)

        raw_payload_bytes = zlib.decompress(compressed_payload_bytes)
        res_dtype = np.dtype(self.config.get("residuals_dtype", "int16"))
        use_splitter = self.config.get("use_splitter", False)

        if use_splitter:
            # Unpack container payload
            offset = 0
            mask_len = int.from_bytes(raw_payload_bytes[offset:offset + 4], 'little', signed=False)
            offset += 4
            mask_bytes_len = int.from_bytes(raw_payload_bytes[offset:offset + 4], 'little', signed=False)
            offset += 4

            mask_packed = raw_payload_bytes[offset:offset + mask_bytes_len]
            offset += mask_bytes_len
            reset_mask = np.unpackbits(np.frombuffer(mask_packed, dtype=np.uint8))[:mask_len].astype(bool)

            amp_bytes_len = int.from_bytes(raw_payload_bytes[offset:offset + 4], 'little', signed=False)
            offset += 4

            amp_bytes = raw_payload_bytes[offset:offset + amp_bytes_len]
            res_dtype = np.dtype(self.config.get("residuals_dtype", "int16"))
            decoded_resets = np.frombuffer(amp_bytes, dtype=res_dtype)

            offset += amp_bytes_len
            smooth_bytes = raw_payload_bytes[offset:]
            smooth_residuals = decode_stream(smooth_bytes, self.config)

            # Reconstruct using ResidualSplitter!
            splitter = ResidualSplitter(-1)
            restored_residuals = splitter.merge(
                reset_mask=reset_mask,
                smooth_residuals=smooth_residuals,
                reset_amplitudes=decoded_resets
            )
        else:
            k = self.config.get("golomb_k", 0)
            restored_residuals = GolombRiceCodec.decode_stream(raw_payload_bytes, k=k, res_dtype=res_dtype)

        return self._decode_buffer(restored_residuals, w, h, channels)


class Codec(CodecBase):
    def _encode_buffer(self, data: np.ndarray) -> bytes:
        """
        Virtual method for encoding the data buffer. 
        Subclasses can override this to implement custom topological/fractal compression.
        """
        return data.ravel()

    def _decode_buffer(self, payload: np.ndarray, w: int, h: int, channels: int) -> np.ndarray:
        """
        Virtual method for decoding the data buffer.
        Subclasses can override this to reconstruct data from custom bitstreams.
        """
        shape = (h, w) if channels == 1 else (h, w, channels)
        return payload.astype(np.uint8).reshape(shape)


# ===============================================================================
#                               MINI-TEST SUITE
# ===============================================================================

def test_codec():
    """
    Validates codec round-trip consistency, types, and spatial dimensions.
    """
    print("Running Codec mini-test...")

    # Instantiate codec with custom configuration bundle
    codec = Codec(quality_mode="fast", collect_stats=True)

    # Create dummy 2D grayscale raster
    h, w = 32, 48
    original_img = np.random.randint(0, 256, size=(h, w), dtype=np.uint8)

    # Encode via codec instance
    bitstream = codec.encode(original_img)
    assert isinstance(bitstream, bytes), "Error: Encoded output must be bytes"
    assert len(bitstream) > 0, "Error: Bitstream cannot be empty"

    # Decode via codec instance
    reconstructed_img = codec.decode(bitstream)
    print(codec.stats)
    assert isinstance(reconstructed_img, np.ndarray), "Error: Decoded output must be np.ndarray"
    assert reconstructed_img.shape == (h, w), f"Error: Shape mismatch {reconstructed_img.shape} vs {(h, w)}"
    assert reconstructed_img.dtype == np.uint8, "Error: Dtype must be uint8"
    assert np.array_equal(original_img, reconstructed_img), "Error: Roundtrip data corruption detected"

    print("Codec mini-test passed successfully!")


if __name__ == "__main__":
    test_codec()