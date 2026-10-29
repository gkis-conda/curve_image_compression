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

    if recommended_coder == "GolombRice":
        k = config.get("golomb_k", 0)
        encoded_bytes = GolombRiceCodec.encode_stream(residuals, k=k)
        total_bits = len(residuals) * (k + 1)
        return encoded_bytes, total_bits, k

    elif recommended_coder == "ExpGolomb":
        k = config.get("core_k", 2)
        return ExpGolombCodec.encode_stream(residuals, k=k)
    elif recommended_coder == "Raw":
        encoded_bytes = residuals.tobytes()
        total_bits = residuals.nbytes * 8
        return encoded_bytes, total_bits, 0
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
            "rice_m": 2,  # Default M parameter for Golomb-Rice coding
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
            self.config["splitter_threshold"] = threshold
            factory_config = estimator.get_codec_factory_config()
            self.config.update(factory_config)

        # Check if splitting should be active (threshold > 0 and distribution calls for a mixture tail)
        use_splitter = threshold > 0 and self.config.get("distribution_type", "mixture_tail") == "mixture_tail"
        self.config["use_splitter"] = use_splitter

        if use_splitter:
            # Instantiate our ResidualSplitter wrapper with the EM-calculated threshold
            splitter = ResidualSplitter(config=self.config)
            # Treat predictions as zeros for pure residual splitting stage
            predictions = np.zeros_like(residuals)
            reset_mask, smooth_residuals, reset_amplitudes = splitter.process_encoding(residuals, predictions)
            encode_stream(smooth_residuals, config=self.config)

            # B. Reset amplitudes (signed data stream) encoded adaptively
            amp_bytes, amp_bits, k_amp = ExpGolombCodec.encode_adaptive(reset_amplitudes)
            self.config["amp_rice_k"] = k_amp

            # C. Reset mask encoded compactly (packed bits into bytes)
            mask_packed = np.packbits(reset_mask)
            mask_len = len(reset_mask)

            # Pack all 3 substreams into a container payload structure:
            # [4 bytes: mask_len] + [4 bytes: mask_bytes_len] + [mask_bytes] +
            # [4 bytes: amp_bytes_len] + [amp_bytes] + [smooth_bytes]
            payload_container = bytearray()
            payload_container.extend(int(mask_len).to_bytes(4, 'little', signed=False))
            payload_container.extend(int(len(mask_packed)).to_bytes(4, 'little', signed=False))
            payload_container.extend(mask_packed)
            payload_container.extend(int(len(amp_bytes)).to_bytes(4, 'little', signed=False))
            payload_container.extend(amp_bytes)
            payload_container.extend(smooth_bytes)

            encoded_bytes = bytes(payload_container)
            total_bits = smooth_bits + amp_bits + len(mask_packed) * 8
        else:
            # Standard single stream mode (Pure Laplacian Core or Bypass)
            encoded_bytes, total_bits, k = ExpGolombCodec.encode_adaptive(residuals)
            self.config["rice_k"] = k

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
            original_bytes = img_data.nbytes
            total_compressed_size = len(header_bytes) + len(compressed_payload_bytes)
            num_symbols = len(residuals)

            hist_counts, hist_bins = np.histogram(residuals, bins=self.config.get("bins", 32))

            self.stats = {
                "total_symbols": num_symbols,
                "bits_per_symbol": total_bits / max(1, num_symbols),
                "original_bytes": original_bytes,
                "compressed_payload_bytes": len(compressed_payload_bytes),
                "total_stream_bytes": total_compressed_size,
                "compression_ratio": original_bytes / max(1, total_compressed_size),
                "residuals_histogram": {
                    "counts": hist_counts.tolist(),
                    "bins": hist_bins.tolist()
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
            offset += amp_bytes_len
            smooth_bytes = raw_payload_bytes[offset:]

            # Decode components
            k_smooth = self.config.get("rice_k", 2)
            restored_smooth = GolombRiceCodec.decode_stream(smooth_bytes, k=k_smooth, res_dtype=res_dtype)

            k_amp = self.config.get("amp_rice_k", 2)
            restored_amps = GolombRiceCodec.decode_stream(amp_bytes, k=k_amp, res_dtype=res_dtype)

            # Reconstruct using ResidualSplitter!
            splitter = ResidualSplitter(threshold=self.config.get("splitter_threshold", 32))
            dummy_predictions = np.zeros(mask_len, dtype=res_dtype)

            restored_residuals = splitter.process_decoding(
                predictions=dummy_predictions,
                reset_mask=reset_mask,
                smooth_residuals=restored_smooth,
                reset_amplitudes=restored_amps
            )
        else:
            k = self.config.get("rice_k", 0)
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