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
from GolombRice_codes import GolombRiceCodec


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
        invokes buffer encoder, applies Golomb-Rice and zlib compression.
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

        # Automatically detect if residuals are signed or unsigned
        encoded_bytes, total_bits, k = GolombRiceCodec.encode_adaptive(residuals)
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

            # Compute residuals distribution histogram (e.g., 32 bins)
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
    codec = Codec(target_anchors=150, quality_mode="fast", collect_stats=True)

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