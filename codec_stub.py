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

import numpy as np
import json

class Codec:

    def __init__(self, **config):
        """
        Initializes the codec with base configuration including spatial dims
        and channels, updated by user-provided parameters.
        """
        self.config = {
            "channels": 1,
            "target_anchors": 300,
        }
        self.config.update(config)

    def update_config(self, **kwargs):
        """
        Dynamically updates codec tuning parameters on the fly.
        """
        self.config.update(kwargs)

    def _encode_buffer(self, data: np.ndarray) -> bytes:
        """
        Virtual method for encoding the data buffer. 
        Subclasses can override this to implement custom topological/fractal compression.
        """
        return data.tobytes()

    def _decode_buffer(self, payload: bytes, w: int, h: int, channels: int) -> np.ndarray:
        """
        Virtual method for decoding the data buffer.
        Subclasses can override this to reconstruct data from custom bitstreams.
        """
        shape = (h, w) if channels == 1 else (h, w, channels)
        return np.frombuffer(payload, dtype=np.uint8).reshape(shape)

    def encode(self, img_data: np.ndarray) -> bytes:
        """
        Template method: extracts dimensions and channels, serializes config,
        and invokes the virtual buffer encoder.
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

        metadata = {
            "config": self.config
        }
        
        meta_json = json.dumps(metadata)
        header = f"FRACTAL_MESH_V3\n{meta_json}\n"
        
        header_bytes = header.encode("ascii")
        payload_bytes = self._encode_buffer(img_data)
        
        return header_bytes + payload_bytes

    def decode(self, bitstream: bytes) -> np.ndarray:
        """
        Template method: parses stream header, extracts configuration,
        applies settings, and invokes the virtual buffer decoder.
        """
        parts = bitstream.split(b"\n", 2)
        if len(parts) < 2:
            raise ValueError("Error: Malformed bitstream structure.")

        version_line = parts[0].decode("ascii")
        meta_json = parts[1].decode("ascii")
        payload_bytes = parts[2] if len(parts) > 2 else b""

        if version_line == "FRACTAL_MESH_V3":
            metadata = json.loads(meta_json)
            stream_config = metadata.get("config", {})
            self.update_config(**stream_config)
        else:
            raise ValueError(f"Unsupported bitstream version: {version_line}")

        w = self.config.get("w")
        h = self.config.get("h")
        channels = self.config.get("channels", 1)

        return self._decode_buffer(payload_bytes, w, h, channels)


# ===============================================================================
#                               MINI-TEST SUITE
# ===============================================================================

def test_codec():
    """
    Validates codec round-trip consistency, types, and spatial dimensions.
    """
    print("Running Codec mini-test...")

    # Instantiate codec with custom configuration bundle
    codec = Codec(target_anchors=150, quality_mode="fast")

    # Create dummy 2D grayscale raster
    h, w = 32, 48
    original_img = np.random.randint(0, 256, size=(h, w), dtype=np.uint8)

    # Encode via codec instance
    bitstream = codec.encode(original_img)
    assert isinstance(bitstream, bytes), "Error: Encoded output must be bytes"
    assert len(bitstream) > 0, "Error: Bitstream cannot be empty"

    # Decode via codec instance
    reconstructed_img = codec.decode(bitstream)
    assert isinstance(reconstructed_img, np.ndarray), "Error: Decoded output must be np.ndarray"
    assert reconstructed_img.shape == (h, w), f"Error: Shape mismatch {reconstructed_img.shape} vs {(h, w)}"
    assert reconstructed_img.dtype == np.uint8, "Error: Dtype must be uint8"
    assert np.array_equal(original_img, reconstructed_img), "Error: Roundtrip data corruption detected"

    print("Codec mini-test passed successfully!")


if __name__ == "__main__":
    test_codec()