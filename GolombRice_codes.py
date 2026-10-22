import numpy as np


class BitWriter:
    """Helper to write bits sequentially into a compact byte buffer."""

    def __init__(self):
        self.buffer = bytearray()
        self.current_byte = 0
        self.bit_count = 0

    def write_bit(self, bit: int):
        self.current_byte = (self.current_byte << 1) | (bit & 1)
        self.bit_count += 1
        if self.bit_count == 8:
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.bit_count = 0

    def write_bits(self, val: int, num_bits: int):
        for i in range(num_bits - 1, -1, -1):
            self.write_bit((val >> i) & 1)

    def flush(self) -> tuple:
        total_bits = len(self.buffer) * 8 + self.bit_count
        if self.bit_count > 0:
            self.current_byte <<= (8 - self.bit_count)
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.bit_count = 0
        return bytes(self.buffer), total_bits


class BitReader:
    """High-performance bit reader using a 32-bit integer buffer (MSB first)."""

    def __init__(self, data: bytes):
        self.data = data
        self.byte_idx = 0
        self.bit_buffer = 0
        self.bits_left = 0
        self.length = len(data)

    def _refill(self) -> None:
        """Loads bytes into the 32-bit buffer while there is space and data."""
        while self.bits_left <= 24 and self.byte_idx < self.length:
            self.bit_buffer = (self.bit_buffer << 8) | self.data[self.byte_idx]
            self.bits_left += 8
            self.byte_idx += 1

    def read_bits(self, num_bits: int) -> int:
        """Reads a fixed number of bits from the buffer."""
        if num_bits == 0:
            return 0

        while self.bits_left < num_bits:
            if self.byte_idx >= self.length:
                raise IndexError("End of bit stream reached.")
            self.bit_buffer = (self.bit_buffer << 8) | self.data[self.byte_idx]
            self.bits_left += 8
            self.byte_idx += 1

        self.bits_left -= num_bits
        val = (self.bit_buffer >> self.bits_left) & ((1 << num_bits) - 1)
        return val

    def count_zeros(self) -> int:
        """
        Reads unary code (q zeros followed by '1') instantly
        using 32-bit buffer and native bit_length() (O(1) time).
        """
        q = 0
        while True:
            # Ensure we have enough bits loaded in the buffer
            if self.bits_left < 8:
                self._refill()

            if self.bits_left == 0:
                raise IndexError("End of bit stream reached.")

            # Mask current available bits in the buffer
            mask = (1 << self.bits_left) - 1
            valid_bits = self.bit_buffer & mask

            # If all available bits in the buffer are zero, consume them all and add to q
            if valid_bits == 0:
                q += self.bits_left
                self.bit_buffer = 0
                self.bits_left = 0
                continue

            # Fast leading zeros calculation using native C-optimized bit_length()
            # Distance from bits_left to bit_length gives exact number of leading zeros (q)
            leading_zeros = self.bits_left - valid_bits.bit_length()
            q += leading_zeros

            # Consume the zeros and the terminating '1' bit
            consumed = leading_zeros + 1
            self.bits_left -= consumed
            return q


class GolombRiceCodec:
    """
    Golomb-Rice entropy coder/decoder with built-in support for
    both one-sided (unsigned) and two-sided (signed +/-) data streams.
    """

    @staticmethod
    def _signed_to_unsigned(arr: np.ndarray) -> np.ndarray:
        """
        Interleaves signed integers into non-negative integers:
        0 -> 0
        -1 -> 1, 1 -> 2
        -2 -> 3, 2 -> 4, etc.
        """
        arr = np.asarray(arr, dtype=np.int32)
        return np.where(arr >= 0, 2 * arr, -2 * arr - 1)

    @staticmethod
    def _unsigned_to_signed(arr: np.ndarray) -> np.ndarray:
        """Reverses the interleaving back to signed integers."""
        arr = np.asarray(arr, dtype=np.int32)
        is_even = (arr % 2) == 0
        return np.where(is_even, arr // 2, -(arr + 1) // 2)

    @classmethod
    def encode_stream(cls, values, k: int = 2, signed: bool = True) -> tuple:
        """
        Packs integers into a Golomb-Rice bit stream.

        Args:
            values: Input data (signed or unsigned).
            M: Rice parameter (chunk size 2^M).
            signed: If True, encodes +/- values by interleaving them into non-negative range.
        """
        if signed:
            unsigned_vals = cls._signed_to_unsigned(values)
        else:
            unsigned_vals = values

        writer = BitWriter()
        divisor = 1 << k

        q_vals = unsigned_vals >> k  # Quotient
        r_vals = unsigned_vals & (divisor - 1)  # Remainder
        # Unary encoding of quotient q: q zeros followed by '1'
        for q,r in zip(q_vals,r_vals):
            writer.write_bits(1, q+1)
            # Fixed-length binary encoding of remainder r
            writer.write_bits(r, k)

        return writer.flush()

    @staticmethod
    def encode_adaptive(residuals: np.ndarray):
        if residuals.size == 0:
            return residuals.tobytes(), 0, -1

        is_signed = np.issubdtype(residuals.dtype, np.signedinteger)
        mean_abs = np.mean(np.abs(residuals))

        if mean_abs > 0:
            optimal_m_float = mean_abs * 0.693147
            k = int(np.round(np.log2(optimal_m_float)))
        else:
            k = -1

        if 0 < k < 4:
            encoded_bytes, total_bits = GolombRiceCodec.encode_stream(residuals, k=k, signed=is_signed)
        else:
            encoded_bytes = residuals.tobytes()
            total_bits = len(encoded_bytes) * 8

        return encoded_bytes, total_bits, k

    @classmethod
    def decode_stream(cls, encoded_bytes: bytes, k: int = 2, res_dtype=np.int16) -> np.ndarray:
        """
        Unpacks and decodes a Golomb-Rice bit stream back into integers
        dynamically until the end of the buffer is reached.
        """

        if not (0 < k < 4):
            return np.frombuffer(encoded_bytes, dtype=res_dtype)

        reader = BitReader(encoded_bytes)
        unsigned_vals = []
        while True:
            try:
                # 1. Read unary quotient q (count leading zeros until '1')
                q = reader.count_zeros()
                # 2. Read M bits for remainder r
                r = reader.read_bits(k)
                # 3. Reconstruct value: val = q * 2^M + r
                val = (q << k) + r
                unsigned_vals.append(val)
            except IndexError:
                break

        if np.issubdtype(res_dtype, np.signedinteger):
            res = cls._unsigned_to_signed(unsigned_vals)
        else:
            res = unsigned_vals
        return  np.array(res, dtype=res_dtype)



from scipy.stats import laplace


def run_codec_tests():
    print("=" * 60)
    print("RUNNING GOLOMB-RICE CODEC & ENTROPY TESTS")
    print("=" * 60)

    np.random.seed(42)
    sample_size = 10000
    loc, scale = 0.0, 3.5

    raw_samples = laplace.rvs(loc=loc, scale=scale, size=sample_size)
    signed_data = np.round(raw_samples).astype(np.int32)

    unsigned_data = np.abs(signed_data)

    print("\n--- TEST 1: Two-Sided Signed Mode (+ / -) ---")
    M_signed = 3
    encoded_bytes_s, total_bits_s = GolombRiceCodec.encode_stream(signed_data, k=M_signed, signed=True)
    decoded_data_s = GolombRiceCodec.decode_stream(encoded_bytes_s, k=M_signed, res_dtype=signed_data.dtype)

    assert np.array_equal(signed_data, decoded_data_s), "[FAIL] Signed mode reconstruction mismatch!"
    print("[PASS] Signed mode bit-accurate reconstruction verified.")

    print("\n--- TEST 2: One-Sided Unsigned Mode ---")
    M_unsigned = 2
    encoded_bytes_u, total_bits_u = GolombRiceCodec.encode_stream(unsigned_data, k=M_unsigned, signed=False)
    decoded_data_u = GolombRiceCodec.decode_stream(encoded_bytes_u, k=M_unsigned, res_dtype=np.uint16)

    assert np.array_equal(unsigned_data, decoded_data_u), "[FAIL] Unsigned mode reconstruction mismatch!"
    print("[PASS] Unsigned mode bit-accurate reconstruction verified.")

    print("\n--- TEST 3: Entropy & Compression Statistics ---")

    _, counts_s = np.unique(signed_data, return_counts=True)
    probs_s = counts_s / len(signed_data)
    shannon_entropy_s = -np.sum(probs_s * np.log2(probs_s))

    avg_bits_s = total_bits_s / sample_size
    print(f"Signed Data:")
    print(f"  - Shannon Entropy: {shannon_entropy_s:.3f} bits/symbol")
    print(f"  - Golomb-Rice Bits: {avg_bits_s:.3f} bits/symbol")
    print(f"  - Total Stream Size: {len(encoded_bytes_s)} bytes for {sample_size} symbols")

    _, counts_u = np.unique(unsigned_data, return_counts=True)
    probs_u = counts_u / len(unsigned_data)
    shannon_entropy_u = -np.sum(probs_u * np.log2(probs_u))

    avg_bits_u = total_bits_u / sample_size
    print(f"\nUnsigned Data (Absolute):")
    print(f"  - Shannon Entropy: {shannon_entropy_u:.3f} bits/symbol")
    print(f"  - Golomb-Rice Bits: {avg_bits_u:.3f} bits/symbol")
    print(f"  - Total Stream Size: {len(encoded_bytes_u)} bytes for {sample_size} symbols")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_codec_tests()