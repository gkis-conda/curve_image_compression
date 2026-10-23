import numpy as np


class BitWriter:
    """Helper to write bits sequentially into a compact byte buffer."""

    def __init__(self):
        self.buffer = bytearray()
        self.current_byte = np.int64(0)
        self.bit_count = 0

    def write_bits(self, val: int, num_bits: int):
        #assert num_bits <= 32
        #assert val >= 0
        if num_bits <= 0:
            return

        self.current_byte = (val << self.bit_count) | self.current_byte
        #assert self.current_byte >= 0
        self.bit_count += num_bits

        while self.bit_count >= 8:
            self.buffer.append(self.current_byte & 0xFF)
            self.current_byte >>= 8
            self.bit_count -= 8

    def flush(self) -> tuple:
        total_bits = len(self.buffer) * 8 + self.bit_count
        if self.bit_count > 0:
            padding_bits = 8 - self.bit_count
            self.write_bits(0, padding_bits)

        return bytes(self.buffer), total_bits


from numba import njit, uint32, int32

@njit
def fast_encode_loop(q_vals, r_vals, k):
    buffer = np.zeros(len(q_vals) * 4, dtype=np.uint8)
    byte_idx = 0
    current_byte = np.uint64(0)
    bit_count = 0

    for i in range(len(q_vals)):
        q = np.uint64(q_vals[i])
        r = np.uint64(r_vals[i])

        while q >= 23:
            bit_count += 23
            while bit_count >= 8:
                buffer[byte_idx] = current_byte & 0xFF
                byte_idx += 1
                current_byte >>= 8
                bit_count -= 8
            q -= np.uint64(23)

        combined_code = (np.uint64(1) | r << 1) << q
        code_len = q + 1 + k
        current_byte |= (combined_code << bit_count)
        bit_count += code_len

        while bit_count >= 8:
            buffer[byte_idx] = current_byte & 0xFF
            byte_idx += 1
            current_byte >>= 8
            bit_count -= 8

    if bit_count > 0:
        buffer[byte_idx] = current_byte & 0xFF
        byte_idx += 1

    return buffer[:byte_idx], len(buffer[:byte_idx]) * 8 + bit_count


class BitReader:
    """High-performance bit reader using a 32-bit integer buffer (Little-Endian / LSB first)."""

    def __init__(self, data: bytes):
        self.data = data
        self.byte_idx = 0
        self.bit_buffer = 0
        self.bits_left = 0
        self.length = len(data)

    def has_bits(self) -> bool:
        """Returns True if there are bits left in the buffer or unread bytes in data."""
        return self.bits_left > 0 or self.byte_idx < self.length

    def read_bits(self, num_bits: int) -> int:
        """Reads a fixed number of bits from the lower part of the buffer."""
        if num_bits == 0:
            return 0

        while self.bits_left < num_bits:
            if self.byte_idx >= self.length:
                raise IndexError("End of bit stream reached.")
            self.bit_buffer |= self.data[self.byte_idx] << self.bits_left
            self.bits_left += 8
            self.byte_idx += 1

        # Extract the lowest bits
        val = self.bit_buffer & ((1 << num_bits) - 1)
        self.bit_buffer >>= num_bits
        self.bits_left -= num_bits
        return val

    def count_zeros(self) -> int:
        """Reads unary code (q zeros followed by '1') instantly in Little-Endian."""
        q = 0
        while self.has_bits():
            if self.bits_left > 0:
                mask = (1 << self.bits_left) - 1
                valid_bits = self.bit_buffer & mask

                if valid_bits != 0:
                    lowest_bit = int(valid_bits & -valid_bits)
                    leading_zeros = lowest_bit.bit_length() - 1
                    q += leading_zeros

                    consumed = leading_zeros + 1
                    self.bit_buffer >>= consumed
                    self.bits_left -= consumed
                    return q

                q += self.bits_left
                self.bits_left = 0
                self.bit_buffer = 0

            bytes_to_read = min(4, self.length - self.byte_idx)
            self.bit_buffer = int.from_bytes(self.data[self.byte_idx: self.byte_idx + bytes_to_read], byteorder='little')
            self.bits_left = bytes_to_read * 8
            self.byte_idx += bytes_to_read

        return -1


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
        if 0 == k:
            encoded_bytes = values.tobytes()
            total_bits = len(encoded_bytes) * 8
            return encoded_bytes, total_bits

        if signed:
            unsigned_vals = cls._signed_to_unsigned(values)
        else:
            unsigned_vals = values

        divisor = 1 << k

        q_vals = unsigned_vals >> k  # Quotient
        r_vals = unsigned_vals & (divisor - 1)  # Remainder

        return fast_encode_loop(q_vals, r_vals, k)

        writer = BitWriter()
        # Unary encoding of quotient q: q zeros followed by '1'
        for q,r in zip(q_vals,r_vals):
            while q >= 23:
                # Write 31 zeros (we write 0 with length 31)
                writer.write_bits(0, 23)
                q -= 23
            writer.write_bits(1<<q, q+1)
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
            k = 0

        if k > 0:
            b = 1 + k + (2 if is_signed else 1) * mean_abs / (1 << k)
            b0 = residuals.itemsize * 8
            if b > b0:
                k = 0

        encoded_bytes, total_bits = GolombRiceCodec.encode_stream(residuals, k=k, signed=is_signed)
        return encoded_bytes, total_bits, k

    @classmethod
    def decode_stream(cls, encoded_bytes: bytes, k: int = 2, res_dtype=np.int16) -> np.ndarray:
        """
        Unpacks and decodes a Golomb-Rice bit stream back into integers
        dynamically until the end of the buffer is reached.
        """
        if 0 == k:
            return np.frombuffer(encoded_bytes, dtype=res_dtype)

        reader = BitReader(encoded_bytes)
        unsigned_vals = []
        while True:
            q = reader.count_zeros()
            if q == -1:
                break
            r = reader.read_bits(k)
            # Reconstruct value: val = q * 2^M + r
            val = (q << k) + r
            unsigned_vals.append(val)

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
    sample_size = 10003
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


def test_bitstream():
    print("--- START BITSTREAM CODEC TEST ---")

    test_data = [
        (5, 3, 'fixed'),  # 0b101
        (1, 1, 'fixed'),  # 0b1
        (4, 4, 'unary'),  # 4 zeros + 1 one (write_bits(1, 5) -> 0b00001)
        (12, 4, 'fixed'),  # 0b1100
        (3, 3, 'unary'),  # 3 zeros + 1 one (write_bits(1, 4) -> 0b0001)
        (1023, 10, 'fixed'),  # Max for 10 bits
        (0, 1, 'fixed'),  # 0b0
    ]

    writer = BitWriter()

    # 1. Encoding phase
    for val, bits, t_type in test_data:
        if t_type == 'fixed':
            writer.write_bits(val, bits)
        elif t_type == 'unary':
            writer.write_bits(1<<bits, bits + 1)

    encoded_bytes, total_bits = writer.flush()
    print(f"Encoded bytes: {len(encoded_bytes)}, Total useful bits: {total_bits}")
    print(f"Hex dump: {encoded_bytes.hex()}")

    # 2. Decoding phase & validation
    reader = BitReader(encoded_bytes)

    for i, (orig_val, bits, t_type) in enumerate(test_data):
        if t_type == 'fixed':
            decoded_val = reader.read_bits(bits)
            assert decoded_val == orig_val, f"Error at step {i}: expected {orig_val}, got {decoded_val}"
            print(f"Step {i} (Fixed): Read {decoded_val} ({bits} bits)")

        elif t_type == 'unary':
            decoded_val = reader.count_zeros()
            assert decoded_val == orig_val, f"Error at step {i} (Unary): expected {orig_val}, got {decoded_val}"
            print(f"Step {i} (Unary): Found {decoded_val} zeros")

    assert reader.count_zeros() == -1
    print("\n[SUCCESS] All data recovered with 100% precision!")


if __name__ == "__main__":
    test_bitstream()
    run_codec_tests()