import numpy as np


import numpy as np

class ExpGolombCodec:
    """
    Exponential-Golomb entropy coder/decoder with built-in support for
    both one-sided (unsigned) and two-sided (signed +/-) data streams.
    Automatic stream termination via trailing zeros padding detection.
    """

    @staticmethod
    def _signed_to_unsigned(arr: np.ndarray) -> np.ndarray:
        """
        Interleaves signed integers into non-negative integers:
        0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, 2 -> 4, etc.
        """
        arr = np.asarray(arr, dtype=np.int32)
        return np.where(arr >= 0, 2 * arr, -2 * arr - 1)

    @staticmethod
    def _unsigned_to_signed(arr: np.ndarray) -> np.ndarray:
        """Reverses the interleaving back to signed integers."""
        arr = np.asarray(arr, dtype=np.int32)
        is_even = (arr % 2) == 0
        return np.where(is_even, arr // 2, -(arr + 1) // 2)

    @staticmethod
    def fast_exp_golomb_encode(unsigned_vals, k: int):
        buffer = bytearray()
        current_byte = 0
        bit_count = 0
        k_int = int(k)

        for val in unsigned_vals:
            val_int = int(val)
            q = val_int >> k_int
            r = val_int & ((1 << k_int) - 1)

            q_inf = q + 1
            q_len = q_inf.bit_length()
            zeros_count = q_len - 1

            # Little-Endian packing: zeros -> q_inf -> r
            combined_code = r
            combined_code |= (q_inf << k_int)
            combined_code <<= zeros_count

            code_len = zeros_count + q_len + k_int

            current_byte |= (combined_code << bit_count)
            bit_count += code_len

            while bit_count >= 8:
                buffer.append(current_byte & 0xFF)
                current_byte >>= 8
                bit_count -= 8

        if bit_count > 0:
            buffer.append(current_byte & 0xFF)

        encoded_bytes = bytes(buffer)
        total_bits = (len(encoded_bytes) - 1) * 8 + bit_count if len(encoded_bytes) > 0 else 0
        return encoded_bytes, total_bits

    @staticmethod
    def fast_exp_golomb_decode(encoded_bytes: bytes, k: int):
        q_vals = []
        r_vals = []

        length = len(encoded_bytes)
        byte_idx = 0
        bit_buffer = 0
        bits_left = 0

        k_int = int(k)
        mask_k = (1 << k_int) - 1

        while True:
            # 1. Пакетная подзарядка буфера до 32 бит
            while bits_left < 32 and byte_idx < length:
                free_bytes = (64 - bits_left) // 8
                bytes_to_read = min(free_bytes, length - byte_idx)
                if bytes_to_read <= 0:
                    break
                chunk = encoded_bytes[byte_idx: byte_idx + bytes_to_read]
                new_data = int.from_bytes(chunk, byteorder='little')
                bit_buffer |= (new_data << bits_left)
                bits_left += bytes_to_read * 8
                byte_idx += bytes_to_read

            if bits_left <= 0:
                break

            mask_valid = (1 << min(bits_left, 62)) - 1
            valid_bits = bit_buffer & mask_valid

            # Автоматическое определение конца потока (остались только нули выравнивания)
            if valid_bits == 0:
                break

            # Считаем количество нулей префикса с младшего конца (Trailing Zeros)
            lowest_bit = valid_bits & (-valid_bits)
            zeros_count = lowest_bit.bit_length() - 1

            # Потребляем унарные нули
            bit_buffer >>= zeros_count
            bits_left -= zeros_count

            # ИСПРАВЛЕНО: required_bits теперь рассчитывается динамически и точно!
            q_len = zeros_count + 1
            required_bits = q_len + k_int

            # Добираем байты, если под унарный остаток и префикс не хватает бит в окне
            while bits_left < required_bits and byte_idx < length:
                free_bytes = (64 - bits_left) // 8
                bytes_to_read = min(free_bytes, length - byte_idx)
                if bytes_to_read <= 0:
                    break
                chunk = encoded_bytes[byte_idx: byte_idx + bytes_to_read]
                new_data = int.from_bytes(chunk, byteorder='little')
                bit_buffer |= (new_data << bits_left)
                bits_left += bytes_to_read * 8
                byte_idx += bytes_to_read

            # ИСПРАВЛЕНО: Если физически данных в буфере всё равно меньше, чем требует токен — поток битый
            if bits_left < required_bits:
                break

            # Читаем информационную часть q_inf
            mask_q = (1 << q_len) - 1
            q_inf = bit_buffer & mask_q
            bit_buffer >>= q_len
            bits_left -= q_len

            q = q_inf - 1
            q_vals.append(q)

            # Читаем остаток r
            if k_int > 0:
                r = bit_buffer & mask_k
                bit_buffer >>= k_int
                bits_left -= k_int
                r_vals.append(r)
            else:
                r_vals.append(0)

        return np.array(q_vals, dtype=np.int32), np.array(r_vals, dtype=np.int32)

    @classmethod
    def encode_stream(cls, values, k: int = 2, signed: bool = True) -> tuple:
        # ИСПРАВЛЕНО: удалена блокировка k == 0. Exp-Golomb при k=0 полностью валиден.
        if signed:
            unsigned_vals = cls._signed_to_unsigned(values)
        else:
            unsigned_vals = values

        return cls.fast_exp_golomb_encode(unsigned_vals, k)

    @staticmethod
    def encode_adaptive(residuals: np.ndarray):
        if residuals.size == 0:
            return residuals.tobytes(), 0, -1

        is_signed = np.issubdtype(residuals.dtype, np.signedinteger)
        mean_abs = np.mean(np.abs(residuals))

        # Оценка оптимального параметра k
        if mean_abs > 0:
            optimal_m_float = mean_abs * 0.693147
            k = int(np.round(np.log2(optimal_m_float)))
            k = max(0, k)
        else:
            k = 0

        # Критерий отката (Fallback)
        if k > 0:
            b = 1 + k + (2 if is_signed else 1) * mean_abs / (1 << k)
            b0 = residuals.itemsize * 8
            if b > b0:
                k = 0

        encoded_bytes, total_bits = ExpGolombCodec.encode_stream(residuals, k=k, signed=is_signed)
        return encoded_bytes, total_bits, k

    @classmethod
    def decode_stream(cls, encoded_bytes: bytes, k: int = 2, res_dtype=np.int16,
                      num_elements: int = None) -> np.ndarray:
        # ИСПРАВЛЕНО: удалена блокировка k == 0.
        q_vals, r_vals = cls.fast_exp_golomb_decode(encoded_bytes, k)

        if len(q_vals) == 0:
            return np.array([], dtype=res_dtype)

        unsigned_vals = (q_vals.astype(np.int64) << k) + r_vals.astype(np.int64)

        if np.issubdtype(res_dtype, np.signedinteger):
            res = cls._unsigned_to_signed(unsigned_vals)
        else:
            res = unsigned_vals

        return np.array(res, dtype=res_dtype)


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
    encoded_bytes_s, total_bits_s = ExpGolombCodec.encode_stream(signed_data, k=M_signed, signed=True)
    decoded_data_s = ExpGolombCodec.decode_stream(encoded_bytes_s, k=M_signed, res_dtype=signed_data.dtype)

    assert np.array_equal(signed_data, decoded_data_s), f"[FAIL] Signed mode reconstruction mismatch!\n" \
    f"Data: {signed_data} \n Decoded: {decoded_data_s}"
    print("[PASS] Signed mode bit-accurate reconstruction verified.")

    print("\n--- TEST 2: One-Sided Unsigned Mode ---")
    M_unsigned = 2
    encoded_bytes_u, total_bits_u = ExpGolombCodec.encode_stream(unsigned_data, k=M_unsigned, signed=False)
    decoded_data_u = ExpGolombCodec.decode_stream(encoded_bytes_u, k=M_unsigned, res_dtype=np.uint16)

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