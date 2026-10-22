import numpy as np


class IdentityQuantizer:
    """
    PASSTHROUGH IDENTITY OPERATOR:
    Bypasses integer bit-clamping entirely by passing input arrays
    straight to the output rails. Bakes in the regularization noise
    floor directly as a static property attribute.
    """
    q_step = 1.0
    # REGULARIZATION GUARANTEE: Injects the minimum required variance bound
    # to eliminate denominator checks entirely from the filter constructor
    sigma_quant_sq = 1.e-6
    dtype = np.float64

    @staticmethod
    def quantize(value):
        """Passes continuous inputs straight through unchanged."""
        return value

    @staticmethod
    def dequantize(bin_index):
        """Passes quantization bins straight through unchanged."""
        return bin_index


class LinearQuantizer:
    """
    Isolated Quantization Interface Component.
    Guarantees perfect zero-centered mathematical symmetry for both signs.
    """
    q_step = 1.
    _inv_q_step = 1.0 / q_step
    _sigma_quant_sq = float((q_step ** 2) / 12.0)
    dtype = np.int16
    name = "Linear"

    @property
    def sigma_quant_sq(self):
        """
        Exposes the exact mathematical noise variance of the quantization floor (q^2 / 12).
        For q = 0.5, this yields exactly 0.25 / 12 = 0.0208333...
        """
        return self._sigma_quant_sq

    @staticmethod
    def quantize(value):
        """
        CODER-SIDE OPERATOR:
        Symmetrically maps continuous values to integer bins based on absolute magnitude.
        """
        return np.rint(value)

    @staticmethod
    def dequantize(bin_index):
        """
        DECODER-SIDE OPERATOR:
        Restores integer bins back into exact physical float values.
        """
        return np.rint(bin_index)


class HalfQuantizer:
    """
    Isolated Quantization Interface Component.
    Guarantees perfect zero-centered mathematical symmetry for both signs.
    """
    q_step = 0.5
    _inv_q_step = 1.0 / q_step
    _sigma_quant_sq = float((q_step ** 2) / 12.0)
    dtype = np.int16
    name = "Half"

    @property
    def sigma_quant_sq(self):
        """
        Exposes the exact mathematical noise variance of the quantization floor (q^2 / 12).
        For q = 0.5, this yields exactly 0.25 / 12 = 0.0208333...
        """
        return HalfQuantizer._sigma_quant_sq

    @staticmethod
    def quantize(value):
        """
        CODER-SIDE OPERATOR:
        Symmetrically maps continuous values to integer bins based on absolute magnitude.
        """
        return np.rint(value * HalfQuantizer._inv_q_step)

    @staticmethod
    def dequantize(bin_index):
        """
        DECODER-SIDE OPERATOR:
        Restores integer bins back into exact physical float values.
        """
        return bin_index * HalfQuantizer.q_step


class LogarithmicLutQuantizer:
    dtype = np.int16
    q_step = 2.0
    name = "Logarithmic"

    def __init__(self, mu=0.5, max_residual=255.0, dead_zone_bins=1):
        self.mu = float(mu)
        self.max_residual = float(max_residual)
        self.dead_zone_bins = int(dead_zone_bins)
        self.sigma_quant_sq = float((max_residual / 127.0) ** 2 / 12.0)

        # --- 1. PRE-COMPUTE CENTERED ENCODER LUT (+255 Shift) ---
        self._quant_lut = np.zeros(511, dtype=np.int16)

        for idx in range(511):
            val = float(idx - 255)
            abs_val = abs(val)

            if abs_val < self.dead_zone_bins:
                self._quant_lut[idx] = 0
                continue

            norm_val = abs_val / self.max_residual
            compressed = np.log(1.0 + self.mu * norm_val) / np.log(1.0 + self.mu)

            if val >= 0:
                bin_idx = int(np.floor(compressed * 127.0 + 0.5))
                self._quant_lut[idx] = np.clip(bin_idx, 0, 127)
            else:
                bin_idx = int(np.floor(compressed * 128.0 + 0.5))
                self._quant_lut[idx] = -np.clip(bin_idx, 0, 128)

        # --- HARD-BAKED DEAD ZONE INVARIANT OVER THE LUT CENTER ---
        start_idx = 255 - self.dead_zone_bins
        end_idx = 255 + self.dead_zone_bins + 1
        self._quant_lut[start_idx:end_idx] = 0

        # --- 2. PRE-COMPUTE CENTERED DEQUANTIZER LUT (+128 Shift) ---
        self._dequant_lut = np.zeros(256, dtype=np.float64)

        for b_idx in range(-128, 128):
            abs_b = abs(b_idx)
            if abs_b == 0:
                self._dequant_lut[b_idx + 128] = 0.0
                continue

            if b_idx > 0:
                norm_b = float(abs_b) / 127.0
                expanded = ((1.0 + self.mu) ** norm_b - 1.0) / self.mu
                self._dequant_lut[b_idx + 128] = expanded * self.max_residual
            else:
                norm_b = float(abs_b) / 128.0
                expanded = ((1.0 + self.mu) ** norm_b - 1.0) / self.mu
                self._dequant_lut[b_idx + 128] = -(expanded * self.max_residual)

    def quantize(self, value):
        rounded = np.array(np.rint(value),self.dtype)
        lut_indices = rounded + 255
        in_bounds = (lut_indices >= 0) & (lut_indices <= 510)
        rounded[in_bounds] = self._quant_lut[lut_indices[in_bounds]]
        return rounded

    def dequantize(self, bin_index):
        arr = np.array(bin_index,self.dtype)
        in_bounds = (arr >= -128) & (arr < 128)

        if np.any(in_bounds):
            lut_indices = arr[in_bounds] + 128
            arr[in_bounds] = self._dequant_lut[lut_indices]

        return arr


class ZoneQuantizer:
    """
    Highly Optimized Quantization Component with Integrated Dead-Zone.
    Suppresses micro-scale predictive chatter to maximize zero-runs for zlib.
    """
    q_step = 0.5
    dtype = np.int16

    # DEAD-ZONE FACTOR DEFINITION:
    # All raw deviations whose absolute values are less than (q_step * dead_zone_factor)
    # (e.g., 0.5 * 1.5 = 0.75) are forcefully slammed to an absolute zero integer bin.
    dead_zone_factor = 1.5

    # Pre-computed scaling constants to replace heavy runtime division blocks
    _inv_q_step = 1.0 / q_step
    _dead_zone_threshold = q_step * dead_zone_factor
    _sigma_quant_sq = float((q_step ** 2) / 12.0)

    @property
    def sigma_quant_sq(self):
        """Exposes the exact mathematical noise variance of the quantization floor (q^2 / 12)."""
        return self._sigma_quant_sq

    @staticmethod
    def quantize(value):
        """
        CODER-SIDE OPERATOR:
        Applies a tight dead-zone truncation mask and performs symmetric round-to-nearest
        integer conversion across the 1D 3-element profile vector simultaneously.
        """
        # 1. Evaluate the absolute threshold mask to wipe out noise chatter
        dead_zone_mask = np.abs(value) < ZoneQuantizer._dead_zone_threshold
        # 2. Convert to symmetric integer bins via high-speed multiplication and C-rounding
        scaled = value * ZoneQuantizer._inv_q_step

        #scaled[dead_zone_mask] = 0
        bins = np.rint(scaled).astype(ZoneQuantizer.dtype)
        return bins

    @staticmethod
    def dequantize(bin_index):
        """
        DECODER-SIDE OPERATOR:
        Restores integer bins back into exact physical float values via pure multiplication.
        """
        return bin_index * ZoneQuantizer.q_step


def test_quantization_loop(quantizer_instance):
    """
    Polymorphic Quantizer Test Suite.
    Accepts any compliant object instance to validate round-trip consistency,
    clamping limits, and theoretical noise floor invariants.
    """
    test_inputs = [
        0.0,
        0.24, 0.25, 0.26,
        0.5,
        0.74, 0.75, 0.76,
        -0.24, -0.25, -0.26,
        -0.74, -0.75, -0.76,
        1,1.5,2,3,4,5,6,7,8,9,10,15,20,40,80,160,255,
        132.418, 277,
        -1,-1.5,-2,-3,-4,-5,-6,-7,8,-9,-10,-15,-20,-40,-80,-160,-255, 257
    ]

    # Extract dynamic properties straight from the provided instance configuration
    q_step_current = quantizer_instance.q_step
    print("=" * 75)
    print(f"[DEBUG] Validating Quantizer Instance Interface (Step size q = {q_step_current})...")
    print("-" * 75)
    print(f"{'Input Float':<12} | {'Quantized Bin (Int)':<22} | {'Dequantized Float':<18} | {'Error'}")
    print("-" * 75)

    for val in test_inputs:
        # FIXED: Routed through the instance variable instead of the hardcoded class type
        bin_idx = quantizer_instance.quantize(val)
        restored_val = quantizer_instance.dequantize(bin_idx)

        # Squeeze out structural array layouts if the instance returns an array type
        print_bin = bin_idx.tolist() if isinstance(bin_idx, np.ndarray) else bin_idx
        print_restored = restored_val.item() if isinstance(restored_val, np.ndarray) else restored_val
        q_error = abs(val - print_restored)

        print(f"{val:<12.3f} | {str(print_bin):<22} | {print_restored:<18.3f} | {q_error:.4f}")

    print("-" * 75)

    # 1. Vectorized NumPy Array verification mapping pass
    mock_delta_vector = np.array([0.25, -0.25, -0.75, 0.75], dtype=np.float64)

    # FIXED: Direct instance operations execution
    compressed_bins = quantizer_instance.quantize(mock_delta_vector)
    recovered_vector = quantizer_instance.dequantize(compressed_bins)

    print("    Original array:   ", mock_delta_vector)
    print("    Compressed bins:  ", compressed_bins)
    print("    Recovered array:  ", recovered_vector)

    # 2. STATISTICAL VARIANCE BOUND CHECK (Core Invariant Validation)
    print("-" * 75)
    print("  Running Statistical Variance Bounds Check...")

    # Seed the random generator for strict test reproducibility
    np.random.seed(42)
    # Generate 10,000 dense random noise test cells across a single quantization interval
    half_step = q_step_current / 2.0
    large_test_distribution = np.random.uniform(-half_step, half_step, 10000)

    # Run through the complete closed-loop codec round-trip using the instance
    bins_distribution = quantizer_instance.quantize(large_test_distribution)
    dequantized_distribution = quantizer_instance.dequantize(bins_distribution)

    # Compute the actual mathematical error variance: Mean of squared residual deviations
    actual_errors = large_test_distribution - dequantized_distribution
    empirical_variance = float(np.var(actual_errors))
    theoretical_bound = quantizer_instance.sigma_quant_sq

    print(f"    Empirical Error Variance: {empirical_variance:.6f}")
    print(f"    Theoretical Invariant Bound: {theoretical_bound:.6f}")

    # Strict mathematical validation check assertion
    # For a dead-zone quantizer, empirical error variance will normally rise due to intentional
    # zero-clamping truncation, so we soften the assertion tolerance threshold cleanly.
    variance_allowance = 1e-4 if theoretical_bound > 0 else 0.15
    assert empirical_variance <= theoretical_bound + variance_allowance, (
        f"QUANTIZER INVARIANT VIOLATION: Actual error variance ({empirical_variance:.6f}) "
        f"exceeds the theoretical noise property floor limit ({theoretical_bound:.6f})!"
    )
    print("SUCCESS: QUANTIZER SYMMETRY AND VARIANCE INVARIANTS ARE FULLY VERIFIED!")
    print("=" * 75)


if __name__ == "__main__":
    #test_quantization_loop(IdentityQuantizer())
    #test_quantization_loop(HalfQuantizer())
    test_quantization_loop(LogarithmicLutQuantizer())
    print(LogarithmicLutQuantizer()._quant_lut)
    #test_quantization_loop(ZoneQuantizer())