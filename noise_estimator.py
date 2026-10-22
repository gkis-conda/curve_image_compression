import numpy as np
from scipy.stats import chi2
from surface import Surface


class LinearNoiseModelSolver:
    """
    Independent Matrix Regression Engine (Pure Solver).
    Simply fits: variance = a * Intensity + b over clean pre-filtered data pairs.
    """

    @staticmethod
    def solve_profile(clean_intensities, clean_variances):
        poly = np.polyfit(clean_intensities, clean_variances, deg=1)
        alpha_noise_est = max(0.0, float(poly[0]))
        beta_noise_est = max(1e-5, float(poly[1]))

        return alpha_noise_est, beta_noise_est

    @staticmethod
    def _noise_profile(raw_I, raw_V, num_bins):
        """Helper to build a smoothed noise profile via Gaussian soft-binning."""
        I_min, I_max = np.min(raw_I), np.max(raw_I)
        if abs(I_max - I_min) < 1e-4:
            I_max += 1.0

        bin_centers = np.linspace(I_min, I_max, num_bins)
        bin_step = bin_centers[1] - bin_centers[0] if num_bins > 1 else 1.0
        s = bin_step if bin_step > 0 else 1.0
        gaussian_multiplier = -0.5 / (s ** 2)

        global_median_var = max(1e-5, np.median(raw_V))
        bin_var_sum = np.zeros(num_bins, dtype=np.float64)
        bin_weight_sum = np.zeros(num_bins, dtype=np.float64)

        for i_val, v_val in zip(raw_I, raw_V):
            weights = np.exp(gaussian_multiplier * ((bin_centers - i_val) ** 2))
            bin_var_sum += weights * v_val
            bin_weight_sum += weights

        return bin_centers, np.where(
            bin_weight_sum > 1e-3,
            bin_var_sum / (bin_weight_sum + 1e-6),
            global_median_var
        ), bin_weight_sum

    @staticmethod
    def filter_via_soft_profile(raw_intensities, raw_variances, degrees_of_freedom, confidence_level=0.95,
                                num_bins=6):
        """
        UNIVERSAL PROFILE-WEIGHTED CHI-SQUARE FILTER (TWO-PASS ROBUST + STRICTNESS INVERSION):
        - confidence_level: acts as purity/strictness. Higher value = tighter threshold = more outliers dropped.
        """
        print("num_bins", num_bins, "conf", confidence_level)
        raw_I = np.array(raw_intensities, dtype=np.float64)
        raw_V = np.array(raw_variances, dtype=np.float64)

        if len(raw_V) < 10:
            return raw_I, raw_V, 0

        chi2_threshold = chi2.ppf(confidence_level, degrees_of_freedom)

        # --- PASS 1: Rough initial profiling & aggressive outlier rejection ---
        bin_centers, bin_profile, bin_weights = LinearNoiseModelSolver._noise_profile(raw_I, raw_V, num_bins)
        print("Rough profile:")
        print("I:", bin_centers)
        print("V:",bin_profile)

        expected_variance = np.interp(raw_I, bin_centers, bin_profile)
        poly = np.polyfit(bin_centers, bin_profile, deg=1, w = bin_weights)
        alpha, beta = float(poly[0]), float(poly[1])
        print("alpha, beta", alpha, beta)
        expected_variance = alpha * raw_I + beta
        chi2_statistics = (float(degrees_of_freedom) * raw_V) / np.maximum(1e-5, expected_variance)

        inliers_mask = chi2_statistics <= chi2_threshold
        if np.sum(inliers_mask) < 5:
            return raw_I, raw_V, 0

        clean_I = raw_I[inliers_mask]
        clean_V = raw_V[inliers_mask]

        #rejected_count = int(np.sum(~inliers_mask))
        #return clean_I, clean_V, rejected_count

        # --- PASS 2: Refined precision profiling using ONLY strict clean inliers ---
        bin_centers, bin_profile, bin_weights = LinearNoiseModelSolver._noise_profile(clean_I, clean_V, num_bins)
        print("Refined profile:")
        print("I:",bin_centers)
        print("V:",bin_profile)

        chi2_strict_threshold = chi2.ppf(confidence_level, degrees_of_freedom)
        expected_variance_refined = np.interp(clean_I, bin_centers, bin_profile)
        poly = np.polyfit(bin_centers, bin_profile, deg=1, w=bin_weights)
        alpha, beta = float(poly[0]), float(poly[1])
        print("alpha, beta", alpha, beta)

        chi2_statistics_refined = (float(degrees_of_freedom) * clean_V) / np.maximum(1e-5, expected_variance_refined)
        final_valid_mask = chi2_statistics_refined <= chi2_strict_threshold

        clean_I = clean_I[final_valid_mask]
        clean_V = clean_V[final_valid_mask]
        rejected_count = int(np.sum(~final_valid_mask))

        return clean_I, clean_V, rejected_count


class EvansNoiseExtractor:
    """
    EXTRACTOR A: Quadratic Evans Paraboloid Model.
    z = A*r^2 + B*c^2 + C*r*c + D*r + E*c + F
    Dynamically maps degrees of freedom directly from operator matrix dimensions.
    """

    def dof(self):
        """
        Extracts the statistical degrees of freedom from the fit matrix topology.
        Columns - Rows = DoF.
        """
        return int(self.K_fit.shape[1] - self.K_fit.shape[0])

    def __init__(self, half_width=2):
        self.s = half_width
        # Creates coordinated vectors from [-s ... +s]
        coords = [(r, c) for r in range(-self.s, self.s + 1) for c in range(-self.s, self.s + 1)]
        self.X = np.array([[float(r ** 2), float(c ** 2), float(r * c), float(r), float(c), 1.0]
                             for r, c in coords], dtype=np.float64)
        # Precompute the fixed geometry pseudo-inverse (6 x 5*5 matrix)
        self.K_fit = np.linalg.inv(self.X.T @ self.X) @ self.X.T

    def extract_patch_samples(self, image_matrix, max_patches=400):
        H, W = image_matrix.shape

        rng = np.random.RandomState(517)

        r_centers = rng.randint(self.s, H - self.s, size=max_patches)
        c_centers = rng.randint(self.s, W - self.s, size=max_patches)

        raw_intensities = np.zeros(max_patches, dtype=np.float64)
        raw_variances = np.zeros(max_patches, dtype=np.float64)

        # Pull the degrees of freedom directly from your class interface method
        current_dof = 1./self.dof()

        for idx, (r, c) in enumerate(zip(r_centers, c_centers)):
            flat_patch = image_matrix[r - self.s:r + self.s + 1, c - self.s:c + self.s + 1].flatten()
            coeffs = self.K_fit @ flat_patch
            z_smooth = self.X @ coeffs
            residuals = flat_patch - z_smooth

            # Index -1 is the 'F' intercept (mean brightness at local center)
            raw_intensities[idx] = z_smooth.mean()
            raw_variances[idx] = residuals.dot(residuals) * current_dof

        return raw_intensities, raw_variances


class BilinearNoiseExtractor:
    """
    EXTRACTOR B: Bilinear Surface Model.
    z = d*r + e*c + f*r*c + g
    Parameters = 4. Degrees of Freedom = 9 - 4 = 5. Chi2(5) Threshold = 11.070.
    """
    def dof(self):
        return self.A_pseudo_inv.shape[1] - self.A_pseudo_inv.shape[0]

    def __init__(self, half_width=2):
        self.s = half_width
        # Creates coordinated vectors from [-s ... +s]
        coords = [(r, c) for r in range(-self.s, self.s + 1) for c in range(-self.s, self.s + 1)]
        self.A_bilinear = np.array([[float(r), float(c), float(r * c), 1.0]
                             for r, c in coords], dtype=np.float64)
        self.A_pseudo_inv = np.linalg.inv(self.A_bilinear.T @ self.A_bilinear) @ self.A_bilinear.T

    def extract_patch_samples(self, image_matrix, max_patches=400):
        H, W = image_matrix.shape
        rng = np.random.RandomState(517)
        r_centers = rng.randint(self.s, H - self.s, size=max_patches)
        c_centers = rng.randint(self.s, W - self.s, size=max_patches)

        raw_intensities = np.zeros(max_patches, dtype=np.float64)
        raw_variances = np.zeros(max_patches, dtype=np.float64)
        current_dof = 1./self.dof()

        for idx, (r, c) in enumerate(zip(r_centers, c_centers)):
            flat_patch = image_matrix[r - self.s:r + self.s + 1, c - self.s:c + self.s + 1].flatten()
            coeffs = self.A_pseudo_inv @ flat_patch
            z_smooth = self.A_bilinear @ coeffs
            residuals = flat_patch - z_smooth
            # Index -1 is the 'g' intercept (mean brightness at local center)
            raw_intensities[idx] = z_smooth.mean()
            raw_variances[idx] = residuals.dot(residuals) * current_dof

        return raw_intensities, raw_variances


class MedianNoiseExtractor:
    """
    Non-parametric Robust Median (MAD) Model.
    Evaluates signal intensity via the patch median, and computes noise variance
    using Median Absolute Deviation scaled by a finite-sample correction factor
    calibrated for the exact statistical Degrees of Freedom.
    """

    def dof(self):
        """
        Calculates the statistical degrees of freedom for the median window.
        Total window cells ((2*s + 1)^2) minus 1 constraint (the median itself).
        """
        total_cells = (2 * self.s + 1) * (2 * self.s + 1)
        return total_cells - 1

    def __init__(self, half_width=2):
        self.s = half_width

    def extract_patch_samples(self, image_matrix, max_patches=400):
        H, W = image_matrix.shape
        rng = np.random.RandomState(517)
        r_centers = rng.randint(self.s, H - self.s, size=max_patches)
        c_centers = rng.randint(self.s, W - self.s, size=max_patches)

        raw_intensities = np.zeros(max_patches, dtype=np.float64)
        raw_variances = np.zeros(max_patches, dtype=np.float64)

        # Extract current degrees of freedom (8 for 3x3, 24 for 5x5)
        current_dof = self.dof()

        # FIXED: Finite-sample correction factor for Gaussian MAD scaling under limited DoF.
        # For DoF=8 (3x3), this accurately shifts the 0.6745 denominator down to ~0.6125.
        # For DoF=24 (5x5), this shifts the denominator to ~0.6483, eliminating the underestimation bias.
        c_mad_corrected = 0.67448975 * np.sqrt((current_dof - 1.15) / current_dof)

        for idx, (r, c) in enumerate(zip(r_centers, c_centers)):
            patch = image_matrix[r - self.s: r + self.s + 1, c - self.s: c + self.s + 1]

            # Robust local intensity estimate (unaffected by single steep cliffs)
            local_median = np.median(patch)

            # Calculate Median Absolute Deviation (MAD) over the cells
            mad = np.median(np.abs(patch - local_median))

            sigma_sq_instant = (mad / c_mad_corrected) ** 2

            raw_intensities[idx] = local_median
            raw_variances[idx] = sigma_sq_instant

        return raw_intensities, raw_variances


# ===========================================================================
#      STRICT TEST INVARIANT (YOUR EXECUTOR RUNNER)
# ===========================================================================
def execute_estimator_test_suite(extractor_type='EVANS'):
    print("=" * 75)
    print(f"   LINEAR NOISE MODEL ESTIMATOR: [{extractor_type}] VALIDATION")
    print("=" * 75)

    gen_alpha = 0.0025
    gen_beta = 0.4500
    max_patches = 300
    print(f"[GENERATOR] Targeted Profiles: Alpha (a) = {gen_alpha:.4f}, Beta (b) = {gen_beta:.4f}")

    H, W = 300, 300
    true_surface = Surface([0.0002, 0.001, -0.001, 0.02, 0.01, 100.0])
    Y, X = np.meshgrid(np.arange(300), np.arange(300), indexing='ij')

    np.random.seed(42)
    noisy_image = true_surface.sample_noisy_sensor(Y, X, a=gen_alpha, b=gen_beta)

    if extractor_type == 'EVANS':
        extractor = EvansNoiseExtractor()
    elif extractor_type == 'BILINEAR':
        extractor = BilinearNoiseExtractor()
    elif extractor_type == 'MEDIAN':
        extractor = MedianNoiseExtractor()
    else:
        raise ValueError("Unknown extractor type")

    raw_I, raw_V = extractor.extract_patch_samples(noisy_image, max_patches=max_patches)

    # Route arrays dynamically through the shared soft-binning filter operation
    I, V, rejected = LinearNoiseModelSolver.filter_via_soft_profile(raw_I, raw_V, extractor.dof(), confidence_level=0.8, num_bins=7)

    # 5. Execute Mathematical Solver over the collected sample arrays
    est_alpha, est_beta = LinearNoiseModelSolver.solve_profile(I, V)

    # 6. Cross-examine Generated vs Estimated coefficients
    print("-" * 75)
    print("--- Estimator Invariant Verification Results ---")
    print(f"Total Evaluated Patches:  {max_patches}")
    print(f"Chi2 Structural Rejections: {rejected} edges successfully discarded.")
    print(f"Target Generation:        Alpha = {gen_alpha:.6f} |   Beta = {gen_beta:.6f}")
    print(f"Solved Estimation:        Alpha = {est_alpha:.6f} |   Beta = {est_beta:.6f}")

    delta_alpha = abs(gen_alpha - est_alpha)
    delta_beta = abs(gen_beta - est_beta)
    print(f"Absolute Deviations:    D_Alpha = {delta_alpha:.6f} | D_Beta = {delta_beta:.6f}")

    print("=" * 75)


if __name__ == "__main__":
    execute_estimator_test_suite(extractor_type='EVANS')
    print("\n\n")
    execute_estimator_test_suite(extractor_type='BILINEAR')
    print("\n\n")
    execute_estimator_test_suite(extractor_type='MEDIAN')
