import numpy as np
import math


def samples_number(w: float, laplace_lambda: float, L: float, U: float, epsilon: float = 1. , confidence: float = 0.95) -> int:
    """
    Calculates the minimum sample size (N) required to estimate the parameters
    of a mixed Laplace-Uniform distribution with a target precision using MLE
    or Method of Moments on a subsampled matrix.

    Theoretical Approach & Derivation:
    ----------------------------------
    The objective is to avoid running heavy Maximum Likelihood Estimation (MLE)
    over multi-million pixel elevation/residual matrices by trimming the input data
    down to a statistically representative random subsample of size N.

    We model the terrain residual variable X as a mixture of two distinct processes:
    1. Smooth Terrain Signal: X ~ Laplace(0, b = 1/lmbda) with mixture weight 'w'.
    2. Geomorphological Cliff/Canyon Noise: Two perfectly symmetric, disconnected outer
       Uniform intervals defined as [-U, -L] U [L, U] with a total weight '1 - w'.
       The threshold 'L' marks the exact absolute boundary boundary where the core
       Laplace predictor fails, leaving a structural hollow gap in the noise profile
       over the central [-L, L] region.

    To determine N via a robust interval estimation, we evaluate the statistical
    behavior of the sample Mean Absolute Deviation (MAD) centered at zero, defined as:
        Sample_MAD = (1 / N) * sum(|X_i|)

    According to the Central Limit Theorem (CLT), as N grows, the Sample_MAD
    let Z=|X|
    converges toward a normal distribution centered at the true population absolute MAD (mu_Z)
    with a standard error scaled by the population variance of absolute deviations (sigma_Z^2):
        Sample_MAD ~ N( mu_Z, sigma_Z^2 / N )

    By bounding the maximum allowed estimation error of the sample MAD by 'epsilon'
    (representing our target estimation precision corridor in physical units/meters)
    under a chosen 'confidence' interval (mapped to its standard normal critical value,
    denoted as z_crit or Z-score), we establish the following inequality:
        z_crit * sqrt( sigma_Z^2 / N ) <= epsilon

    Squaring both sides and solving for N gives the core interval equation:
        N >= (z_crit / epsilon)^2 * sigma_Z^2
    where:
        X          = the raw signed terrain residual (input data point)
        Z          = the absolute residual magnitude, defined as Z = |X|
        mu_Z       = E[Z] = E[|X|] (the theoretical population absolute MAD)
        E[X^2]     = E[Z^2] = E[|X|^2] (the second raw moment of absolute deviations)
        sigma_Z^2  = E[X^2] - (mu_Z)^2 (the true population variance of the absolute variable Z)
        z_crit     = the standard normal critical value (Z-score multiplier, e.g., 1.96 for 95% confidence)


    Parameters:
    -----------
    w                   : float, [0, 1] -> Estimated weight (ratio) of smooth Laplace data.
    laplace_lambda      : float         -> Expected baseline Laplace scale parameter (lambda = 1/b).
    L                   : float         -> Lower boundary of the shifted Uniform terrain noise (e.g., canyon depth).
    U                   : float         -> Upper boundary of the shifted Uniform terrain noise (e.g., ridge height).
    epsilon             : float         -> Maximum allowed margin of error for the estimate (in meters/units).
    confidence          : float         -> Confidence level (default 0.95 maps to Z = 1.96).

    Returns:
    --------
    int -> Minimum number of random samples required.
    """
    # 1. Map confidence level to standard normal distribution critical value (Z-score)
    # Using classical two-tailed thresholds
    if confidence == 0.9973:
        z_score = 3.0  # Rigid 3-sigma rule: 3 * sigma <= epsilon
    elif math.isclose(confidence, 0.95):
        z_score = 1.960
    elif math.isclose(confidence, 0.99):
        z_score = 2.576
    elif math.isclose(confidence, 0.90):
        z_score = 1.645
    else:
        # Fallback approximation for arbitrary confidence intervals
        from scipy.stats import norm
        z_score = norm.ppf(1.0 - (1.0 - confidence) / 2.0)

    b = 1.0 / laplace_lambda
    w_noise = 1.0 - w

    # --- PERFECTLY SYMMETRIC MOMENTS ---
    # 1. Theoretical MAD (E[|X|])
    # The absolute mean of the uniform noise is simply the midpoint of [L, U]
    mu_noise_abs = (L + U) / 2.0
    mu_Z = w * b + w_noise * mu_noise_abs

    # 2. Second raw moment (E[X^2])
    # Squaring mirrors the negative interval onto the positive one seamlessly
    e_x2_noise = (L ** 2 + L * U + U ** 2) / 3.0
    e_X2 = (w * (2.0 * b ** 2)) + (w_noise * e_x2_noise)

    # 3. Population variance of absolute deviations (sigma_Z^2)
    sigma_Z_sq = e_X2 - (mu_Z ** 2)

    if sigma_Z_sq <= 0:
        return 1

    # 4. Apply the CLT interval dimension boundary rule
    N_float = ((z_score / epsilon) ** 2) * sigma_Z_sq
    return int(math.ceil(N_float))


class UniformTailEstimator:
    """
    The 'Microscope' analyzer: evaluates residual statistics using EM-based MLE
    for a Laplace + Uniform mixture, classifies the distribution type, and provides
    factory configurations for downstream entropy coders.
    """

    def __init__(self, tail_min: int = 32, max_iters: int = 15, target_lambda_limit: float = 12.0):
        self.tail_min = tail_min
        self.max_iters = max_iters
        self.target_lambda_limit = target_lambda_limit

        # Estimated parameters
        self.lambda_core = None
        self.pi = None
        self.threshold = None
        self.distribution_type = None  # 'laplacian_core', 'mixture_tail', 'uniform_bypass'

    def fit_(self, residuals: np.ndarray):
        n_samples = residuals.size

        if n_samples < 10:
            self.lambda_core = 2.0
            self.pi = 0.05
            self.threshold = 50
            self.distribution_type = 'laplacian_core'
            return self

        # --- DYNAMIC SAMPLE SIZE VIA CLT & MAD VARIANCE ---
        # 1. Quick pilot to get rough parameters (w, lmbda, bounds)
        pilot_size = min(1000, n_samples)
        pilot_indices = np.random.choice(n_samples, size=pilot_size, replace=False)
        pilot_residuals = residuals[pilot_indices]

        pilot_residuals_abs = np.abs(pilot_residuals)
        rough_mad = np.median(pilot_residuals_abs)
        rough_lambda = max(float(rough_mad / 0.693147), 0.5)

        # Classify as out-of-model noise strictly past the threshold boundary
        tail_mask = pilot_residuals_abs >= self.tail_min
        rough_pi = np.clip(np.sum(tail_mask) / pilot_size, 0.001, 0.5)

        tail_abs = pilot_residuals_abs[tail_mask]
        L_bound = self.tail_min
        U_bound = L_bound

        if tail_abs.size > 0:
            L_bound = float(np.min(tail_abs))  # Safely captures the lower absolute limit (>= tail_min)
            U_bound = float(np.max(tail_abs))  # Safely captures the maximum absolute magnitude

        # Tiny safety margin only if all tail elements happen to be identical
        if L_bound >= U_bound:
            U_bound = L_bound + 100.0

        # 2. Compute required sample size using the CLT function
        # L_bound is now strictly your tail_min threshold, and U_bound is the maximum outlier magnitude.
        optimal_fit_samples = samples_number(
            w=1.0 - rough_pi,
            laplace_lambda=rough_lambda,
            L=L_bound,
            U=U_bound
        )

        optimal_fit_samples = min(optimal_fit_samples, n_samples)

        print("optimal fit samples:", optimal_fit_samples)
        # Representative subsampling for heavy EM optimization
        # Statistically, 5000 samples are more than enough for robust MLE convergence.
        if n_samples > optimal_fit_samples:
            # Uniform random sampling without replacement to preserve distribution shape
            indices = np.random.choice(n_samples, size=optimal_fit_samples, replace=False)
            selection = residuals[indices]
            n_samples = optimal_fit_samples
        else:
            selection = residuals

        # 1. EM-based MLE Estimation for Laplace Core + Uniform Tail
        abs_res = np.abs(selection)
        self.lambda_core = rough_lambda
        self.pi = rough_pi
        tail_span = U_bound - L_bound
        C = 0.5 / tail_span

        for _ in range(self.max_iters):
            pdf_core = (1.0 / (2.0 * self.lambda_core)) * np.exp(-abs_res / self.lambda_core)
            pdf_tail = np.where(abs_res >= L_bound, C, 1e-10)

            denom = (1.0 - self.pi) * pdf_core + self.pi * pdf_tail + 1e-12
            gamma = (self.pi * pdf_tail) / denom

            sum_gamma = np.sum(gamma)
            sum_one_gamma = n_samples - sum_gamma

            self.pi = np.clip(sum_gamma / n_samples, 0.001, 0.5)

            if sum_one_gamma > 1.0:
                self.lambda_core = np.sum((1.0 - gamma) * abs_res) / sum_one_gamma
            self.lambda_core = max(self.lambda_core, 0.5)

        # 2. Classification & Routing Decision based on Microscope findings

        # Case A: If distribution is too wide/diffuse (lambda exceeds target limit),
        # entropy coding via Golomb is inefficient -> Bypass / Uniform mode.
        if self.lambda_core >= self.target_lambda_limit and self.pi > 0.2:
            self.distribution_type = 'uniform'
            self.threshold = -1  # Disable splitting, uncompressed or raw mode
            return self

        # Case B: If outlier proportion (pi) is negligible or core lambda is very small,
        # no heavy tail exists -> Pure Laplacian Core (Golomb-Rice).
        if self.pi < 0.005 or U_bound <= self.tail_min:
            self.distribution_type = 'laplacian_core'
            self.threshold = int(U_bound + 1)  # Splitter effectively inactive
            return self

        # Case C: Classical Mixture (Smooth Core + Structural Break Tail)
        self.distribution_type = 'mixture_tail'
        ratio = (1.0 - self.pi) / (2.0 * self.lambda_core * self.pi * C + 1e-12)
        if ratio > 1.0:
            calc_T = self.lambda_core * np.log(ratio)
            self.threshold = int(np.clip(np.round(calc_T), self.tail_min * 0.5, U_bound))
        else:
            self.threshold = self.tail_min

        return self

    def fit(self, residuals: np.ndarray):
        n_samples = residuals.size

        if n_samples < 10:
            self.lambda_core = 2.0
            self.pi = 0.05
            self.threshold = 50
            self.distribution_type = 'laplacian_core'
            return self

        # --- DYNAMIC SAMPLE SIZE VIA CLT & MAD VARIANCE ---
        pilot_size = min(1000, n_samples)
        pilot_indices = np.random.choice(n_samples, size=pilot_size, replace=False)
        pilot_residuals = residuals[pilot_indices]

        pilot_abs = np.abs(pilot_residuals)

        # FIX: Filter out potential spikes from pilot estimation to get true core lambda
        # We take only points below tail_min or below 90th percentile to avoid tail bias
        safe_core_mask = pilot_abs < max(self.tail_min, np.percentile(pilot_abs, 85))
        core_pilot = pilot_abs[safe_core_mask]

        if core_pilot.size > 10:
            rough_mad = np.median(core_pilot)
        else:
            rough_mad = np.median(pilot_abs)

        rough_lambda = max(float(rough_mad / 0.693147), 0.5)

        self.lambda_core = rough_lambda
        # Initial guess for threshold r_t (e.g., 3 * MAD or tail_min)
        self.r_t = float(max(self.tail_min, rough_mad * 3.0))
        # Select only the pilot elements that belong to the tail region
        tail_pilot = pilot_abs[pilot_abs >= self.r_t]
        if tail_pilot.size > 5:
            # Measure empirical spread of actual tail elements relative to r_t
            rough_sigma = float(np.std(tail_pilot - self.r_t))
        else:
            # Fallback if pilot didn't catch enough tail points
            rough_sigma = rough_lambda * 4.0

        self.sigma_tail = max(rough_sigma, rough_lambda * 2.0)
        self.pi = np.clip(tail_pilot.size / pilot_size, 0.001, 0.4)

        # CLT Sample size estimation
        optimal_fit_samples = samples_number(
            w=0.9,
            laplace_lambda=rough_lambda,
            L=self.r_t,
            U=self.r_t + self.sigma_tail * 3,
            epsilon=0.1,
            confidence=0.95
        )
        optimal_fit_samples = int(np.clip(optimal_fit_samples, 1000, min(n_samples, 50000)))

        # Subsampling
        if n_samples > optimal_fit_samples:
            indices = np.random.choice(n_samples, size=optimal_fit_samples, replace=False)
            selection = residuals[indices]
            n_samples = optimal_fit_samples
        else:
            selection = residuals

        abs_res = np.abs(selection)

        # --- EM-ALGORITHM (Laplace Core + Shifted Gaussian-like Tail) ---
        for _ in range(self.max_iters):
            # 1. Core density (Laplace)
            pdf_core = (1.0 / (2.0 * self.lambda_core)) * np.exp(-abs_res / self.lambda_core)

            # 2. Shifted Tail density: active when |r| >= r_t
            # Normalization factor C for the two-sided shifted tail
            # We can approximate normalization or compute it dynamically based on sigma_tail
            tail_exponent = -((abs_res - self.r_t) ** 2) / (2.0 * (self.sigma_tail ** 2))
            # Restrict tail activation strictly beyond r_t to avoid bleeding into the core
            pdf_tail = np.where(abs_res >= self.r_t, np.exp(tail_exponent), 1e-10)

            denom = (1.0 - self.pi) * pdf_core + self.pi * pdf_tail + 1e-12
            gamma = (self.pi * pdf_tail) / denom

            sum_gamma = np.sum(gamma)
            sum_one_gamma = n_samples - sum_gamma

            # M-step: Update mixing weight pi
            self.pi = np.clip(sum_gamma / n_samples, 0.001, 0.4)

            # M-step: Update Laplace core scale
            if sum_one_gamma > 1.0:
                self.lambda_core = np.sum((1.0 - gamma) * abs_res) / sum_one_gamma
            self.lambda_core = max(self.lambda_core, 0.5)

            # M-step: Update tail dispersion (sigma) weighted by responsibilities past r_t
            sum_gamma_safe = max(sum_gamma, 1.0)
            diff_from_rt = np.maximum(0.0, abs_res - self.r_t)
            self.sigma_tail = np.sqrt(np.sum(gamma * (diff_from_rt ** 2)) / sum_gamma_safe)
            self.sigma_tail = max(self.sigma_tail, 1.0)

        # --- BAYESIAN DECISION BOUNDARY (Optimal Threshold T) ---
        max_abs_res = float(np.max(abs_res))

        if self.lambda_core >= self.target_lambda_limit:
            self.distribution_type = 'uniform'
            self.threshold = -1
            return self

        if self.pi < 0.005:
            self.distribution_type = 'laplacian'
            self.threshold = int(max_abs_res + 1)
            return self

        # Find exact intersection grid where Laplace core density matches the Shifted Tail density
        grid = np.linspace(0, max_abs_res, num=500)
        c_core = (1.0 - self.pi) * (1.0 / (2.0 * self.lambda_core)) * np.exp(-grid / self.lambda_core)
        c_tail = self.pi * np.where(grid >= self.r_t,
                                    np.exp(-((grid - self.r_t) ** 2) / (2.0 * (self.sigma_tail ** 2))), 0.0)

        crossover_indices = np.where(c_core <= c_tail)[0]
        if crossover_indices.size > 0:
            self.threshold = int(grid[crossover_indices[0]])
        else:
            self.threshold = int(self.r_t)

        self.distribution_type = 'mixture_tail'
        return self

    def get_codec_factory_config(self) -> dict:
        """
        Factory method to provide optimal coder routing parameters
        based on the analyzed distribution type.
        """
        config = {
            "distribution_type": self.distribution_type,
            "lambda_core": self.lambda_core,
            "mixing_pi": self.pi,
            "threshold": self.threshold,
        }

        if self.distribution_type == 'laplacian':
            # Optimal for standard Golomb-Rice
            k = max(0, int(np.floor(np.log2(self.lambda_core * 0.693147))))
            recommended_coder = "GolombRice"

        elif self.distribution_type == 'mixture_tail':
            # Optimal for Exp-Golomb core + Raw tail
            k = max(0, int(np.floor(np.log2(self.lambda_core * 0.693147))))
            recommended_coder = "ExpGolomb"
        else:
            # Fallback for diffuse noise
            recommended_coder = "Raw"
            k = 0

        config["golomb_k"] = k
        config["recommended_coder"] = recommended_coder

        return config


class ResidualSplitter:
    """
    Execution engine: splits residuals into core and tail streams
    based on the provided config or automatic estimation via UniformTailEstimator.
    """
    def __init__(self, splitter_threshold):
        self.splitter_threshold = splitter_threshold

    def split(self, residuals: np.ndarray) -> tuple:
        # Check if splitting should be active
        if self.splitter_threshold <= 0:
            # Splitter bypassed: return empty reset mask, all residuals go to core stream
            return np.zeros_like(residuals, dtype=bool), residuals, np.array([], dtype=np.int32)
        else:
            reset_mask = np.abs(residuals) > self.splitter_threshold
            smooth_residuals = np.where(reset_mask, 0, residuals)
            reset_amplitudes = residuals[reset_mask]

        return reset_mask, smooth_residuals, reset_amplitudes

    def merge(self, reset_mask: np.ndarray,
              smooth_residuals: np.ndarray, reset_amplitudes: np.ndarray) -> np.ndarray:
        reset_indices = np.where(reset_mask)[0]
        residuals = smooth_residuals.copy()
        residuals[reset_indices] = reset_amplitudes
        return residuals


def test_residual_splitter():
    splitter = ResidualSplitter(5)
    print(f"\n--- TEST: {splitter.__class__.__name__} 3-Stream Architecture ---")

    signal = np.array([100, 102, 105, 103, 500, 502, 505, 501], dtype=np.int32)
    predictions = np.array([98, 101, 103, 104, 200, 490, 500, 498], dtype=np.int32)

    reset_mask, smooth_residuals, reset_amplitudes = splitter.split(signal - predictions)

    print(f"Signal:            {signal}")
    print(f"Predictions:       {predictions}")
    print(f"Raw Residuals:     {signal - predictions}")
    print(f"Reset Mask:        {reset_mask.astype(int)}")
    print(f"Smooth Residuals:  {smooth_residuals}")
    print(f"Reset Amplitudes:  {reset_amplitudes}")

    reconstructed_residuals = splitter.merge(reset_mask, smooth_residuals, reset_amplitudes)
    print(f"Reconstructed:     {reconstructed_residuals + predictions}")

    assert np.array_equal(signal, reconstructed_residuals + predictions), "[FAIL] Splitter reconstruction mismatch!"
    assert np.sum(reset_mask) == len(reset_amplitudes), "[FAIL] Amplitude count mismatch with reset mask!"

    print(f"[PASS] {splitter.__class__.__name__} 3-stream roundtrip verified successfully.")


def test_uniform_tail_parameter_estimation():
    print("\n--- TEST: Parameter Estimation Accuracy for UniformTailResidualSplitter ---")

    np.random.seed(42)
    n_samples = 5000

    true_lambda = 3.5
    true_pi = 0.04  # 4% heavy tails

    core_noise = np.round(np.random.laplace(loc=0.0, scale=true_lambda, size=n_samples)).astype(np.int32)

    spike_mask = np.random.rand(n_samples) < true_pi
    spikes = np.where(np.random.rand(n_samples) > 0.5,
                      np.random.randint(40, 150, size=n_samples),
                      np.random.randint(-150, -40, size=n_samples))

    residuals = np.where(spike_mask, spikes, core_noise)
    predictions = np.random.randint(500, 1000, size=n_samples, dtype=np.int32)
    signal = predictions + residuals
    estimator = UniformTailEstimator()
    estimator.fit(residuals)
    splitter = ResidualSplitter(estimator.threshold)
    reset_mask, smooth_residuals, reset_amplitudes = splitter.split(residuals)

    print(f"Total Samples:       {n_samples}")
    print(f"True Core Lambda:    {true_lambda}")
    print(f"Estimated Lambda:    {estimator.lambda_core:.2f}")
    print(f"True Mixing Pi:      {true_pi}")
    print(f"Estimated Mixing Pi: {estimator.pi:.4f}")
    print(f"Calculated Threshold:{estimator.threshold}")
    print(f"Detected Resets:     {np.sum(reset_mask)} (True spikes: {np.sum(spike_mask)})")

    assert abs(estimator.lambda_core - true_lambda) < 0.5, "[FAIL] Lambda core estimation error is too high!"
    assert abs(estimator.pi - true_pi) < 0.02, "[FAIL] Mixing weight pi estimation error is too high!"

    reconstructed_signal = splitter.merge(reset_mask, smooth_residuals, reset_amplitudes) + predictions
    assert np.array_equal(signal, reconstructed_signal), "[FAIL] Reconstruction mismatch!"

    print("[PASS] Parameter estimation and roundtrip test passed successfully.")


def test_pure_laplace_estimation():
    print("\n--- TEST: Pure Laplace (Smooth Plains/Hills) ---")

    np.random.seed(42)
    n_samples = 5000
    true_lambda = 2.5

    residuals = np.random.laplace(loc=0.0, scale=true_lambda, size=n_samples).astype(np.int32)
    predictions = np.random.randint(500, 1000, size=n_samples, dtype=np.int32)
    signal = predictions + residuals
    estimator = UniformTailEstimator()
    estimator.fit(residuals)
    splitter = ResidualSplitter(estimator.threshold)
    reset_mask, smooth_residuals, reset_amplitudes = splitter.split(residuals)

    print(f"Total Samples:       {n_samples}")
    print(f"True Core Lambda:    {true_lambda}")
    print(f"Estimated Lambda:    {estimator.lambda_core:.2f}")
    print(f"Estimated Mixing Pi: {estimator.pi:.4f}")
    print(f"Distribution Type:   {estimator.distribution_type}")
    print(f"Detected Resets:     {np.sum(reset_mask)} (Expected: 0)")

    assert abs(estimator.lambda_core - true_lambda) < 0.5, "[FAIL] Lambda core estimation error for pure Laplace is too high!"
    assert estimator.distribution_type == 'laplacian', "[FAIL] Expected 'laplacian_core' classification!"
    assert np.sum(reset_mask) == 0, "[FAIL] False positive spikes detected in pure Laplace terrain!"

    reconstructed_signal = splitter.merge(reset_mask, smooth_residuals, reset_amplitudes) + predictions
    assert np.array_equal(signal, reconstructed_signal), "[FAIL] Reconstruction mismatch in pure Laplace mode!"

    print("[PASS] Pure Laplace estimation and roundtrip test passed successfully.")


def test_pure_uniform_noise_estimation():
    print("\n--- TEST: High-Entropy Noise / Uniform Bypass Regime ---")

    np.random.seed(42)
    n_samples = 5000

    residuals = np.random.randint(-80, 81, size=n_samples).astype(np.int32)
    predictions = np.random.randint(500, 1000, size=n_samples, dtype=np.int32)
    signal = predictions + residuals
    estimator = UniformTailEstimator()
    estimator.fit(residuals)
    splitter = ResidualSplitter(estimator.threshold)

    reset_mask, smooth_residuals, reset_amplitudes = splitter.split(residuals)

    print(f"Total Samples:       {n_samples}")
    print(f"Estimated Lambda:    {estimator.lambda_core:.2f}")
    print(f"Estimated Mixing Pi: {estimator.pi:.4f}")
    print(f"Distribution Type:   {estimator.distribution_type}")

    assert estimator.distribution_type == 'uniform', "[FAIL] Expected 'uniform_bypass' for high-entropy noise!"

    reconstructed_signal = splitter.merge(reset_mask, smooth_residuals, reset_amplitudes) + predictions
    assert np.array_equal(signal, reconstructed_signal), "[FAIL] Reconstruction mismatch in bypass mode!"
    print("[PASS] Uniform noise bypass and roundtrip test passed successfully.")


if __name__ == "__main__":
    test_residual_splitter()
    test_uniform_tail_parameter_estimation()
    test_pure_uniform_noise_estimation()
    test_pure_laplace_estimation()
