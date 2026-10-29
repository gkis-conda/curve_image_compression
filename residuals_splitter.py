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

    def fit(self, residuals: np.ndarray):
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

    def get_codec_factory_config(self) -> dict:
        """
        Factory method to provide optimal coder routing parameters
        based on the analyzed distribution type.
        """
        config = {
            "distribution_type": self.distribution_type,
            "lambda_core": self.lambda_core,
            "mixing_pi": self.pi,
            "threshold": self.threshold
        }

        if self.distribution_type == 'laplacian_core':
            # Optimal for standard Golomb-Rice
            k = max(0, int(np.floor(np.log2(self.lambda_core * 0.693147))))
            config["recommended_coder"] = "GolombRice"
            config["golomb_k"] = k

        elif self.distribution_type == 'mixture_tail':
            # Optimal for Exp-Golomb core + Raw tail
            k = max(0, int(np.floor(np.log2(self.lambda_core * 0.693147))))
            config["recommended_coder"] = "ExpGolomb"
            config["core_k"] = k

        else:
            # Fallback for diffuse noise
            config["recommended_coder"] = "Raw"

        return config


class ResidualSplitter:
    """
    Execution engine: splits residuals into core and tail streams
    based on the provided config or automatic estimation via UniformTailEstimator.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}

    def process_encoding(self, signal: np.ndarray, predictions: np.ndarray) -> tuple:
        signal = np.asarray(signal, dtype=np.int32)
        predictions = np.asarray(predictions, dtype=np.int32)
        residuals = signal - predictions

        threshold = self.config.get("splitter_threshold", -1)

        # Automatically compute threshold using the estimator if not explicitly provided
        if threshold <= 0:
            if residuals.size > 0:
                estimator = UniformTailEstimator(tail_min=self.config.get("tail_min", 32))
                estimator.fit(residuals)

                threshold = estimator.threshold
                # Save analyzed stats back to config for encoder routing
                self.config.update(estimator.get_codec_factory_config())
            else:
                threshold = 32

            self.config["splitter_threshold"] = threshold

        # Check if splitting should be active
        use_splitter = threshold > 0
        if not use_splitter:
            # Splitter bypassed: return empty reset mask, all residuals go to core stream
            return np.zeros_like(residuals, dtype=bool), residuals, np.array([], dtype=np.int32)

        reset_mask = np.abs(residuals) > threshold
        smooth_residuals = np.where(reset_mask, 0, residuals)
        reset_amplitudes = signal[reset_mask]

        return reset_mask, smooth_residuals, reset_amplitudes

    def process_decoding(self, predictions: np.ndarray, reset_mask: np.ndarray,
                         smooth_residuals: np.ndarray, reset_amplitudes: np.ndarray) -> np.ndarray:
        predictions = np.asarray(predictions, dtype=np.int32)
        reset_mask = np.asarray(reset_mask, dtype=bool)
        smooth_residuals = np.asarray(smooth_residuals, dtype=np.int32)
        reset_amplitudes = np.asarray(reset_amplitudes, dtype=np.int32)

        reconstructed = np.empty_like(predictions)
        amp_idx = 0
        for i in range(len(predictions)):
            if reset_mask[i]:
                reconstructed[i] = reset_amplitudes[amp_idx]
                amp_idx += 1
            else:
                reconstructed[i] = predictions[i] + smooth_residuals[i]
        return reconstructed


def test_residual_splitter(splitter):
    print(f"\n--- TEST: {splitter.__class__.__name__} 3-Stream Architecture ---")

    signal = np.array([100, 102, 105, 103, 500, 502, 505, 501], dtype=np.int32)
    predictions = np.array([98, 101, 103, 104, 200, 490, 500, 498], dtype=np.int32)

    reset_mask, smooth_residuals, reset_amplitudes = splitter.process_encoding(signal, predictions)

    print(f"Signal:            {signal}")
    print(f"Predictions:       {predictions}")
    print(f"Raw Residuals:     {signal - predictions}")
    if hasattr(splitter, 'threshold'):
        print(f"Active Threshold:  {splitter.config['threshold']}")
    if hasattr(splitter, 'lambda_core'):
        print(f"Estimated Lambda:  {splitter.config['lambda_core']:.2f}")
    print(f"Reset Mask:        {reset_mask.astype(int)}")
    print(f"Smooth Residuals:  {smooth_residuals}")
    print(f"Reset Amplitudes:  {reset_amplitudes}")

    reconstructed_signal = splitter.process_decoding(predictions, reset_mask, smooth_residuals, reset_amplitudes)
    print(f"Reconstructed:     {reconstructed_signal}")

    assert np.array_equal(signal, reconstructed_signal), "[FAIL] Splitter reconstruction mismatch!"
    assert np.sum(reset_mask) == len(reset_amplitudes), "[FAIL] Amplitude count mismatch with reset mask!"

    print(f"[PASS] {splitter.__class__.__name__} 3-stream roundtrip verified successfully.")


def test_uniform_tail_parameter_estimation(splitter):
    print("\n--- TEST: Parameter Estimation Accuracy for UniformTailResidualSplitter ---")

    np.random.seed(42)
    n_samples = 5000

    true_lambda = 3.5
    true_pi = 0.04  # 4% heavy tails

    core_noise = np.random.laplace(loc=0.0, scale=true_lambda, size=n_samples).astype(np.int32)

    spike_mask = np.random.rand(n_samples) < true_pi
    spikes = np.where(np.random.rand(n_samples) > 0.5,
                      np.random.randint(40, 150, size=n_samples),
                      np.random.randint(-150, -40, size=n_samples))

    residuals = np.where(spike_mask, spikes, core_noise)
    predictions = np.random.randint(500, 1000, size=n_samples, dtype=np.int32)
    signal = predictions + residuals

    reset_mask, smooth_residuals, reset_amplitudes = splitter.process_encoding(signal, predictions)

    print(f"Total Samples:       {n_samples}")
    print(f"True Core Lambda:    {true_lambda}")
    print(f"Estimated Lambda:    {splitter.config['lambda_core']:.2f}")
    print(f"True Mixing Pi:      {true_pi}")
    print(f"Estimated Mixing Pi: {splitter.config['mixing_pi']:.4f}")
    print(f"Calculated Threshold:{splitter.config['threshold']}")
    print(f"Detected Resets:     {np.sum(reset_mask)} (True spikes: {np.sum(spike_mask)})")

    assert abs(splitter.config['lambda_core'] - true_lambda) < 0.5, "[FAIL] Lambda core estimation error is too high!"
    assert abs(splitter.config['mixing_pi'] - true_pi) < 0.02, "[FAIL] Mixing weight pi estimation error is too high!"

    reconstructed_signal = splitter.process_decoding(predictions, reset_mask, smooth_residuals, reset_amplitudes)
    assert np.array_equal(signal, reconstructed_signal), "[FAIL] Reconstruction mismatch!"

    print("[PASS] Parameter estimation and roundtrip test passed successfully.")


if __name__ == "__main__":
    test_residual_splitter(ResidualSplitter())
    test_uniform_tail_parameter_estimation(ResidualSplitter())

