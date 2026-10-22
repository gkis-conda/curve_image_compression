import numpy as np
from surface import Surface

from quantizers import HalfQuantizer, IdentityQuantizer

class EvansWindowFilter:
    _lookahead_coords = \
        [[(-1, 2), (0, 2), (1, 2)], # EAST
        [(2, -1), (2, 0), (2, 1)],
        [(-1, -2), (0, -2), (1, -2)], # WEST
        [(-2, -1), (-2, 0), (-2, 1)]]  # NORTH

    def __init__(self, alpha_noise=0.0, sigma_0=0.0, gamma=0.15, quantize_fn=None,
                 init_brightness=128.0, quantized_signal=False, adaptive_noise=False):
        """
        Initialize the 9-Point Physical Window Evans Filter.
        Fully compatible with Python 3.6+ and clean ASCII.
        """
        self.alpha_noise = alpha_noise
        self.beta_noise =  sigma_0**2 - (self.alpha_noise * 128.0)
        self.quantization_noise = 1/12. if quantized_signal else 0.
        self.quantize_fn = quantize_fn if quantize_fn else IdentityQuantizer()
        self.quantization_noise += self.quantize_fn.sigma_quant_sq
        self.gamma = gamma # relief smoothness (memory length)
        self.e_relief = 0.0 # relief energy estimation
        self.adaptive_noise = adaptive_noise
        # STATE MATRIX: 3x3 physical window of filtered heights centered at (0,0)
        self.Z_window = np.full((3, 3), init_brightness, dtype=np.float64)
        # Precompute the fixed geometry pseudo-inverse to fit a paraboloid from 9 points
        self.X_9, self.cov, self.K_fit = EvansWindowFilter._precompute_paraboloid_fit_matrix()
        self.gain_alpha = 0.
        self.total_noise = sigma_0**2 + self.quantization_noise
        self._lookahead = [EvansWindowFilter._get_lookahead(i) for i in range(4)]

    @staticmethod
    def _precompute_paraboloid_fit_matrix():
        coords = [(r, c) for r in [-1, 0, 1] for c in [-1, 0, 1]]
        X = np.array([[float(r**2), float(c**2), float(r * c), float(r), float(c), 1.0] 
                      for r, c in coords], dtype=np.float64)
        cov = np.linalg.inv(X.T @ X)
        return X, cov, cov @ X.T

    @staticmethod
    def _get_direction(direction_idx):
        directions = {0:'EAST', 2:'WEST', 1:'SOUTH', 3:'NORTH'}
        return directions.get(direction_idx)

    @staticmethod
    def _get_lookahead(direction_idx) -> np.ndarray:
        coords = EvansWindowFilter._lookahead_coords[direction_idx]
        return np.array([[float(r ** 2), float(c ** 2), float(r * c), float(r), float(c), 1.0]
                                for r, c in coords])

    def _get_beta_coefficients(self):
        """Fits the 6 Evans paraboloid coefficients dynamically from the current 9-cell state window."""
        flat_Z = self.Z_window.ravel()
        return self.K_fit @ flat_Z

    def predict(self, direction):
        """PROJECTS heights forward for the look-ahead profile at distance 2."""
        return self._lookahead[direction] @ self.K_fit @ self.Z_window.ravel()

    def predict_std(self, direction) -> np.ndarray:
        """Calculates the prediction standard deviation for the 3 look-ahead points."""
        X_lookahead = self._lookahead[direction]
        leverage = np.sum((X_lookahead @ self.cov) * X_lookahead, axis=1)
        std_vector = np.sqrt(self.total_noise * leverage)
        return std_vector

    def coder_update(self, direction, z):
        """
        CODER-SIDE UPDATE:
        Calculates residual at the look-ahead profile (distance 2), 
        filters and compresses the innovation, updates the physical 3x3 window 
        state, and advances the sliding matrix frame natively.
        """
        # 1. Predict look-ahead profile values at column/row distance 2
        z_pred = self.predict(direction)
        delta = z - z_pred
        # 2. Dynamic Gain
        e_t = (delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2) / 3.
        mean_intensity = max(0.0, z_pred[0] + z_pred[1] + z_pred[2]) / 3.
        sensor_noise = self.alpha_noise * mean_intensity + self.beta_noise
        total_noise = sensor_noise + self.quantization_noise
        self.total_noise = total_noise
        pure_signal_energy = max(0.0, e_t - total_noise)
        self.e_relief = (1. - self.gamma) * self.e_relief + self.gamma * pure_signal_energy
        alpha = min(1., self.e_relief / (self.e_relief + total_noise))
        self.gain_alpha = alpha
        # 3. Quantize and filter the innovation
        delta_filtered = alpha * delta
        delta_compressed = self.quantize_fn.quantize(delta_filtered)
        delta_restored = self.quantize_fn.dequantize(delta_compressed)

        # 4. Compute the actual reconstructed/filtered heights for this new profile
        y_filtered_profile = z_pred + delta_restored
        self._advance_sliding_window(direction, y_filtered_profile)
        if self.adaptive_noise:
            self._adapt_noise_profile()
        return delta_compressed

    def decoder_update(self, direction, delta_compressed):
        """
        DECODER-SIDE UPDATE:
        Synchronously reconstructs the 3-pixel profile and advances the 3x3 window state
        frame identically to the encoder to maintain perfect bit-exact tracking sync.
        """
        z_pred = self.predict(direction)
        delta_restored = self.quantize_fn.dequantize(np.array(delta_compressed))
        y_recovered = z_pred + delta_restored
        self._advance_sliding_window(direction, y_recovered)
        return y_recovered

    def _advance_sliding_window(self, direction, new_profile):
        """
        PHYSICAL FRAME SLIDING MECHANISM:
        Shifts the 3x3 window data values by 1 step in the chosen direction
        and injects the newly verified/recovered profile cleanly.
        """
        if direction == 0: # 'EAST':
            self.Z_window[:, 0] = self.Z_window[:, 1]  # Old col 0 becomes col -1
            self.Z_window[:, 1] = self.Z_window[:, 2]  # Old col 1 becomes col 0
            self.Z_window[:, 2] = new_profile          # New profile becomes col 1

        elif direction == 2: # 'WEST':
            self.Z_window[:, 2] = self.Z_window[:, 1]
            self.Z_window[:, 1] = self.Z_window[:, 0]
            self.Z_window[:, 0] = new_profile

        elif direction == 1: # 'SOUTH':
            self.Z_window[0, :] = self.Z_window[1, :]
            self.Z_window[1, :] = self.Z_window[2, :]
            self.Z_window[2, :] = new_profile

        elif direction == 3: # 'NORTH':
            self.Z_window[2, :] = self.Z_window[1, :]
            self.Z_window[1, :] = self.Z_window[0, :]
            self.Z_window[0, :] = new_profile

    def _adapt_noise_profile(self):
        """
        IN-LOOP ADAPTIVE NOISE CORRECTOR (LMS Subspace Loop).
        Analyzes the internal 3x3 model grid residuals to dynamically tune
        alpha_noise and beta_noise parameters synchronously on coder and decoder.
        """
        # 1. Extract smooth analytical surface coefficients from the current grid state
        beta_curr = self._get_beta_coefficients()

        # 2. Reconstruct the clean, noise-free MNK paraboloid envelope over 9 points
        z_smooth = (self.X_9 @ beta_curr).flatten()
        residuals = self.Z_window.flatten() - z_smooth

        # 3. Calculate instant variance of the residuals (Degrees of Freedom: 9 - 6 = 3)
        sigma_instant_sq = residuals.dot(residuals) / 3.0

        # 4. Gather model parameters for the current zero-centered window intensity
        mean_intensity = max(1.0, float(beta_curr[-1]))
        expected_var = self.alpha_noise * mean_intensity + self.beta_noise

        # 5. Robust Chi-Squared Filter: protects the profile from sudden topographic fractures
        chi2_stat = (3.0 * sigma_instant_sq) / max(expected_var, 1e-5)

        # Critical value for Chi-Square distribution (k=3, alpha=0.05) is exactly 7.815
        if chi2_stat <= 7.815:
            # Conservative step-size to prevent loop chatter or optimization divergence
            mu = 0.001
            noise_error = sigma_instant_sq - expected_var

            # Stochastic Gradient Descent / LMS tuning pass over model weights
            self.alpha_noise += mu * noise_error * mean_intensity / (mean_intensity + 1.0)
            self.beta_noise += mu * noise_error

            # Bound configurations to maintain physical non-negative integrity
            self.alpha_noise = max(0.0, self.alpha_noise)
            self.beta_noise = max(1e-4, self.beta_noise)


def run_comprehensive_test_suite():
    print("=" * 75)
    print("      EVANS WINDOW FILTER: 5-CASE SYSTEMATIC VERIFICATION RUN")
    print("=" * 75)

    # True global paraboloid coefficients (smooth continuous surface shape)
    true_beta = Surface([0.1, 0.05, -0.2, 0.2, -0.1, 128.0])

    # Complete 5-Case Test Configurations Map
    cases_config = {
        1: {
            "name": "Analog Signal, No Noise, Float Residuals",
            "sigma_0": 0.0, "has_noise": False, "quantized_signal": False, "use_quantizer": False, "adaptive":False
        },
        2: {
            "name": "Analog Signal + Noise, Float Residuals",
            "sigma_0": 0.3, "has_noise": True, "quantized_signal": False, "use_quantizer": False, "adaptive":False
        },
        3: {
            "name": "Analog Signal, No Noise + Quantizer",
            "sigma_0": 0.0, "has_noise": False, "quantized_signal": False, "use_quantizer": True, "adaptive":False
        },
        4: {
            "name": "Analog Signal + Noise + Quantizer",
            "sigma_0": 0.3, "has_noise": True, "quantized_signal": False, "use_quantizer": True, "adaptive":False
        },
        5: {
            "name": "Discrete Input Signal + Noise + Quantizer",
            "sigma_0": 0.3, "has_noise": True, "quantized_signal": True, "use_quantizer": True, "adaptive":False
        },
        6: {
            "name": "Discrete Input Signal + Noise + Quantizer + Adaptive noise model",
            "sigma_0": 0.3, "has_noise": True, "quantized_signal": True, "use_quantizer": True, "adaptive":True
        }
    }

    np.random.seed(42)  # Lock seed for reproducible stochastics

    for case_id, cfg in cases_config.items():
        print(f"\n[RUNNING CASE {case_id}] -> {cfg['name']}")
        print("-" * 80)
        print(f"{'Step':<5} | {'Max Recon Err':<14} | {'Std':<6} | {'Window Diff':<15} | {'Alpha (Gain)':<12} | {'Status'}")
        print("-" * 80)

        q_object = Quantizer() if cfg["use_quantizer"] else None

        # Build clean distinct instances for encoder and decoder pipelines
        encoder = EvansWindowFilter(sigma_0=cfg["sigma_0"], gamma=0.15, quantize_fn=q_object,
                                    init_brightness=128.0, quantized_signal=cfg["quantized_signal"],
                                    adaptive_noise=cfg["adaptive"])
        decoder = EvansWindowFilter(sigma_0=cfg["sigma_0"], gamma=0.15, quantize_fn=q_object,
                                    init_brightness=128.0, quantized_signal=cfg["quantized_signal"],
                                    adaptive_noise=cfg["adaptive"])

        enc_origin = [0, 0]
        direction = 0
        rel_coords = [(-1, 2), (0, 2), (1, 2)]

        steps_n = 25
        for step in range(1, steps_n + 1):
            # 1. Sample continuous or discrete inputs at look-ahead profile distance +2
            real_heights = []
            true_heights = []
            for dr, dc in rel_coords:
                h_input = true_beta.evaluate(enc_origin[0] + dr, enc_origin[1] + dc)
                true_heights.append(h_input)
                # Injects baseline sensor Gaussian noise floor
                if cfg["has_noise"]:
                    h_input += np.random.normal(0.0, cfg["sigma_0"])

                # Check source discreteness flag (Case 5 simulates pre-quantized pixels)
                if cfg["quantized_signal"]:
                    h_input = round(h_input)

                real_heights.append(h_input)

            # 2. Run execution cycles symmetrically
            std = np.max(encoder.predict_std(direction))
            delta_compressed = encoder.coder_update(direction, real_heights)
            recovered_heights = decoder.decoder_update(direction, delta_compressed)
            # 3. Advance hardware global axis simulator track
            enc_origin[1] += 1

            # 4. Evaluation and Cross-Sync metrics tracking
            window_sync_diff = np.linalg.norm(encoder.Z_window - decoder.Z_window)
            sync_status = "SYNCHRONIZED" if window_sync_diff < 1e-7 else "DIVERGED!"

            # Reconstruction accuracy at processing front edge profile
            max_err = np.max(np.abs(np.array(true_heights) - np.array(recovered_heights)))
            # Distance of internal model buffer to pure analog reference window
            true_window = true_beta.get_3x3_window(enc_origin[0], enc_origin[1])
            true_window_diff = np.linalg.norm(encoder.Z_window - true_window)

            print(
              f"{step:<5} | {max_err:<14.4f} | {std:<6.4f} |{true_window_diff:<16.4f} | {encoder.gain_alpha:<12.4f} | {sync_status}")
            assert step < 5 or max_err < 3 * std + 1.e-6, f"Failed on step {step}"

        print("-" * 80)
        print(f"Final True Surface Distance on Window Matrix: {true_window_diff:.6f}")


def test_parabolic_adaptation_with_linear_noise():
    print("\n[TEST 2] Running long-horizon test with Dynamic Linear Sensor Noise...")
    
    # Seed the random generator for strict test reproducibility
    np.random.seed(42)

    # True global paraboloid coefficients
    true_beta = Surface([0.05, 0.03, -0.02, 0.2, -0.1, 128.0])

    b = 0.1
    a = 0.005
    sigma_0 = np.sqrt(a * 128. + b)

    # Initialize filters with gamma=0.15 for smooth noise averaging over time
    encoder = EvansWindowFilter(alpha_noise = a, sigma_0=sigma_0, gamma=0.15, init_brightness=128.0)
    decoder = EvansWindowFilter(alpha_noise = a, sigma_0=sigma_0, gamma=0.15, init_brightness=128.0)

    enc_origin = [0, 0]
    direction = 0
    rel_coords = [(-1, 2), (0, 2), (1, 2)]

    total_steps = 40  # Give it more steps to average out the stochastic noise
    print(f"\n--- Starting {total_steps}-Step Noisy Walk ---")
    print(f"{'Step':<6} | {'Max Error':<10} | {'Std':<10} | {'Window True Diff':<16} | {'Status'}")
    print("-" * 65)

    for step in range(1, total_steps + 1):
        # Coder samples the NOISY physical sensor matrix
        noisy_heights = [true_beta.sample_noisy_sensor(enc_origin[0] + dr, enc_origin[1] + dc, a, b) for dr, dc in rel_coords]
        # Ideal true reference for error observation
        ideal_heights = [true_beta.evaluate(enc_origin[0] + dr, enc_origin[1] + dc) for dr, dc in rel_coords]
        std = np.max(encoder.predict_std(direction))
        # Process through the synchronous loop
        delta_compressed = encoder.coder_update(direction, noisy_heights)
        recovered_heights = decoder.decoder_update(direction, delta_compressed)

        enc_origin[1] += 1

        # Check stream sync between encoder and decoder matrices
        sync_diff = np.linalg.norm(encoder.Z_window - decoder.Z_window)
        status = "SYNCHRONIZED" if sync_diff < 1e-7 else "DIVERGED!"

        max_recon_err = np.max(np.abs(np.array(ideal_heights) - np.array(recovered_heights)))
        # Distance to true mathematical window
        true_window = true_beta.get_3x3_window(enc_origin[0], enc_origin[1])
        true_window_diff = np.linalg.norm(encoder.Z_window - true_window)
        print(f"{step:<6} | {max_recon_err:<10.4f} |  {std:<10.4f} |{true_window_diff:<16.4f} | {status}")
        assert step < 5 or max_recon_err < 3 * std + 1.e-6, f"Failed on step {step}"

    print("-" * 65)
    print("True Ideal Reference Window 3x3 at step 25:")
    print(np.round(true_window, 2))
    print("Recovered Filtered Window 3x3 (Denoised State):")
    print(np.round(encoder.Z_window, 2))
    print(f"Final Euclidean distance to true surface grid: {true_window_diff:.6f}")
    print("SUCCESS: THE 9-POINT WINDOW FILTER PROVED ROBUST AGAINST LINEAR NOISE!")


if __name__ == "__main__":
    run_comprehensive_test_suite()
    test_parabolic_adaptation_with_linear_noise()
