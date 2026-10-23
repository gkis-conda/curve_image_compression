import numpy as np
import numba
from quantizers import IdentityQuantizer

class EvansSlidingFilter:
    # Direction identifiers mapping: 0: EAST, 1: SOUTH, 2: WEST, 3: NORTH
    _lookahead_coords = [
        [(-1, 2), (0, 2), (1, 2)],  # 0: EAST  (Target profile sits at c + 2)
        [(2, -1), (2, 0), (2, 1)],  # 1: SOUTH (Target profile sits at r + 2)
        [(-1, -2), (0, -2), (1, -2)],  # 2: WEST  (Target profile sits at c - 2)
        [(-2, -1), (-2, 0), (-2, 1)]  # 3: NORTH (Target row sits at r - 2)
    ]

    def __init__(self, alpha_noise=0.0, sigma_0=0.0, gamma=0.15, quantize_fn=None,
                 quantized_signal=False, adaptive_noise=False):
        """
        Initialize the Canvas-Buffered Evans Filter.
        Completely self-contained: extracts actual pixels internally from the canvas.
        """
        self.alpha_noise = alpha_noise
        self.beta_noise = sigma_0 ** 2 - (self.alpha_noise * 128.0)
        self.quantization_noise = 1 / 12. if quantized_signal else 0.
        self.quantize_fn =  self.quantize_fn = quantize_fn if quantize_fn is not None else IdentityQuantizer()
        self.flat_neighborhood = np.empty(9, dtype=np.float64)

        if self.quantize_fn:
            self.quantization_noise += self.quantize_fn.sigma_quant_sq

        # FIXED: Gain regularization guarantee locks the lower noise floor limit
        if self.quantization_noise < 1.e-6:
            self.quantization_noise = 1.e-6

        self.gamma = gamma
        self.e_relief = 0.0
        self.adaptive_noise = adaptive_noise

        # Combined projection operators (Maps 9-cell window directly to 3-cell prediction)
        self._lookahead = [EvansSlidingFilter._get_lookahead(i) for i in range(4)]
        # Precompute standard 9-point OLS fit matrices
        #self.X_9, self.cov, self.K_fit = EvansSlidingFilter._precompute_paraboloid_fit_matrix()
        #self._predict_operators = [self._lookahead[i] @ self.K_fit for i in range(4)]
        self.X_9, self.cov, self.K_fit = self._precompute_directional_paraboloid_operators()
        self._predict_operators = [self._lookahead[i] @ self.K_fit[i] for i in range(4)]

        self.gain_alpha = 0.
        self.total_noise = sigma_0 ** 2 + self.quantization_noise

    @staticmethod
    def _gen_projection(coords):
        X = np.array([[float(r ** 2), float(c ** 2), float(r * c), float(r), float(c), 1.0]
                      for r, c in coords], dtype=np.float64)
        return X

    @staticmethod
    def _precompute_paraboloid_fit_matrix():
        coords = [(r, c) for r in [-1, 0, 1] for c in [-1, 0, 1]]
        X = EvansSlidingFilter._gen_projection(coords)
        cov = np.linalg.inv(X.T @ X)
        return X, cov, cov @ X.T

    @staticmethod
    def _get_lookahead(direction_idx) -> np.ndarray:
        coords = EvansSlidingFilter._lookahead_coords[direction_idx]
        return np.array([[float(r ** 2), float(c ** 2), float(r * c), float(r), float(c), 1.0]
                         for r, c in coords], dtype=np.float64)

    @staticmethod
    def _precompute_directional_paraboloid_operators():
        """
        +-----------------------------------------------------------------+
        | DIRECTIONAL WLS OPERATORS PRECOMPUTATION                        |
        | Precomputes weighted least-squares projection matrices for      |
        | all 4 spatial directions (EAST, SOUTH, WEST, NORTH).            |
        +-----------------------------------------------------------------+
        """
        coords = [(r, c) for r in [-1, 0, 1] for c in [-1, 0, 1]]
        X = EvansSlidingFilter._gen_projection(coords)
        operators = []
        cov = []
        for i in range(4):
            r0, c0 = EvansSlidingFilter._lookahead_coords[i][1]
            weights = []
            for r, c in coords:
                # --- Compute directional anisotropic weights ---
                distance = np.hypot(r - r0, c - c0)
                weights.append(1/distance)

            W = np.array(weights, dtype=np.float64)

            # --- Weighted Gram matrix: X^T * W * X ---
            XT_W_X = X.T @ (W[:, np.newaxis] * X)
            cov_w = np.linalg.inv(XT_W_X)
            cov.append(cov_w)
            # --- Closed-form WLS operator mapping neighborhood to parameters beta ---
            operator_w = cov_w @ (X.T * W)
            operators.append(operator_w)

        return X, cov, operators


    def coder_update_canvas(self, padded_canvas, r, c, direction):
        """
        DIRECT CANVAS CODER:
        Extracts both the 9-cell neighborhood and the 3-cell target profile
        internally based on coordinate states and direction vector tokens.

        :param padded_canvas: The global 2D float64 image array (with border padding).
        :param r: Current neighborhood center row coordinate index.
        :param c: Current neighborhood center column coordinate index.
        :param direction: Integer direction tracker (0: EAST, 1: SOUTH, 2: WEST, 3: NORTH).
        """
        # 1. Grab local 9-cell neighborhood instantly as a flat, zero-copy memory view
        c_start = c - 1
        c_end = c + 2
        self.flat_neighborhood[0:3] = padded_canvas[r - 1, c_start:c_end]
        self.flat_neighborhood[3:6] = padded_canvas[r, c_start:c_end]
        self.flat_neighborhood[6:9] = padded_canvas[r + 1, c_start:c_end]

        # 2. Compute prediction vector using a single matrix multiplication step
        z_pred = self._predict_operators[direction] @ self.flat_neighborhood

        # 3. ENCAPSULATED TARGET PROFILE SELECTION FROM CANVAS
        if direction == 0:
            z_actual = padded_canvas[r - 1:r + 2, c + 2]  # EAST
        elif direction == 2:
            z_actual = padded_canvas[r - 1:r + 2, c - 2]  # WEST
        elif direction == 1:
            z_actual = padded_canvas[r + 2, c - 1:c + 2]  # SOUTH
        elif direction == 3:
            z_actual = padded_canvas[r - 2, c - 1:c + 2]  # NORTH

        delta = z_actual - z_pred

        # 4. Fast scalar calculation of relief signal energy
        e_t = (delta[0] ** 2 + delta[1] ** 2 + delta[2] ** 2) / 3.0
        mean_intensity = (z_pred[0] + z_pred[1] + z_pred[2])
        mean_intensity = max(0.0, mean_intensity / 3.0)

        sensor_noise = self.alpha_noise * mean_intensity + self.beta_noise
        total_noise = sensor_noise + self.quantization_noise
        self.total_noise = total_noise

        pure_signal_energy = max(0.0, e_t - total_noise)
        self.e_relief = (1. - self.gamma) * self.e_relief + self.gamma * pure_signal_energy

        alpha = self.e_relief / (self.e_relief + total_noise)
        if alpha > 1.0: alpha = 1.0
        self.gain_alpha = alpha

        # 5. Innovation quantization
        delta_filtered = alpha * delta
        delta_compressed = self.quantize_fn.quantize(delta_filtered)
        delta_restored = self.quantize_fn.dequantize(delta_compressed)
        # 6. Overwrite the reconstructed/filtered heights directly onto the canvas layout memory
        y_filtered = z_pred + delta_restored

        if direction == 0:
            padded_canvas[r - 1:r + 2, c + 2] = y_filtered
        elif direction == 2:
            padded_canvas[r - 1:r + 2, c - 2] = y_filtered
        elif direction == 1:
            padded_canvas[r + 2, c - 1:c + 2] = y_filtered
        elif direction == 3:
            padded_canvas[r - 2, c - 1:c + 2] = y_filtered

        return delta_compressed

    def decoder_update_canvas(self, padded_canvas, r, c, direction, delta_compressed):
        """DIRECT CANVAS DECODER: Reconstructs values in-place inside the master canvas."""
        c_start = c - 1
        c_end = c + 2
        self.flat_neighborhood[0:3] = padded_canvas[r - 1, c_start:c_end]
        self.flat_neighborhood[3:6] = padded_canvas[r, c_start:c_end]
        self.flat_neighborhood[6:9] = padded_canvas[r + 1, c_start:c_end]
        z_pred = self._predict_operators[direction] @ self.flat_neighborhood
        delta_restored = self.quantize_fn.dequantize(delta_compressed)
        y_recovered = z_pred + delta_restored

        # Overwrite values straight into the master canvas layout
        if direction == 0:
            padded_canvas[r - 1:r + 2, c + 2] = y_recovered
        elif direction == 2:
            padded_canvas[r - 1:r + 2, c - 2] = y_recovered
        elif direction == 1:
            padded_canvas[r + 2, c - 1:c + 2] = y_recovered
        elif direction == 3:
            padded_canvas[r - 2, c - 1:c + 2] = y_recovered

        return y_recovered
