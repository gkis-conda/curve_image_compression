import numpy as np

class Surface:
    def __init__(self, beta: list):
        """
        Initializes the global surface shape parameters.
        beta = [A, B, C, D, E, F] mapping:
        z(r, c) = A*r^2 + B*c^2 + C*r*c + D*r + E*c + F
        """
        self.beta = np.array(beta, dtype=np.float64)

    def evaluate(self, r_abs, c_abs):
        """
        VECTORIZED EVALUATION OPERATOR:
        Accepts scalars, 1D arrays, or 2D meshgrids of absolute coordinates
        and evaluates the continuous quadratic envelope without loops.
        """
        return (
                self.beta[0] * (r_abs ** 2) +
                self.beta[1] * (c_abs ** 2) +
                self.beta[2] * (r_abs * c_abs) +
                self.beta[3] * r_abs +
                self.beta[4] * c_abs +
                self.beta[5]
        )

    def generate_full_frame(self, height, width):
        """
        FAST FULL-FRAME GENERATOR:
        Generates the clean analytical continuous surface array for the entire grid layout.
        """
        r_grid, c_grid = np.ogrid[:height, :width]
        return self.evaluate(r_grid, c_grid)

    def get_3x3_window(self, origin_r, origin_c):
        """
        VECTORIZED LOCAL LOOKUP:
        Extracts the true mathematical 3x3 reference window centered at (origin_r, origin_c)
        using broadcasting slice matrices.
        """
        r_target = np.array([origin_r-1, origin_r, origin_r+1]).reshape(3, 1)
        c_target = np.array([origin_c-1, origin_c, origin_c+1]).reshape(1, 3)
        return self.evaluate(r_target, c_target)

    def sample_noisy_sensor(self, r_abs, c_abs, a, b):
        """
        VECTORIZED PHYSICAL SENSOR MACHINE:
        Simulates shot-noise and thermal-readout degradation over any coordinate input shape.
        Accepts full 2D coordinate matrices and applies parallelized Gaussian perturbation.
        """
        ideal_val = self.evaluate(r_abs, c_abs)

        # Vectorized Linear Noise Model: variance scales dynamically with signal intensity
        variance = a * np.maximum(0.0, ideal_val) + b
        sigma = np.sqrt(variance)

        # Parallelized Gaussian injection mapping the exact dimensions of the inputs
        noise = np.random.normal(0.0, sigma)
        return ideal_val + noise
