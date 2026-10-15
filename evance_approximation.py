import numpy as np

def idx(length):
    return [0, 2, 4] if length == 5 else [0, 1, 2]
  
class EvansInterpolationPredictor:
    """
    Evans Paraboloid Predictor with static 3x3 kernel initialization
    and step-based scaling for arbitrary (H, W) block shapes.
    Calculations are strictly based on matrix/array index coordinates.
    """
    def __init__(self):
        # 1. Static invariant 3x3 Evans kernels
        self.A_kernel = np.array([[ 1.0, -2.0,  1.0], [ 1.0, -2.0,  1.0], [ 1.0, -2.0,  1.0]], dtype=np.float64) / 6.0
        self.B_kernel = np.array([[ 1.0,  1.0,  1.0], [-2.0, -2.0, -2.0], [ 1.0,  1.0,  1.0]], dtype=np.float64) / 6.0
        self.C_kernel = np.array([[ 1.0,  0.0, -1.0], [ 0.0,  0.0,  0.0], [-1.0,  0.0,  1.0]], dtype=np.float64) / 4.0
        self.D_kernel = np.array([[-1.0,  0.0,  1.0], [-1.0,  0.0,  1.0], [-1.0,  0.0,  1.0]], dtype=np.float64) / 6.0
        
        # FIXED: Inverted row kernel signs to match downward-growing row indexes (r_rel)
        self.E_kernel = np.array([[-1.0, -1.0, -1.0], [ 0.0,  0.0,  0.0], [ 1.0,  1.0,  1.0]], dtype=np.float64) / 6.0
        self.F_kernel = np.array([[-1.0,  2.0, -1.0], [ 2.0,  5.0,  2.0], [-1.0,  2.0, -1.0]], dtype=np.float64) / 9.0
        
        # Initialize 3x3 state matrix S
        self.S = np.eye(3, dtype=np.float64)

    def set_surface_matrix(self, Z_window):
        """
        Computes symmetric 3x3 S matrix for any Z_window shape (H, W).
        """
        height, width = Z_window.shape
        assert height in (3, 5) and width in (3, 5), "Shape must be combination of 3 and 5"

        # Resample regular 3x3 grid from input Z window
        Z_3x3 = Z_window[np.ix_(idx(height), idx(width))]

        # Base 3x3 tensor contraction
        a_raw = float(np.sum(self.A_kernel * Z_3x3))
        b_raw = float(np.sum(self.B_kernel * Z_3x3))
        c_raw = float(np.sum(self.C_kernel * Z_3x3))
        d_raw = float(np.sum(self.D_kernel * Z_3x3))
        e_raw = float(np.sum(self.E_kernel * Z_3x3))
        f_raw = float(np.sum(self.F_kernel * Z_3x3))

        # Scale steps for index spacing
        lambda_y = 1.0 / ((height - 1) // 2)
        lambda_x = 1.0 / ((width - 1) // 2)
        
        a = a_raw * (lambda_x ** 2)
        b = b_raw * (lambda_y ** 2)
        c = c_raw * (lambda_x * lambda_y)
        d = d_raw * lambda_x
        e = e_raw * lambda_y
        f = f_raw
        
        # Assemble canonical symmetric matrix S
        self.S = np.array([
            [a,         c / 2.0,   d / 2.0],
            [c / 2.0,   b,         e / 2.0],
            [d / 2.0,   e / 2.0,   f]
        ], dtype=np.float64)

        return self.S

    def interpolate_points(self, x_array, y_array):
        """
        Fast vectorized quadratic evaluation: z = X * S * X^T
        """
        x_arr = np.asarray(x_array, dtype=np.float64)
        y_arr = np.asarray(y_array, dtype=np.float64)
        assert x_arr.shape == y_arr.shape, "Coordinate arrays must match in size"

        # Build homogeneous coordinates matrix N x 3
        X = np.column_stack((x_arr, y_arr, np.ones_like(x_arr)))

        # Matrix optimization chunk for parallel quad evaluation
        return np.sum(X * (X @ self.S), axis=1)


# =============================================================================
#                               PURE ASSERT TESTS
# =============================================================================

def run_tests():
    print("[TEST] Running EvansInterpolationPredictor in pure RC-space assertions...")

    predictor = EvansInterpolationPredictor()

    # Test all 4 anisotropic block shapes: (3x3, 3x5, 5x3, 5x5)
    test_shapes = [(3, 3), (3, 5), (5, 3), (5, 5)]

    for height, width in test_shapes:
        lambda_r = (height - 1) // 2
        lambda_c = (width - 1) // 2

        # Initialize test data block
        Z_block = np.zeros((height, width), dtype=np.float64)
        
        # Populate matrix with ideal paraboloid relative to central indices
        for r in range(height):
            for c in range(width):
                r_rel = float(r - lambda_r)  # Relative row offset from center
                c_rel = float(c - lambda_c)  # Relative col offset from center
                
                Z_block[r, c] = 100.0 + 2.0 * c_rel - 3.0 * r_rel + 0.5 * (c_rel**2) + 0.25 * (r_rel**2)

        # Calculate S matrix state
        S = predictor.set_surface_matrix(Z_block)
        assert S.shape == (3, 3), "Surface matrix S must be 3x3"

        # Evaluate at relative offset: 2 columns right, 1 row down
        # Expected value: 100 + 2*(2) - 3*(1) + 0.5*(2^2) + 0.25*(1^2)
        # = 100 + 4 - 3 + 2 + 0.25 = 103.25
        pred_z = predictor.interpolate_points([2.0], [1.0])[0]
        
        # Strict floating point delta check
        assert abs(pred_z - 103.25) < 1e-7, "Prediction failed for block shape ({}, {})".format(height, width)

    print("[SUCCESS] All pure RC-space predictor tests passed successfully!")


if __name__ == "__main__":
    run_tests()
