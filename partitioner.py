import numpy as np
import math
import cv2

class DiophantineBitVectorPartitioner:
    """
    Partitions an image into global anisotropic strips represented as bit vectors.
    A bit value of 0 represents a block of size 5, and 1 represents a block of size 3.
    Uses OpenCV for convolutions and Dynamic Programming for exact 3k + 5m = L solutions.
    """
    def __init__(self):
        # Standard Sobel/Scharr style 3x3 differential kernels
        self.kernel_dx = np.array([[-1.0, 0.0, 1.0],
                                   [-2.0, 0.0, 2.0],
                                   [-1.0, 0.0, 1.0]], dtype=np.float64) / 8.0
                                   
        self.kernel_dy = np.array([[-1.0, -2.0, -1.0],
                                   [ 0.0,  0.0,  0.0],
                                   [ 1.0,  2.0,  1.0]], dtype=np.float64) / 8.0

    def _optimize_axis_dp(self, profile, length):
        """
        Dynamic Programming to track optimal block step layouts.
        Returns a bit vector where 0 means size 5, and 1 means size 3.
        """
        dp_cost = np.full(length + 1, float('inf'), dtype=np.float64)
        dp_parent = np.zeros(length + 1, dtype=np.int32)
        
        # Base anchor
        dp_cost[0] = 0.0
        
        # Linear DP evaluation loop
        for i in range(length + 1):
            if dp_cost[i] == float('inf'):
                continue
                
            # Evaluation path for block size 3
            if i + 3 <= length:
                segment = profile[i : i + 3]
                mean_grad = np.mean(segment) if len(segment) > 0 else 0.0
                cost_3 = (1.0 / (mean_grad + 1e-5)) * 3.0
                
                if dp_cost[i] + cost_3 < dp_cost[i + 3]:
                    dp_cost[i + 3] = dp_cost[i] + cost_3
                    dp_parent[i + 3] = 3
                    
            # Evaluation path for block size 5
            if i + 5 <= length:
                segment = profile[i : i + 5]
                mean_grad = np.mean(segment) if len(segment) > 0 else 0.0
                cost_5 = mean_grad * 5.0
                
                if dp_cost[i] + cost_5 < dp_cost[i + 5]:
                    dp_cost[i + 5] = dp_cost[i] + cost_5
                    dp_parent[i + 5] = 5

        assert dp_cost[length] != float('inf'), f"No exact Diophantine solution for length {length}."
        
        # Reconstruct block configuration back-to-front
        block_sizes = []
        curr = length
        while curr > 0:
            size = dp_parent[curr]
            block_sizes.append(size)
            curr -= size
            
        block_sizes.reverse()
        
        # Explicit mapping: block size 3 -> 1, block size 5 -> 0
        bit_vector = [1 if size == 3 else 0 for size in block_sizes]
                
        return np.array(bit_vector, dtype=np.int32)

    def generate_bit_vectors(self, img):
        """
        Processes image via OpenCV filters, collapses into 1D profiles,
        and computes the row and column allocation bit vectors.
        """
        img_64 = img.astype(np.float64)
        
        # Fast OpenCV hardware-accelerated 2D convolutions with replicate padding
        grad_x = np.abs(cv2.filter2D(img_64, cv2.CV_64F, self.kernel_dx, borderType=cv2.BORDER_REPLICATE))
        grad_y = np.abs(cv2.filter2D(img_64, cv2.CV_64F, self.kernel_dy, borderType=cv2.BORDER_REPLICATE))
        
        # Extract 1D projection axis profiles via mean reduction
        col_profile = np.mean(grad_x, axis=0)  # Collapses vertical data for column-wise tracking
        row_profile = np.mean(grad_y, axis=1)  # Collapses horizontal data for row-wise tracking
        
        # Compute optimal Diophantine bit vector responses
        row_bit_vector = self._optimize_axis_dp(row_profile, img.shape[0])
        col_bit_vector = self._optimize_axis_dp(col_profile, img.shape[1])
        
        return row_bit_vector, col_bit_vector


def calculate_blocks_1d(N, name="Dimension"):
    """
    Pure analytical algorithm supporting k=0.
    Directly solves the linear system for blocks 3 and 5 under 2^k partition.
    """
    # Small dimensions 3 and 5 natively yield k=0 (2^0 = 1 block)
    # For any other dimension, we require evenness for integer Diophantine matching.
    if N not in (3, 5) and N % 2 != 0:
        N += 1

    # Analytical search for k
    k = int(math.floor(math.log(N / 3.0, 2))) if N > 0 else 0
    if k < 0:
        k = 0
    two_k = 2**k

    # Validate lower bound constraint
    if two_k < N / 5.0:
        k += 1
        two_k = 2**k

    # Analytical solution of the Diophantine system
    y = (N - 3 * two_k) // 2
    x = (5 * two_k - N) // 2

    # If constraints conflict, incrementally pad even numbers up to find a fit
    if x < 0 or y < 0:
        return calculate_blocks_1d(N + 2, name)

    return k, x, y, N


def compute_image_grid(width, height):
    """
     Moore-Snoeyink generalized Hilbert curve on a rectangular macro-grid of size \(2^{k_w} \times 2^{k_h}\)
    Computes the 2D macro-grid configuration for an image of size (width x height).
    Returns a dictionary with complete grid specifications.
    """
    k_w, x_w, y_w, final_w = calculate_blocks_1d(width)
    k_h, x_h, y_h, final_h = calculate_blocks_1d(height)

    return {
        "original_size": (width, height),
        "padded_size": (final_w, final_h),
        "width_grid": {"k": k_w, "blocks_total": 2 ** k_w, "blocks_3": x_w, "blocks_5": y_w},
        "height_grid": {"k": k_h, "blocks_total": 2 ** k_h, "blocks_3": x_h, "blocks_5": y_h},
        "macro_mrf_shape": (2 ** k_w, 2 ** k_h)
    }


# =============================================================================
#                                 VALIDATION TEST
# =============================================================================
def run_grid_tests():
    print("Running grid unit tests...")

    # Test Case 1: Standard Power of 2 (1024)
    # Expected: 2^k = 256, x = 128, y = 128
    k, x, y, n_adj = calculate_blocks_1d(1024)
    assert n_adj == 1024, "Error: 1024 should not need padding"
    assert k == 8, "Error: k should be 8 for N=1024"
    assert x == 128 and y == 128, "Error: block mismatch for 1024"
    assert x + y == 2 ** k, "Error: Topological total constraint broken"
    assert 3 * x + 5 * y == n_adj, "Error: Physical size constraint broken"

    # Test Case 2: Standard non-power of 2 (1200)
    # Expected: 2^k = 256, x = 40, y = 216
    k, x, y, n_adj = calculate_blocks_1d(1200)
    assert n_adj == 1200, "Error: 1200 should not need padding"
    assert k == 8, "Error: k should be 8 for N=1200"
    assert x == 40 and y == 216, "Error: block mismatch for 1200"
    assert 3 * x + 5 * y == 1200, "Error: Size equation failed"

    # Test Case 3: Odd number requiring padding (853 -> 854)
    k, x, y, n_adj = calculate_blocks_1d(853)
    assert n_adj == 854, "Error: 853 must be padded to 854"
    assert x + y == 2 ** k, "Error: Total blocks constraint broken"
    assert 3 * x + 5 * y == n_adj, "Error: Sum of dimensions broken"

    # Test Case 4: Smallest valid constraint check
    k, x, y, n_adj = calculate_blocks_1d(6)
    assert n_adj == 6 and x == 2 and y == 0 and k == 1, "Error: Smallest constraint failed"

    # Test Case 5: Micro bounds check (Handles padding fallback)
    k, x, y, n_adj = calculate_blocks_1d(14)
    # 14 yields a perfect solution (k=2, x=3, y=1) -> 3*3 + 5*1 = 14, total blocks 4 = 2^2
    assert n_adj == 14, "Error: 14 can be perfectly partitioned without padding"
    assert k == 2, "Error: k should be 2 for N=14"
    assert x == 3 and y == 1, "Error: block mismatch for 14"
    assert 3 * x + 5 * y == 14, "Error: Size match failed"

    # Test Case 6: Full 2D Image processing simulation
    grid_data = compute_image_grid(1300, 853)
    assert grid_data["original_size"] == (1300, 853)
    assert grid_data["padded_size"] == (1300, 854), grid_data

    # Verify horizontal line
    w_g = grid_data["width_grid"]
    assert w_g["blocks_3"] + w_g["blocks_5"] == w_g["blocks_total"]
    assert 3 * w_g["blocks_3"] + 5 * w_g["blocks_5"] == grid_data["padded_size"][0]

    # Verify vertical line
    h_g = grid_data["height_grid"]
    assert h_g["blocks_3"] + h_g["blocks_5"] == h_g["blocks_total"]
    assert 3 * h_g["blocks_3"] + 5 * h_g["blocks_5"] == grid_data["padded_size"][1]

    # Verify that MRF shape matches 2^k_w x 2^k_h topology
    assert grid_data["macro_mrf_shape"] == (w_g["blocks_total"], h_g["blocks_total"])

    print("All tests passed successfully! [OK]")


def run_bit_vector_test():
    print("[TEST] Running OpenCV and Dynamic Programming bit vector extraction...")
    
    # Generate an arbitrary 14x11 image (14 = 3*3 + 5*1; 11 = 3*2 + 5*1)
    # Inject a clean horizontal ridge to trigger size-3 refinement along rows
    mock_img = np.zeros((14, 11), dtype=np.uint8)
    mock_img[6:8, :] = 255
    
    partitioner = DiophantineBitVectorPartitioner()
    row_vec, col_vec = partitioner.generate_bit_vectors(mock_img)
    
    # Verify bit vectors data formats
    print(f" -> Generated Row Bit Vector: {list(row_vec)}")
    print(f" -> Generated Col Bit Vector: {list(col_vec)}")
    
    # Check that lengths successfully sum up to match image dimensions
    # According to the spec: 1 -> block of size 3, 0 -> block of size 5
    total_row_len = sum(3 if b == 1 else 5 for b in row_vec)
    total_col_len = sum(3 if b == 1 else 5 for b in col_vec)
    
    print(f" -> Decoded boundary lengths: Rows={total_row_len}, Cols={total_col_len}")
    assert total_row_len == 14, "Row bit allocation does not sum to height boundary!"
    assert total_col_len == 11, "Col bit allocation does not sum to width boundary!"
    print("[SUCCESS] OpenCV differential pipeline and bit vectors matching passed!")


if __name__ == "__main__":
    run_bit_vector_test()
    run_grid_tests()

    # Print a sample output for verification
    sample_result = compute_image_grid(1920, 1080)
    print("\nSample Execution for 1920x1080 Full HD Image:")
    print(sample_result)
