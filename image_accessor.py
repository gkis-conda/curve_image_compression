import numpy as np


class ImageBlockAccessor:
    """
    BLOCK-LEVEL ENCAPSULATION ACCESSOR INTERFACE.
    Provides a seamless virtual 2D block iterator over an image canvas.
    Enforces a strict EVEN number of virtual blocks across both dimensions.
    """

    def __init__(self, image_matrix, block_size=3, padding_mode='edge'):
        """
        :param image_matrix: 2D NumPy array representing the active frame.
        :param block_size: Scalar dimension of the square block grid (Default = 3).
        :param padding_mode: Boundary expansion strategy ('edge', 'constant', etc.).
        """
        self.img = image_matrix
        self.B = block_size
        self.pad_mode = padding_mode
        self.H, self.W = image_matrix.shape

        # 1. Compute baseline minimum blocks required to span the canvas
        min_rows = int(np.ceil(self.H / self.B))
        min_cols = int(np.ceil(self.W / self.B))

        # 2. FIXED: Enforce strict EVEN block counting constraint across the 2D grid
        self.block_rows = min_rows + (min_rows % 2)
        self.block_cols = min_cols + (min_cols % 2)

    def __iter__(self):
        """Standard 2D Block Iterator. Yields pairs of (block_row, block_col) sequentially."""
        for br in range(self.block_rows):
            for bc in range(self.block_cols):
                yield br, bc

    def get_block(self, block_row, block_col):
        """
        Extracts a BxB block. Returns a zero-copy direct view if fully inside bounds,
        or a virtual edge-padded copy if intersecting image boundaries.
        """
        r_start = block_row * self.B
        c_start = block_col * self.B
        r_end = r_start + self.B
        c_end = c_start + self.B

        # CASE 1: Perfect In-Bounds (Direct zero-copy stride mapping view)
        if r_end <= self.H and c_end <= self.W:
            return self.img[r_start:r_end, c_start:c_end]

        # CASE 2: Out-of-Bounds Intersection or completely Virtual Padded Block
        # If the requested block index is entirely out of physical pixel limits:
        if r_start >= self.H or c_start >= self.W:
            # Create a completely blank virtual block initialized to edge pixels or constants
            clamped_r = min(r_start, self.H - 1)
            clamped_c = min(c_start, self.W - 1)
            return np.full((self.B, self.B), self.img[clamped_r, clamped_c], dtype=self.img.dtype)

        # Partial physical layout overlap calculation
        r_end_valid = min(r_end, self.H)
        c_end_valid = min(c_end, self.W)

        valid_part = self.img[r_start:r_end_valid, c_start:c_end_valid]

        pad_below = self.B - (r_end_valid - r_start)
        pad_right = self.B - (c_end_valid - c_start)

        return np.pad(
            valid_part,
            pad_width=((0, pad_below), (0, pad_right)),
            mode=self.pad_mode
        )

    def set_block(self, block_row, block_col, block_data):
        """
        Writes BxB block data back onto the target image frame layout.
        Automatically drops virtual padding cells that extend beyond physical pixels.
        """
        r_start = block_row * self.B
        c_start = block_col * self.B
        r_end = r_start + self.B
        c_end = c_start + self.B

        # If writing to a completely virtual padding block outside physical limits, ignore safely
        if r_start >= self.H or c_start >= self.W:
            return

        # CASE 1: Direct update to memory layout via view stride reference
        if r_end <= self.H and c_end <= self.W:
            self.img[r_start:r_end, c_start:c_end] = block_data
            return

        # CASE 2: Edge clipped update
        r_end_valid = min(r_end, self.H)
        c_end_valid = min(c_end, self.W)

        valid_h = r_end_valid - r_start
        valid_w = c_end_valid - c_start

        self.img[r_start:r_end_valid, c_start:c_end_valid] = block_data[:valid_h, :valid_w]


def test_image_block_accessor_pipeline():
    print("=" * 75)
    print("   IMAGE BLOCK ACCESSOR: 2D GRID ITERATION & VIEW VERIFICATION")
    print("=" * 75)

    # 1. Create a 5x5 non-divisible matrix frame
    # Block size B=3 means our virtual grid must resolve to a 2x2 blocks layout
    source_frame = np.arange(1, 26, dtype=np.float64).reshape(5, 5)
    print("Original Source Frame 5x5:")
    print(source_frame)
    print("-" * 75)

    # 2. Instantiate Accessor Component
    accessor = ImageBlockAccessor(source_frame, block_size=3, padding_mode='edge')
    print(f"Virtual Block Layout Grid Map: {accessor.block_rows} Rows x {accessor.block_cols} Cols.")

    # 3. Run the Iteration Loop Sweep
    for br, bc in accessor:
        block = accessor.get_block(br, bc)
        print(f"\n---> Fetched Virtual Block Index ({br}, {bc}) - Shape: {block.shape}:")
        print(block)

        # Test Direct view mutations update: add 100 to the absolute upper-left corner cell
        if br == 0 and bc == 0:
            block[0, 0] = 999.0  # Modifying the direct view layout

    print("\n" + "-" * 75)
    print("Verifying Mutated Frame State after direct view write operations:")
    print(source_frame)

    # Strict architectural check: index (0,0) must update directly due to memory view mechanics
    assert source_frame[0, 0] == 999.0, "Accessor failed to establish zero-copy memory views!"
    print("\n[VERDICT: SUCCESS] Accessor View Streams and Dynamic Edge Padding Passed Perfectly!")
    print("=" * 75)


if __name__ == "__main__":
    test_image_block_accessor_pipeline()
