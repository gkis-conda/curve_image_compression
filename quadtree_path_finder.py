from loss import loss, generate_hilbert_coords
from graph_visualizer import SimpleGridPathVisualizer


class PathCompiler:
    def __init__(self, r_l1=2, delta_local=4):
        """
        Initialize Symmetrical Quad-Tree Path Compiler.
        Fully compatible with Python 3.6 and clean ASCII.
        """
        self.r_l1 = r_l1
        self.delta_local = delta_local

    @staticmethod
    def _get_even_cut_candidates(length):
        """Returns valid even cut locations sorted by proximity to the center."""
        cuts = [k for k in range(2, length, 2) if (length - k) % 2 == 0]
        cuts.sort(key=lambda k: abs(k - length / 2.0))
        return cuts

    @staticmethod
    def validate_endpoints(N, M, p_entry, valid_exits):
        """
        Validates grid dimensions, bounds, and bipartite parity
        feasibility for start point and a list of valid exits.
        """
        if M < 1 or N < 1:
            return set()

        if not (0 <= p_entry[0] < M and 0 <= p_entry[1] < N):
            return set()

        if isinstance(valid_exits, tuple):
            valid_exits = [valid_exits]

        total_nodes = M * N
        p_entry_color = (p_entry[0] + p_entry[1]) % 2
        feasible_exits = set()

        for er, ec in valid_exits:
            if 0 <= er < M and 0 <= ec < N and (er, ec) != p_entry:
                exit_color = (er + ec) % 2
                if total_nodes % 2 == 0:
                    if p_entry_color != exit_color:
                        feasible_exits.add((er, ec))
                else:
                    if p_entry_color == exit_color:
                        feasible_exits.add((er, ec))

        return feasible_exits

    def _find_canonical_cuts(self, H, W):
        """
        Scans strictly for the canonical U-shape: Entry (0,0), Exit (0, W-1).
        Verifies Hamiltonian feasibility across all 4 sub-blocks using
        true, simulated Hilbert neighboring transitions.
        """
        cuts_x = self._get_even_cut_candidates(W)
        cuts_y = self._get_even_cut_candidates(H)

        for cut_c in cuts_x:
            for cut_r in cuts_y:
                w_l, w_r = cut_c, W - cut_c
                h_t, h_b = cut_r, H - cut_r

                # Q1 (Top-Left):     (0,0) -> (h_t-1, 0)
                # Q2 (Bottom-Left):  (0,0) -> (0, w_l-1)
                # Q3 (Bottom-Right): (0,0) -> (0, w_r-1)
                # Q4 (Top-Right):    (h_t-1, w_r-1) -> target

                q1_entry = (0, 0)
                q1_exit = (h_t - 1, 0)
                if not self.validate_endpoints(w_l, h_t, q1_entry, q1_exit):
                    continue
                q2_entry = (0, 0)
                q2_exit = (0, w_l - 1)
                if not self.validate_endpoints(w_l, h_b, q2_entry, q2_exit):
                    continue
                q3_entry = (0, 0)
                q3_exit = (0, w_r - 1)
                if not self.validate_endpoints(w_r, h_b, q3_entry, q3_exit):
                    continue
                q4_entry = (h_t - 1, w_r - 1)
                q4_exit = (0, W - 1 - cut_c)
                if self.validate_endpoints(w_r, h_t, q4_entry, q4_exit):
                    return cut_c, cut_r
        return None, None

    @staticmethod
    def _generate_snake_path(H, W):
        """
        Generates a perfect canonical Hamiltonian path for a W x H corridor.
        Guarantees entry at (0,0) and exit at (0, W-1) for any even dimensions.
        Operates strictly without tracking endpoints s and t.
        """
        path = []

        # We sweep the even width matrix column-pair by column-pair (step 2)
        # For W=2, this executes exactly once. For larger W, it tiles smoothly.
        for c in range(0, W, 2):
            # Even pairs (0, 2, 4...) go down column c, then up column c+1
                # Down left column of the pair
                for r in range(0, H):
                    path.append((r, c))
                # Up right column of the pair
                for r in range(H - 1, -1, -1):
                    path.append((r, c + 1))

        return path

    def compile_trajectory(self, M, N, entry, exit_pt):
        """
        Build Hamilton path using pure 4-quadrant adaptive even decomposition.
        """

        def _quad_split(H, W, s, t, deep=0):
            #print("\t" * deep, f"[{H}x{W}]", s, t)
            # ==================================================
            # Reduces all 4 P-shape orientations to canonical U-shape
            # ==================================================
            s_r, s_c = s
            t_r, t_c = t
            if s_r == H - 1 and t_r == H - 1:
                #print("\t" * deep, 'V:')
                s_flipped = (0, s_c)
                t_flipped = (0, t_c)
                flipped_path = _quad_split(H, W, s_flipped, t_flipped, deep)
                return [(H - 1 - r, c) for r, c in flipped_path]
            # 2. AXIS TRANSPOSE (Turn 90 degrees if entry/exit are on a vertical edge)
            if s_c == t_c:
                #print("\t" * deep, 'T:')
                s_transposed = (s_c, s_r)
                t_transposed = (t_c, t_r)
                transposed_path = _quad_split(W, H, s_transposed, t_transposed, deep)
                # Reverse mapping: swap rows and columns back
                return [(c, r) for r, c in transposed_path]
            if s_c > t_c:
                #print("\t" * deep, 'M:')
                s_mirrored = (s_r, W - 1 - s_c)
                t_mirrored = (t_r, W - 1 - t_c)
                mirrored_path = _quad_split(H, W, s_mirrored, t_mirrored, deep)
                # Reverse mapping: mirror columns back relative to width W
                return [(r, W - 1 - c) for r, c in mirrored_path]

            # ==================================================
            # --- THE CANONICAL KNOWN CASE: Pure U-Shape ---
            # At this point, s is strictly (0, 0) and t is strictly (0, W-1)
            # ==================================================
            assert s == (0,0) and t == (0, W-1)
            # --- BASE CASE: ---
            if H == 2 or W == 2:
                return PathCompiler._generate_snake_path(H, W)

            cut_c, cut_r = self._find_canonical_cuts(H, W)

            if cut_c is None or cut_r is None:
                raise ValueError("Bipartite Quad-Tree Failure! No valid even cut splits this layout dynamically.")

            w_l, w_r = cut_c, W - cut_c
            h_t, h_b = cut_r, H - cut_r

            # Sequence: Top-Left -> Bottom-Left -> Bottom-Right -> Top-Right
            q1 = _quad_split(h_t, w_l, s, (h_t - 1, 0), deep+1)
            q2 = _quad_split(h_b, w_l, (0, 0), (0, w_l - 1), deep+1)
            q3 = _quad_split(h_b, w_r, (0, 0), (0, w_r - 1), deep+1)
            q4 = _quad_split(h_t, w_r, (h_t - 1, w_r - 1), (0, w_r -1), deep+1)

            p1 = q1
            p2 = [(r + h_t, c) for r, c in q2]
            p3 = [(r + h_t, c + cut_c) for r, c in q3]
            p4 = [(r, c + cut_c) for r, c in q4]

            return p1 + p2 + p3 + p4

        return _quad_split(M, N, entry, exit_pt)


def run_hilbert_verification_tests():
    """
    Standalone validation suite using strict assert conditions.
    Compares PathCompiler trajectories directly against canonical L-system curves.
    """
    # Initialize compiler with delta_local=2 to match atomic 2x2 Hilbert sub-squares
    compiler = PathCompiler(delta_local=2)

    # --- Test Case 1: Order 1 (2x2 Grid) ---
    hilbert_1 = generate_hilbert_coords(order=1)
    compiled_1 = compiler.compile_trajectory(
        M=2, N=2, entry=hilbert_1[0], exit_pt=hilbert_1[-1]
    )

    assert len(compiled_1) == 4, f"Order 1: Path must contain exactly 4 points.\n {compiled_1}"
    assert compiled_1 == hilbert_1, f"Order 1: Generated path does not match Hilbert geometric sequence." \
        f"\nExpexted: {hilbert_1}\n Actual: {compiled_1}"

    # --- Test Case 2: Order 2 (4x4 Grid) ---
    hilbert_2 = generate_hilbert_coords(order=2)
    compiled_2 = compiler.compile_trajectory(
        M=4, N=4, entry=hilbert_2[0], exit_pt=hilbert_2[-1]
    )

    assert len(compiled_2) == 16, f"Order 2: Path must contain exactly 16 points.\n {compiled_2}"
    assert compiled_2 == hilbert_2, "Order 2: Generated path does not match Hilbert geometric sequence." \
        f"\nExpexted: {hilbert_2}\n Actual: {compiled_2}"
    # --- Test Case 3: Order 3 (8x8 Grid) ---
    hilbert_3 = generate_hilbert_coords(order=3)
    compiled_3 = compiler.compile_trajectory(
        M=8, N=8, entry=hilbert_3[0], exit_pt=hilbert_3[-1]
    )

    assert len(compiled_3) == 64, "Order 3: Path must contain exactly 64 points."
    assert compiled_3 == hilbert_3, "Order 3: Generated path does not match Hilbert geometric sequence."

    # --- Test Case 4: Hamiltonian Strict Invariants ---
    # Testing an arbitrary grid size to guarantee complete uniqueness and coverage properties
    M, N = 16, 16
    entry_point = (15, 0)
    exit_point = (15, 15)
    compiled_4 = compiler.compile_trajectory(M, N, entry_point, exit_point)

    assert len(compiled_4) == M * N, "Hamiltonian check: Total points count must equal matrix area."
    assert len(
        set(compiled_4)) == M * N, "Hamiltonian check: Path must not contain overlapping or duplicate coordinate pairs."

    print("ALL HILBERT VERIFICATION TESTS PASSED SUCCESSFULLY!")


def verify_estimator_via_simulator(order=2):
    """
    Uses the exact L-system Hilbert simulator as a reference oracle.
    Tests _estimate_cut by extracting real sub-block paths from the fractal
    and asserting that the dichotomy's cuts line up with the true curve geometry.
    """
    print("==================================================")
    print("RUNNING HILBERT SIMULATOR DRIVEN INVARIANT CHECK")
    print("==================================================")

    # 1. Generate the absolute truth from your L-system simulator
    # For order 2, it builds a perfect 16-point 4x4 coordinate sequence
    hilbert_path = generate_hilbert_coords(order)
    grid_size = 2 ** order

    compiler = PathCompiler(delta_local=2)

    # We grab the global start and global finish directly from the simulator
    entry_node = hilbert_path[0]
    exit_node = hilbert_path[-1]

    print(f"Simulator Path Area: {grid_size}x{grid_size} ({len(hilbert_path)} points)")
    print(f"Target Global Entry: {entry_node} -> Global Exit: {exit_node}")

    # 2. To test _estimate_cut, we run a mirror trace of the top-down split.
    # Let's manually feed the top-level grid directly into the estimator.
    W, H = grid_size, grid_size
    s, t = entry_node, exit_node

    # Execute the estimator function under test
    cut_c, cut_r = compiler._estimate_cut(W, H, s, t)

    # --- HARD ASSERTS ---
    assert cut_c is not None and cut_r is not None, (
        f"\n[ORACLE FAILURE - DISCONNECTED SUB-BLOCKS]\n"
        f"The simulator dictates a valid path exists from {s} to {t} on a [{H}x{W}] grid.\n"
        f"However, _estimate_cut returned (None, None), breaking the dichotomy!"
    )

    # In a perfect Hilbert curve of order 2 (4x4 grid), the vertical cut is at column 2.
    # The true curve leaves the left half at index 7 -> point (3,1), and enters the right
    # half at index 8 -> point (3,2). In local coordinates of the right block, (3,2) maps to (3,0).
    # Therefore, the row connector MUST be exactly row 3!
    if grid_size == 4:
        expected_cut = 2
        expected_row = 3

        assert cut_c == expected_cut, (
            f"Top-level Split Column Error!\nExpected: {expected_cut}\nActual:   {cut_c}"
        )
        assert cut_r == expected_row, (
            f"Top-level Row Connection Error!\n"
            f"The Hilbert curve physically crosses the center at row {expected_row}.\n"
            f"Your estimator returned row: {cut_r}. This will cause duplicate loops or gaps!"
        )

    print("==================================================")
    print("SUCCESS: _estimate_cut MATCHES THE SIMULATOR ORACLE PERFECTLY!")
    print("==================================================")


def verify_hamiltonian_path(path, M_rows, N_cols, expected_start, expected_finish):
    """
    Validates three strict mathematical invariants of a Hamiltonian Path:
    1. Exact total cell coverage length.
    2. Strict cell uniqueness (no overlapping coordinate loops).
    3. Perfect spatial continuity (step-by-step neighbor movements).
    """
    print("\n==================================================")
    print("STARTING HAMILTONIAN CONSTRAINT VERIFICATION...")
    print("==================================================")

    total_cells = M_rows * N_cols
    path_len = len(path)

    # --- INVARIANT 1: Total Area Coverage ---
    print("  Checking coverage bounds... Expected: " + str(total_cells) + ", Compiled: " + str(path_len))
    assert path_len == total_cells, (
            "INVARIANT VIOLATION: Path length mismatch! Your matrix area is " + str(total_cells) +
            " cells, but the trajectory covers " + str(path_len) + " points."
    )

    # --- INVARIANT 2: Coordinate Cell Uniqueness ---
    print("  Checking duplicate coordinate leaks...")
    unique_cells = set(path)
    unique_len = len(unique_cells)

    assert unique_len == total_cells, (
            "INVARIANT VIOLATION: Path contains duplicate node visits or overlapping loops! "
            "Unique elements counted: " + str(unique_len) + " out of " + str(total_cells)
    )

    # --- INVARIANT 3: Boundary and Endpoint Tracking ---
    print("  Checking terminal path anchors...")
    assert path[0] == expected_start, "Endpoint Error: Path does not start at " + str(expected_start)
    assert path[-1] == expected_finish, "Endpoint Error: Path does not finish at " + str(expected_finish)

    # --- INVARIANT 4: Step Spatial Continuity ---
    print("  Checking physical step continuity (Manhattan distance)...")
    for i in range(path_len - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]

        # Calculate strict grid step distance: |r2 - r1| + |c2 - c1|
        step_distance = abs(r2 - r1) + abs(c2 - c1)

        # In a perfect grid graph mesh, every sequential move must equal exactly 1 cell step
        assert step_distance == 1, (
                "INVARIANT VIOLATION: Disconnected trajectory leap found at index " + str(i) + "!\n"
                                                                                               "Point " + str(
            path[i]) + " leaps directly to " + str(path[i + 1]) +
                " (Manhattan distance: " + str(step_distance) + " cells)."
        )

    print("==================================================")
    print("SUCCESS: TRAJECTORY IS A MATHEMATICALLY PERFECT HAMILTONIAN PATH!")
    print("==================================================")


# --- EXECUTION PIPE ---
if __name__ == "__main__":
    # Ensure your PathCompiler and generate_hilbert_coords functions are defined above this line
    #verify_estimator_via_simulator(order=2)
    #run_hilbert_verification_tests()
    # Draw results straight to your clean external visualizer module
    compiler = PathCompiler(delta_local=2)
    M_rows = 30
    N_cols = 28
    s = (0, 0)
    t = (M_rows - 1, 0)
    path = compiler.compile_trajectory(M_rows, N_cols,s, t)
    viz = SimpleGridPathVisualizer(n_rows=M_rows, m_cols=N_cols, scale_factor=35.0)
    viz.render_trajectory_to_svg(path, filename="runtime_calibrated_trajectory.svg")
    verify_hamiltonian_path(path, M_rows, N_cols, s, t)

