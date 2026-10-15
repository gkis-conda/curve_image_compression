
# =============================================================================
# 1. GENERATORS (Return Start Position, Initial Direction, FLR Code)
# =============================================================================


def generate_u(N):
    """
    Generates U-shaped (loop) trajectory for 2 x N grid.
    Returns: (0,0), (0,1), FLR_string
    """
    assert N >= 2, "N must be >= 2"
    if N == 2:
        flr = "FRR"
    else:
        flr = ("F" * (N - 1)) + "RR" + ("F" * (N - 2))
    return (0, 0), (0, 1), flr


def generate_meander(N):
    """
    Generates Meander (Zig-Zag) trajectory for 2 x N grid.
    Returns: (0,0), (0,1), FLR_string
    """
    assert N >= 2, "N must be >= 2"
    flr_list = "R"
    for i in range(2 * N - 2):
       flr_list += "L" if (i // 2) % 2 == 0 else "R"
    
    return (0, 0), (0, 1), flr_list


def generate_hilbert_coords(order):
    """
    Generates 2D coordinates for Hilbert Curve of given order on (2^order x 2^order) grid.
    Uses L-system grammar:
      A -> - B F + A F A + F B -
      B -> + A F - B F B - F A +
    """
    def expand_l_system(order):
        axiom = "A"
        for _ in range(order):
            next_str = []
            for ch in axiom:
                if ch == 'A':
                    next_str.append("-BF+AFA+FB-")
                elif ch == 'B':
                    next_str.append("+AF-BFB-FA+")
                else:
                    next_str.append(ch)
            axiom = "".join(next_str)
        return axiom

    instructions = expand_l_system(order)
    
    # Direction vectors: 0: Right (0,1), 1: Down (1,0), 2: Left (0,-1), 3: Up (-1,0)
    DR = [0, 1, 0, -1]
    DC = [1, 0, -1, 0]
    
    curr_r, curr_c = 0, 0
    curr_h = 0  # Initial direction: Down (1,0)
    
    coords = [(curr_r, curr_c)]
    
    for cmd in instructions:
        if cmd == '+':
            curr_h = (curr_h - 1) % 4
        elif cmd == '-':
            curr_h = (curr_h + 1) % 4
        elif cmd == 'F':
            curr_r += DR[curr_h]
            curr_c += DC[curr_h]
            coords.append((curr_r, curr_c))
            
    return coords


def generate_snake_coords(width, height):
    """
    Generates standard row-by-row S-snake trajectory for width x height grid.
    """
    coords = []
    for r in range(height):
        if r % 2 == 0:
            for c in range(width):
                coords.append((r, c))
        else:
            for c in range(width - 1, -1, -1):
                coords.append((r, c))
    return coords


def entry_side_to_vec(entry_side):
    """
    Maps initial entry boundary side to inward direction vector (dr, dc).
    """
    mapping = {
        'LEFT': (0, 1),  # Enter from Left -> move Right (+col)
        'RIGHT': (0, -1),  # Enter from Right -> move Left (-col)
        'TOP': (1, 0),  # Enter from Top -> move Down (+row)
        'BOTTOM': (-1, 0)  # Enter from Bottom -> move Up (-row)
    }
    return mapping[entry_side]


def coords_to_flr(coords, entry_side=None):
    """
    DIFFERENTIATOR
    Extracts FLR movement string and initial (P0, D) tuple from absolute 2D coordinate sequence.
    Pre-allocates a fixed-size list for O(N) memory performance.

    Length of returned FLR string is strictly len(coords) - 1.
    """
    n_steps = len(coords) - 1
    assert n_steps >= 1, "Path must have at least 2 points"

    dir_map = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}
    sym_map = {0: "F", 1: "R", 3: "L"}

    P0 = coords[0]
    first_vec = (coords[1][0] - coords[0][0], coords[1][1] - coords[0][1])
    assert first_vec in dir_map, "Invalid step vector at index 0: " + str(first_vec)

    # Determine starting orientation heading
    if entry_side is not None:
        entry_vec = entry_side_to_vec(entry_side)
        curr_h = dir_map[entry_vec]
        initial_D = entry_vec
    else:
        curr_h = dir_map[first_vec]
        initial_D = first_vec

    # Pre-allocate array of exact required size
    flr_arr = [None] * n_steps

    for i in range(n_steps):
        p_curr = coords[i]
        p_next = coords[i + 1]
        step_vec = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])

        assert step_vec in dir_map, "Invalid step vector at index " + str(i) + ": " + str(step_vec)

        target_h = dir_map[step_vec]
        diff = (target_h - curr_h) % 4

        assert diff != 2, "Invalid back step (180 deg) at index " + str(i) + " between " + str(p_curr) + " and " + str(
            p_next)

        flr_arr[i] = sym_map[diff]
        curr_h = target_h

    return P0, initial_D, "".join(flr_arr)


def integrate(P0, D, path):
    """
    INTEGRATOR (Converts P0, D, FLR_path -> List of (r, c) Coordinates)
    Integrates FLR differential path starting from P0 with initial direction D.
    D vector map: (0,1) -> Right, (1,0) -> Down, (0,-1) -> Left, (-1,0) -> Up
    """
    DR = [0, 1, 0, -1]
    DC = [1, 0, -1, 0]
    dir_map = {(0, 1): 0, (1, 0): 1, (0, -1): 2, (-1, 0): 3}

    assert D in dir_map, "Invalid direction vector D"
    curr_h = dir_map[D]
    curr_r, curr_c = P0

    coords = [(curr_r, curr_c)]

    for cmd in path:
        if cmd == "R":
            curr_h = (curr_h + 1) % 4
        elif cmd == "L":
            curr_h = (curr_h - 1) % 4
        elif cmd == "F":
            pass
        else:
            assert False, "Invalid FLR command '%s'" % cmd

        curr_r += DR[curr_h]
        curr_c += DC[curr_h]
        coords.append((curr_r, curr_c))

    return coords


def loss(coords, r_l1=2, delta_thresh=4):
    K = len(coords)
    if K < 2:
        return 0.0

    local_count = 0
    distant_count = 0

    for i in range(K):
        p1 = coords[i]
        for j in range(i + 1, K):
            p2 = coords[j]
            l1_dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
            if l1_dist <= r_l1:
                curve_dist = j - i
                if curve_dist <= delta_thresh:
                    local_count += 1
                else:
                    distant_count += 1

    rupture_ratio = distant_count / local_count

    return rupture_ratio


def find_path__(M, N, p_entry, valid_exits, r_l1=2, delta_local=4):
    total_nodes = M * N

    if isinstance(valid_exits, tuple):
        valid_exits = [valid_exits]

    # 1. PARITY CHECK
    p_entry_color = (p_entry[0] + p_entry[1]) % 2
    feasible_exits = set()

    for er, ec in valid_exits:
        if 0 <= er < M and 0 <= ec < N:
            exit_color = (er + ec) % 2
            if total_nodes % 2 == 0:
                if p_entry_color != exit_color:
                    feasible_exits.add((er, ec))
            else:
                if p_entry_color == exit_color:
                    feasible_exits.add((er, ec))

    if not feasible_exits:
        return {"path": None, "loss": float('inf')}

    best_loss = float('inf')
    best_path = None

    DR = [0, 1, 0, -1]
    DC = [1, 0, -1, 0]

    visited = set()
    for r in range(-1, M + 1):
        visited.add((r, -1))
        visited.add((r, N))
    for c in range(-1, N + 1):
        visited.add((-1, c))
        visited.add((M, c))

    def is_isolated(r, c, visited):
        # Returns True if cell (r, c) has 0 free unvisited neighbors.
        # Early returns False as soon as 1 free neighbor is found.
        for i in range(4):
            if (r + DR[i], c + DC[i]) not in visited:
                return False
        return True

    def dfs(r, c, visited, path_coords, current_loss):
        nonlocal best_loss, best_path
        path_len = len(path_coords)
        # BASE CASE
        if path_len == total_nodes:
            if (r, c) in feasible_exits:
                best_loss = current_loss
                best_path = list(path_coords)
            return

        # 3. DEAD-END ORPHAN PRUNING
        # Check performed once before the loop
        if path_len < total_nodes - 1:
            for i in range(4):
                nr, nc = r + DR[i], c + DC[i]
                if (nr, nc) not in visited:
                    if is_isolated(nr, nc, visited):
                        return

        # 4. NEIGHBOR EXPLORATION
        for i in range(4):
            nr, nc = r + DR[i], c + DC[i]

            if (nr, nc) not in visited:
                path_coords.append((nr, nc))
                next_loss = loss(path_coords, r_l1=r_l1, delta_thresh=delta_local)

                if next_loss < best_loss:
                    visited.add((nr, nc))
                    dfs(nr, nc, visited, path_coords, next_loss)
                    visited.remove((nr, nc))

                path_coords.pop()

    # START EXPLORATION
    visited.add(p_entry)
    initial_path = [p_entry]
    initial_loss = 0

    dfs(p_entry[0], p_entry[1], visited, initial_path, initial_loss)

    return {
        "path": best_path,
        "loss": best_loss
    }

# Hilbert loss thresholds map: (M, N) -> max_allowed_loss
HILBERT_LOSS_BOUNDS = {
    (4, 4): 0.32,
    (8, 8): 0.59,
    (16, 16): 0.74,
    (32, 32): 0.83
}


def find_path(M, N, p_entry, valid_exits, r_l1=2, delta_local=4):
    total_nodes = M * N

    if isinstance(valid_exits, tuple):
        valid_exits = [valid_exits]

    # 1. PARITY CHECK
    p_entry_color = (p_entry[0] + p_entry[1]) % 2
    feasible_exits = set()

    for er, ec in valid_exits:
        if 0 <= er < M and 0 <= ec < N:
            exit_color = (er + ec) % 2
            if total_nodes % 2 == 0:
                if p_entry_color != exit_color:
                    feasible_exits.add((er, ec))
            else:
                if p_entry_color == exit_color:
                    feasible_exits.add((er, ec))

    if not feasible_exits:
        return {"path": None, "loss": float('inf')}

    # 2. HARD CEILING INITIALIZATION FROM HILBERT TABLE
    # If explicit bound is known, start with it; otherwise fallback to inf
    max_allowed_loss = HILBERT_LOSS_BOUNDS.get((M, N), float('inf'))
    best_loss = max_allowed_loss
    best_path = None

    DR = [0, 1, 0, -1]
    DC = [1, 0, -1, 0]

    # 3. SENTINEL BOUNDS
    visited = set()
    for r in range(-1, M + 1):
        visited.add((r, -1))
        visited.add((r, N))
    for c in range(-1, N + 1):
        visited.add((-1, c))
        visited.add((M, c))

    def is_isolated(r, c, visited):
        for i in range(4):
            if (r + DR[i], c + DC[i]) not in visited:
                return False
        return True

    def dfs(r, c, visited, path_coords, current_loss):
        nonlocal best_loss, best_path
        path_len = len(path_coords)

        # BASE CASE
        if path_len == total_nodes:
            if (r, c) in feasible_exits:
                if current_loss <= best_loss:
                    best_loss = current_loss
                    best_path = list(path_coords)
            return

        # DEAD-END ORPHAN PRUNING
        if path_len < total_nodes - 1:
            for i in range(4):
                nr, nc = r + DR[i], c + DC[i]
                if (nr, nc) not in visited:
                    if is_isolated(nr, nc, visited):
                        return

        # NEIGHBOR EXPLORATION WITH CEILING CUTOFF
        for i in range(4):
            nr, nc = r + DR[i], c + DC[i]

            if (nr, nc) not in visited:
                path_coords.append((nr, nc))
                next_loss = loss(path_coords, r_l1=r_l1, delta_thresh=delta_local)

                # HARD CEILING PRUNING: Reject branch immediately if loss > bound
                if next_loss <= best_loss:
                    visited.add((nr, nc))

                    dfs(nr, nc, visited, path_coords, next_loss)

                    visited.remove((nr, nc))

                    # Stop search as soon as a valid path meeting Hilbert bound is found
                    if best_path is not None:
                        path_coords.pop()
                        return

                path_coords.pop()

    # START EXPLORATION
    visited.add(p_entry)
    initial_path = [p_entry]
    initial_loss = 0

    dfs(p_entry[0], p_entry[1], visited, initial_path, initial_loss)

    return {
        "path": best_path,
        "loss": best_loss
    }

def get_side_nodes(M, N, side):
    """
    Returns list of coordinate tuples on the requested boundary side.
    """
    if side == 'LEFT':
        return [(r, 0) for r in range(M)]
    elif side == 'RIGHT':
        return [(r, N - 1) for r in range(M)]
    elif side == 'TOP':
        return [(0, c) for c in range(N)]
    elif side == 'BOTTOM':
        return [(M - 1, c) for c in range(N)]
    else:
        raise ValueError("Invalid side specified: " + str(side))


def find_flr_path(M, N, p0=(0, 0), entry_side='LEFT', exit_side='RIGHT', r_l1=2, delta_local=4):
    """
    Top-level wrapper around terminal micro-solver find_path.
    1. Generates candidate exit boundary coordinates.
    2. Runs find_path DFS solver to obtain optimal Hamiltonian path & loss.
    3. Converts absolute path coordinates to relative FLR movement string.

    Returns:
    {
        "flr": best_flr,
        "loss": best_loss,
        "exit_node": best_exit_node,
        "path": best_path
    }
    """
    valid_exits = get_side_nodes(M, N, exit_side)
    sol = find_path(M, N, p_entry=p0, valid_exits=valid_exits, r_l1=r_l1, delta_local=delta_local)
    best_path = sol.get("path")
    best_loss = sol.get("loss")

    assert best_path is not None, "No feasible Hamiltonian path found for specified parameters"
    best_exit_node = best_path[-1]
    _, _, best_flr = coords_to_flr(best_path, entry_side=entry_side)

    return {
        "flr": best_flr,
        "loss": best_loss,
        "exit_node": best_exit_node,
        "path": best_path
    }


def get_free_degree(r, c, M, N, visited):
    """
    Calculates dynamic degree (number of unvisited neighbors) for cell (r, c).
    """
    DR = [0, 1, 0, -1]
    DC = [1, 0, -1, 0]
    degree = 0
    for i in range(4):
        nr, nc = r + DR[i], c + DC[i]
        if 0 <= nr < M and 0 <= nc < N and (nr, nc) not in visited:
            degree += 1
    return degree


def test_coords_to_flr():
    """
    Isolated unit test function for coords_to_flr.
    Compatible with pytest.
    """
    # Test path forming a square loop of 4 points (3 steps)
    path_coords = [(0, 0), (0, 1), (1, 1), (1, 0)]

    # 1. Standard run (no entry side)
    p0, d, flr_str = coords_to_flr(path_coords)
    assert p0 == (0, 0), "Expected P0 (0,0), got " + str(p0)
    assert d == (0, 1), "Expected D (0,1), got " + str(d)
    assert flr_str == "FRR", "Expected 'FRR', got: " + str(flr_str)

    # 2. Run with entry_side='TOP' (Entering downwards vector (1, 0))
    p0_top, d_top, flr_top = coords_to_flr(path_coords, entry_side='TOP')
    assert d_top == (1, 0), "Expected D (1,0), got " + str(d_top)
    assert flr_top == "LRR", "Expected 'LRR', got: " + str(flr_top)

    print("[SUCCESS] test_coords_to_flr passed successfully.")

# =============================================================================
#  PROFILE COMPARISON TEST
# =============================================================================
def run_test():
    print("=" * 72)
    print("       METRIC PROFILE COMPARISON: U-SHAPE vs MEANDER (2 x N)")
    print("=" * 72 + "\n")

    n_values = [2, 4, 8, 16, 32, 64, 128, 256]

    print("%-6s | %-12s | %-10s | %-10s | %-12s" % 
          ("N", "Grid Size", "U-shape", "Meander", "Winner"))
    print("-" * 72)

    for N in n_values:
        # Generate triplets
        p0_u, d_u, path_u = generate_u(N)
        p0_m, d_m, path_m = generate_meander(N)

        # Basic invariant check
        assert len(path_u) == 2 * N - 1 , f"{N}: {path_u}"
        coords_u = integrate(p0_u, d_u, path_u)
        assert len(coords_u) == 2 * N and len(set(coords_u)) == 2 * N , f"{N}, {path_u}\n {coords_u}"
        assert len(path_m) == 2 * N - 1, f"{N}: {path_m}"
        coords_m = integrate(p0_m, d_m, path_m)
        assert len(coords_m) == 2 * N and len(set(coords_m)) == 2 * N , f"{N}, {path_m}\n {coords_m}"

        # Compute loss via loss(P0, D, path)
        viol_u = loss(coords_u)
        viol_m = loss(coords_m)

        winner = "MEANDER" if (viol_m < viol_u) else "U-SHAPE"

        print("%-6d | %-12s | %.2f | %.2f | %-12s" % 
              (N, "2x%d" % N, viol_u, viol_m, winner))

    print("-" * 72)
    
    # Detailed breakdown for 2 x 16
    p0_u, d_u, path_u = generate_u(16)
    p0_m, d_m, path_m = generate_meander(16)
    v_u = loss(integrate(p0_u, d_u, path_u))
    v_m = loss(integrate(p0_m, d_m, path_m))

    print("\nDetailed FLR Strings & Profiles for N = 16 (32 cells):")
    print("  U-Path FLR      : %s" % path_u)
    print("  U-Path Loss     : Violations loss = %.2f" % v_u)
    print("  Meander FLR     : %s" % path_m)
    print("  Meander Loss    : Violations loss = %.2f" % v_m)

    assert v_m < v_u, "Meander MUST have strictly fewer violations than U-shape!"
    print("\n[SUCCESS] TEST PASSED! Meander metric profile is mathematically superior.")

# =============================================================================
# 4. BENCHMARK TEST SUITE
# =============================================================================

def run_benchmark():
    print("=" * 80)
    print("      METRIC PROFILE COMPARISON: HILBERT CURVE vs CLASSIC SNAKE")
    print("      Grid Sizes: 4x4, 8x8, 16x16, 32x32")
    print("=" * 80 + "\n")

    orders = [2, 3, 4, 5]  # 4x4, 8x8, 16x16 grids

    print("%-6s | %-10s | %-8s | %-10s | %-10s | %-8s" % 
          ("Order", "Grid Size", "Nodes K", "Snake", "Hilbert", "Winner"))
    print("-" * 80)

    for order in orders:
        side = 2 ** order
        grid_str = "%dx%d" % (side, side)

        # Generate coordinates
        coords_hilbert = generate_hilbert_coords(order)
        coords_snake = generate_snake_coords(side, side)

        # Confirm length and uniqueness
        assert len(coords_hilbert) == side * side and len(set(coords_hilbert)) == side * side
        assert len(coords_snake) == side * side and len(set(coords_snake)) == side * side

        # Validate FLR integration consistency
        p0_h, d_h, path_h = coords_to_flr(coords_hilbert)
        p0_s, d_s, path_s = coords_to_flr(coords_snake)
        assert integrate(p0_h, d_h, path_h) == coords_hilbert , f"{side}, {path_h}\n {coords_hilbert}"
        assert integrate(p0_s, d_s, path_s) == coords_snake , f"{side}, {path_s}\n {coords_snake}"

        # Compute Locality Loss
        viol_s = loss(coords_snake, delta_thresh=4)
        viol_h = loss(coords_hilbert, delta_thresh=4)

        winner = "HILBERT" if (viol_h < viol_s) else "SNAKE"

        print("%-6d | %-10s | %-8d | %10.2f | %10.2f | %-8s" %
              (order, grid_str, len(coords_hilbert), viol_s, viol_h, winner))

    print("-" * 80)
    
    # Detailed breakdown for Order 3 (8x8 = 64 nodes)
    coords_h8 = generate_hilbert_coords(3)
    coords_s8 = generate_snake_coords(8, 8)
    p0_h, d_h, path_h = coords_to_flr(coords_h8)
    p0_s, d_s, path_s = coords_to_flr(coords_s8)
    
    v_s = loss(coords_s8, delta_thresh=4)
    v_h = loss(coords_h8, delta_thresh=4)

    print("\nDetailed FLR Analysis for 8x8 Grid (64 nodes):")
    print("  Classic Snake FLR : %s" % path_s)
    print("  Classic Snake Loss: Violations = %.2f" % v_s)
    print("  Hilbert Curve FLR : %s" % path_h)
    print("  Hilbert Curve Loss: Violations = %.2f" % v_h)

    assert v_h < v_s, "Hilbert curve MUST have strictly lower weighted locality score than classic snake!"
    print("\n[SUCCESS] TEST PASSED! Hilbert curve exhibits superior 2D locality preservation.")


def run_meander_finder():
    print("=" * 85)
    print("      RUNNING STRICT ASSERT BENCHMARK: find_path vs generate_meander")
    print("=" * 85 + "\n")

    for N in [2, 3, 4, 5, 6, 7]:
        p0,_,expected_flr = generate_meander(N)
        res = find_flr_path(M=2, N=N, p0=p0, entry_side='LEFT', exit_side='RIGHT')

        found_flr = res["flr"]
        found_loss = res["loss"]

        print(f"Grid 2x{N} | Expected FLR: {expected_flr:s} | Found FLR: {found_flr} | Loss: {found_loss:.4f}")

        assert found_loss == 0.0, f"[FAIL] Loss for 2x{N} must be strictly 0.0, but got {found_loss}"
        assert found_flr == expected_flr, f"[FAIL] FLR mismatch for 2x{N}: Expected {expected_flr}, got {found_flr}"

    print("\n" + "=" * 85)
    print("[ALL ASSERTS PASSED] find_path() perfectly matches generate_meander() with 0.0 loss!")
    print("=" * 85 + "\n")


def test_roundtrip_compatibility():
    test_coords = [
        (0, 0), (0, 1), (0, 2), (0, 3),
        (1, 3), (1, 2), (1, 1), (1, 0)
    ]

    P0, D, path = coords_to_flr(test_coords)
    
    reconstructed_coords = integrate(P0, D, path)

    print(f"P0: {P0}, D: {D}")
    print(f"FLR Path: {path}")
    print(f"Original len: {len(test_coords)}, FLR len: {len(path)}")

    # Strict Assert
    assert len(path) == len(test_coords) - 1, "FLR path length must be exactly len(coords) - 1"
    assert reconstructed_coords == test_coords, "Reconstructed path does not match original coordinates!"
    
    print("[SUCCESS] coords_to_flr and integrate are 100% compatible and reversible!")


def run_hilbert_vs_find_path():
    """
    New test: compares classical Hilbert 4x4 with find_path DFS output.
    Uses existing: generate_hilbert_coords, coords_to_flr, loss, find_path
    """
    print("=" * 85)
    print("      COMPARING OPTIMIZED find_path(4, 4) WITH CLASSICAL HILBERT CURVE")
    print("=" * 85 + "\n")
    order = 2
    # 1. Classical Hilbert Curve
    hilbert_coords = generate_hilbert_coords(order=order)
    entry_side = "LEFT"
    hilbert_p0, hilbert_d, hilbert_flr = coords_to_flr(hilbert_coords, entry_side=entry_side)
    hilbert_loss = loss(hilbert_coords, r_l1=2, delta_thresh=4)

    print("[EVAL] Classical Hilbert (order {order}):")
    print("       P0: %s, D_in: %s" % (str(hilbert_p0), str(hilbert_d)))
    print("       Exit Node: %s" % str(hilbert_coords[-1]))
    print("       FLR:  %s" % hilbert_flr)
    print("       Loss: %.4f\n" % hilbert_loss)

    print(f"[RUN] Running find_path(M={2**order}, N={2**order}, p0={hilbert_p0}, {entry_side}, exit_side='LEFT')...")
    res = find_flr_path(
        M=2**order, N=2**order,
        p0=hilbert_p0,
        entry_side="LEFT",
        exit_side='LEFT',
        r_l1=2,
        delta_local=4
    )

    found_flr = res["flr"]
    found_loss = res["loss"]
    found_exit = res["exit_node"]

    print("[EVAL] DFS Optimized Path 4x4:")
    print("       Exit Node: %s" % str(found_exit))
    print("       FLR:  %s" % str(found_flr))
    print("       Loss: %.4f\n" % found_loss)

    print("-" * 85)
    if found_loss < hilbert_loss:
        print("[RESULT] find_path found a NEW path that is BETTER than Hilbert! (%.4f < %.4f)" % 
              (found_loss, hilbert_loss))
    elif abs(found_loss - hilbert_loss) < 1e-9:
        if found_flr == hilbert_flr:
            print("[RESULT] MATCH! find_path generated EXACTLY the classical Hilbert Curve!")
        else:
            print("[RESULT] ISOMORPHIC OPTIMUM! find_path found an alternative path with EQUAL loss to Hilbert.")
    else:
        print("[WARNING] Hilbert loss (%.4f) is lower than found path (%.4f)." % 
              (hilbert_loss, found_loss))
    print("=" * 85 + "\n")

def entry_side_to_din(entry_side):
    """
    Converts entry side string ('TOP', 'BOTTOM', 'LEFT', 'RIGHT') 
    into a unique initial direction vector D_in.
    """
    mapping = {
        'TOP': (1, 0),     # Entering from TOP means moving Down
        'BOTTOM': (-1, 0), # Entering from BOTTOM means moving Up
        'LEFT': (0, 1),    # Entering from LEFT means moving Right
        'RIGHT': (0, -1)   # Entering from RIGHT means moving Left
    }
    if entry_side not in mapping:
        raise ValueError("Invalid entry_side: %s. Must be TOP, BOTTOM, LEFT, or RIGHT." % str(entry_side))
    return mapping[entry_side]

def din_to_entry_side(d_in):
    """
    Converts initial direction vector D_in into the corresponding entry side string.
    Inverse of entry_side_to_din.
    """
    mapping = {
        (1, 0): 'TOP',      # Moving Down means entered from TOP
        (-1, 0): 'BOTTOM',  # Moving Up means entered from BOTTOM
        (0, 1): 'LEFT',     # Moving Right means entered from LEFT
        (0, -1): 'RIGHT'    # Moving Left means entered from RIGHT
    }
    if d_in not in mapping:
        raise ValueError("Invalid direction vector d_in: %s." % str(d_in))

    return mapping[d_in]


if __name__ == "__main__":
    test_coords_to_flr()
    test_roundtrip_compatibility()
    run_test()
    run_benchmark()
    run_meander_finder()
    run_hilbert_vs_find_path()
