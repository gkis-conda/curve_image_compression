import os

class SimpleGridPathVisualizer:
    """
    Renders a backdrop grid of size N x M and connects the centers of the 
    specified cells sequentially according to a custom input coordinate list.
    Strictly compliant with Python 3.6 ASCII standards.
    """
    def __init__(self, n_rows, m_cols, scale_factor=40.0, padding=40.0):
        self.n_rows = n_rows
        self.m_cols = m_cols
        self.scale = scale_factor
        self.pad = padding
        
        # Calculate canvas limits based on bounding box steps
        self.w_canvas = m_cols * self.scale + 2.0 * self.pad
        self.h_canvas = n_rows * self.scale + 2.0 * self.pad

    def _get_cell_center_pixel(self, r, c):
        """
        Calculates the absolute pixel coordinates for the geometric center 
        of a given grid cell index (r, c).
        Rows (r) map to Y-axis, Columns (c) map to X-axis.
        """
        x = c * self.scale + self.scale / 2.0 + self.pad
        y = r * self.scale + self.scale / 2.0 + self.pad
        return x, y

    # Centered style configuration block (Can be moved to class level or config files)
    # Простой и понятный блок конфигурации презентации
    STYLE_CONFIG = {
        "grid_line_stroke": "#333344",
        "grid_line_width": "1",
        "path_line_stroke": "#00ff66",
        "path_line_width": "3",
        "node_dot_fill": "#00aa44",
        "start_node_fill": "#ffff00",
        "end_node_fill": "#ff3333",
        "background": '#888888'
    }

    def render_trajectory_to_svg(self, path_list, filename="custom_snake_path.svg", styles=None):
        """
        BALANCED HIGH-SPEED SVG GENERATOR:
        Leverages parent <g> grouping tags for colors (stroke/fill) to avoid repetition,
        while maintaining standard geometric 'r' parameters explicitly inside the circle tags.
        """
        if not path_list:
            print("[WARNING] Path list is empty. SVG rendering skipped.")
            return

        cfg = styles if styles is not None else self.STYLE_CONFIG

        # --- 1. SINGLE-PASS COORDINATE PRE-COMPUTATION ---
        computed_centers = [self._get_cell_center_pixel(r, c) for r, c in path_list]

        # --- 2. HEADER AND BACKDROP INITIALIZATION ---
        svg_lines = [
            '<svg width="{}" height="{}" viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg" style="background:{};">'.format(
                self.w_canvas, self.h_canvas, self.w_canvas, self.h_canvas, cfg["background"]
            )
        ]

        # --- 3. GRID BACKDROP LAYER (Parent-Group Inheritance for strokes) ---
        svg_lines.append('<g stroke="{}" stroke-width="{}">'.format(cfg["grid_line_stroke"], cfg["grid_line_width"]))

        y_min, y_max = self.pad, self.n_rows * self.scale + self.pad
        for c in range(self.m_cols + 1):
            x = int(c * self.scale + self.pad)
            svg_lines.append('  <line x1="{}" y1="{}" x2="{}" y2="{}"/>'.format(x, y_min, x, y_max))

        x_min, x_max = self.pad, self.m_cols * self.scale + self.pad
        for r in range(self.n_rows + 1):
            y = int(r * self.scale + self.pad)
            svg_lines.append('  <line x1="{}" y1="{}" x2="{}" y2="{}"/>'.format(x_min, y, x_max, y))

        svg_lines.append('</g>')

        # --- 4. TRAJECTORY PATH LAYER ---
        polyline_pts_str = " ".join("{},{}".format(int(cx), int(cy)) for cx, cy in computed_centers)
        svg_lines.append(
            '<polyline points="{}" fill="none" stroke="{}" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"/>'.format(
                polyline_pts_str, cfg["path_line_stroke"], cfg["path_line_width"]
            )
        )

        # --- 5. NODE INDICATORS LAYER (Group fill color inheritance + Explicit radius) ---
        #svg_lines.append('<g fill="{}">'.format(cfg["node_dot_fill"]))
        #svg_lines.extend(
        #    '  <circle cx="{}" cy="{}" r="3"/>'.format(cx, cy)
        #    for cx, cy in computed_centers
        #)
        #svg_lines.append('</g>')

        # --- 6. CRITICAL WAYPOINTS (START / END CORNER ANCHORS) ---
        sx, sy = computed_centers[0]
        svg_lines.append('<circle cx="{}" cy="{}" r="6" fill="{}" stroke="#ffffff" stroke-width="1"/>'.
                         format(sx, sy,cfg["start_node_fill"]))
        ex, ey = computed_centers[-1]
        svg_lines.append('<circle cx="{}" cy="{}" r="6" fill="{}" stroke="#ffffff" stroke-width="1"/>'.
                         format(ex, ey, cfg["end_node_fill"]))

        svg_lines.append('</svg>')

        # --- 7. DISK SERIALIZATION PASS ---
        with open(filename, "w") as f:
            f.write("\n".join(svg_lines))

        print("[SUCCESS] Stable grid map and custom trajectory path rendered to: {}".format(filename))


# =============================================================================
#                                 TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    # Define an arbitrary grid size: N rows by M columns
    N_rows = 4
    M_cols = 5

    # Input format: a literal manual sequence of (row, col) coordinates
    # Simulating a simple custom test snake trace
    test_snake = [
        (0, 0), (0, 1), (1, 1), (1, 0),
        (2, 0), (2, 1), (2, 2), (1, 2),
        (0, 2), (0, 3), (1, 3), (2, 3),
        (3, 3), (3, 2), (3, 1), (3, 0)
    ]

    # Initialize visualizer tool (Grid: N_rows x M_cols)
    visualizer = SimpleGridPathVisualizer(n_rows=N_rows, m_cols=M_cols, scale_factor=40.0)
    
    # Export drawing
    visualizer.render_trajectory_to_svg(test_snake, filename="custom_snake_path.svg")
