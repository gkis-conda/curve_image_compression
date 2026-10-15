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

    def render_trajectory_to_svg(self, path_list, filename="custom_snake_path.svg"):
        """
        Builds the gray grid backdrop, draws the custom path line through 
        the centers, and saves the standalone compliant XML dataset.
        """
        svg_lines = [
            '<svg width="{}" height="{}" viewBox="0 0 {} {}" xmlns="http://www.w3.org/2000/svg" style="background:#888888;">'.format(
                self.w_canvas, self.h_canvas, self.w_canvas, self.h_canvas
            )
        ]

        # 1. Structural Layer: Render gray grid lines bounding the cells
        # Vertical grid line splitters
        for c in range(self.m_cols + 1):
            x = c * self.scale + self.pad
            y1 = self.pad
            y2 = self.n_rows * self.scale + self.pad
            svg_lines.append(
                '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#333344" stroke-width="1"/>'.format(x, y1, x, y2)
            )
            
        # Horizontal grid line splitters
        for r in range(self.n_rows + 1):
            y = r * self.scale + self.pad
            x1 = self.pad
            x2 = self.m_cols * self.scale + self.pad
            svg_lines.append(
                '<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="#333344" stroke-width="1"/>'.format(x1, y, x2, y)
            )

        # 2. Path Layer: Render the sequential trajectory trail polyline
        formatted_centers = []
        for r, c in path_list:
            cx, cy = self._get_cell_center_pixel(r, c)
            formatted_centers.append("{},{}".format(cx, cy))
            
        polyline_pts_str = " ".join(formatted_centers)
        svg_lines.append(
            '<polyline points="{}" fill="none" stroke="#00ff66" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>'.format(
                polyline_pts_str
            )
        )

        # 3. Node Center Indicators: Overlay dots on every visited cell center
        for r, c in path_list:
            cx, cy = self._get_cell_center_pixel(r, c)
            svg_lines.append(
                '<circle cx="{}" cy="{}" r="3" fill="#00aa44"/>'.format(cx, cy)
            )

        # 4. Critical Waypoints: Highlighting explicit Start and End anchors
        if path_list:
            sx, sy = self._get_cell_center_pixel(path_list[0][0], path_list[0][1])
            ex, ey = self._get_cell_center_pixel(path_list[-1][0], path_list[-1][1])
            
            # Start Node (Glowing Yellow)
            svg_lines.append('<circle cx="{}" cy="{}" r="6" fill="#ffff00" stroke="#fff" stroke-width="1"/>'.format(sx, sy))
            # End Node (Glowing Red)
            svg_lines.append('<circle cx="{}" cy="{}" r="6" fill="#ff3333" stroke="#fff" stroke-width="1"/>'.format(ex, ey))

        svg_lines.append('</svg>')
        
        # Write output to disk
        with open(filename, "w") as f:
            f.write("\n".join(svg_lines))
            
        print("[SUCCESS] Grid map and custom path rendered to: {}".format(filename))


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
