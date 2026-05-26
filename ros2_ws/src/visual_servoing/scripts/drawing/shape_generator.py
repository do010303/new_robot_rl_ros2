#!/usr/bin/env python3
"""
Shape Generator for Drawing Tasks (Board-Local Coordinates)

Generates waypoint sequences in BOARD-LOCAL 2D coordinates:
  - X = left/right on board surface
  - Y = up/down on board surface  
  - Z = 0 (on the board plane)
  - 4th coord = 1.0 (homogeneous for 4×4 transform)

Board center = origin (0, 0, 0).
Shapes are transformed to base_link via BoardTransform pipeline.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class Shape:
    """A drawable shape defined by waypoints in board-local coordinates."""
    name: str
    waypoints: np.ndarray  # (N, 4) array of [x, y, 0, 1] in board-local frame
    closed: bool = True
    
    @property
    def num_waypoints(self) -> int:
        return len(self.waypoints)
    
    def get_waypoint(self, index: int) -> np.ndarray:
        """Get waypoint at index (wraps around if closed)."""
        if self.closed:
            return self.waypoints[index % len(self.waypoints)]
        else:
            return self.waypoints[min(index, len(self.waypoints) - 1)]


class ShapeGenerator:
    """
    Generates drawable shapes in board-local 2D coordinates.
    
    All shapes are centered at board origin (0, 0).
    X/Y are on the board surface, Z=0.
    Coordinates are in METERS.
    """
    
    def __init__(self, safe_zone_m: float = 0.035, vertical_shift_m: float = -0.03):
        """
        Args:
            safe_zone_m: Half-width of safe drawing zone in meters.
                         Default 3.5cm = half of 7cm safe zone.
                         Shapes are scaled to fit within this radius.
            vertical_shift_m: Downward shift on the board plane in meters (default -3cm).
                              Positions the drawing in the robot's sweet spot,
                              avoiding maximum reach boundaries at high corners.
        """
        self.safe_zone_m = safe_zone_m
        self.vertical_shift_m = vertical_shift_m
    
    def _to_board_points(self, xy_points: List[Tuple[float, float]], z_offset: float = -0.005) -> np.ndarray:
        """Convert list of (x, y) tuples to board-local homogeneous coords."""
        points = []
        for x, y in xy_points:
            # Negative Z offset pulls points TOWARDS the camera (Out of the board)
            # in the OpenCV/ROS-Optical frame (Z is depth into scene).
            # -5mm keeps waypoints visible and safe from clipping.
            # Apply vertical shift (negative Y in board frame translates to lower Z in base_link)
            y_shifted = y + self.vertical_shift_m
            points.append([x, y_shifted, z_offset, 1.0])
        return np.array(points, dtype=np.float64)

    @staticmethod
    def _chaikin_closed(points: np.ndarray, iterations: int = 2) -> np.ndarray:
        """Smooth a closed polygon using Chaikin corner cutting."""
        pts = np.array(points, dtype=np.float64)
        for _ in range(iterations):
            refined = []
            for i in range(len(pts)):
                p0 = pts[i]
                p1 = pts[(i + 1) % len(pts)]
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                refined.extend([q, r])
            pts = np.array(refined, dtype=np.float64)
        return pts

    @staticmethod
    def _resample_closed_polyline(points: np.ndarray, n_points: int) -> np.ndarray:
        """Resample a closed polyline to a fixed number of points, including closure."""
        pts = np.array(points, dtype=np.float64)
        if len(pts) < 2:
            return pts

        seg_end = np.vstack([pts[1:], pts[:1]])
        seg_lengths = np.linalg.norm(seg_end - pts, axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total_length = cumulative[-1]
        if total_length <= 1e-9:
            return np.repeat(pts[:1], n_points, axis=0)

        sample_distances = np.linspace(0.0, total_length, max(n_points - 1, 1), endpoint=False)
        resampled = []
        for dist in sample_distances:
            seg_idx = np.searchsorted(cumulative[1:], dist, side='right')
            seg_start_d = cumulative[seg_idx]
            seg_len = seg_lengths[seg_idx]
            t = 0.0 if seg_len <= 1e-9 else (dist - seg_start_d) / seg_len
            p0 = pts[seg_idx]
            p1 = seg_end[seg_idx]
            resampled.append((1.0 - t) * p0 + t * p1)

        resampled.append(resampled[0].copy())
        return np.array(resampled, dtype=np.float64)

    def _triangle_corners(self, size: float, center: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Return inverted triangle corners without closure."""
        cx, cy = center
        height = size * np.sqrt(3) / 2
        p1 = (cx,          cy - 2 * height / 3)  # Bottom apex
        p2 = (cx - size/2, cy + height / 3)      # Top-left
        p3 = (cx + size/2, cy + height / 3)      # Top-right
        return [p1, p2, p3]
    
    def equilateral_triangle(self, 
                             size: float = None,
                             center: Tuple[float, float] = (0.0, 0.0),
                             points_per_edge: int = 1) -> Shape:
        """
        Generate INVERTED equilateral triangle (apex at bottom) in board-local coords.
        
        Args:
            size: Side length in meters (default: 2 * safe_zone_m)
            center: (x, y) center on board surface (default: origin)
            points_per_edge: Points per edge (1=corners only)
            
        Returns:
            Shape with waypoints in board-local [x, y, 0, 1] format
        """
        size = size or (self.safe_zone_m * 2)
        p1, p2, p3 = self._triangle_corners(size, center)
        corners = [p1, p2, p3, p1]  # Close the triangle
        
        if points_per_edge == 1:
            return Shape(
                name=f"triangle_4wp",
                waypoints=self._to_board_points(corners),
                closed=True
            )
        
        # Interpolate points along edges
        points = []
        for i in range(3):
            start = np.array(corners[i])
            end = np.array(corners[i + 1])
            for t in np.linspace(0, 1, points_per_edge, endpoint=False):
                pt = start + t * (end - start)
                points.append(tuple(pt))
        points.append(p1)  # Return to start
        
        return Shape(
            name=f"triangle_{len(points)}wp",
            waypoints=self._to_board_points(points),
            closed=True
        )

    def rounded_triangle(self,
                         size: float = None,
                         center: Tuple[float, float] = (0.0, 0.0),
                         points_per_edge: int = 7,
                         smoothing_iterations: int = 2) -> Shape:
        """
        Generate a rounded inverted triangle on the 2D board.

        The base triangle corners are smoothed with Chaikin corner cutting, then
        resampled to keep waypoint count comparable with the sharp triangle.
        """
        size = size or (self.safe_zone_m * 2)
        corners = np.array(self._triangle_corners(size, center), dtype=np.float64)
        smooth = self._chaikin_closed(corners, iterations=smoothing_iterations)
        target_points = max(points_per_edge * 3 + 1, 10)
        points = self._resample_closed_polyline(smooth, target_points)
        return Shape(
            name=f"rounded_triangle_{len(points)}wp",
            waypoints=self._to_board_points([tuple(pt) for pt in points]),
            closed=True,
        )
    
    def dense_triangle(self, size: float = None,
                       center: Tuple[float, float] = (0.0, 0.0),
                       points_per_edge: int = 10) -> Shape:
        """Dense inverted triangle with many waypoints per edge."""
        return self.equilateral_triangle(size, center, points_per_edge)
    
    def square(self, size: float = None,
               center: Tuple[float, float] = (0.0, 0.0),
               points_per_edge: int = 1) -> Shape:
        """Generate square in board-local coords."""
        size = size or (self.safe_zone_m * 2)
        cx, cy = center
        half = size / 2
        
        corners = [
            (cx - half, cy - half),  # Bottom-left
            (cx + half, cy - half),  # Bottom-right
            (cx + half, cy + half),  # Top-right
            (cx - half, cy + half),  # Top-left
            (cx - half, cy - half),  # Close
        ]
        
        if points_per_edge == 1:
            return Shape(name="square_5wp", waypoints=self._to_board_points(corners), closed=True)
            
        # Interpolate points along edges
        points = []
        for i in range(4):
            start = np.array(corners[i])
            end = np.array(corners[i + 1])
            for t in np.linspace(0, 1, points_per_edge, endpoint=False):
                pt = start + t * (end - start)
                points.append(tuple(pt))
        points.append(corners[0])  # Return to start
        
        return Shape(
            name=f"square_{len(points)}wp",
            waypoints=self._to_board_points(points),
            closed=True
        )
    
    def polygon(self, n_sides: int, scale: float = 1.0) -> Shape:
        """Generate regular polygon centered at origin."""
        radius = self.safe_zone_m * scale
        offset_angle = -math.pi / 2  # Start from top
        
        points = []
        for i in range(n_sides + 1):
            theta = offset_angle + (2 * math.pi * i / n_sides)
            x = radius * math.cos(theta)
            y = radius * math.sin(theta)
            points.append((x, y))
        
        return Shape(
            name=f"polygon_{n_sides}",
            waypoints=self._to_board_points(points),
            closed=True
        )
    
    def line(self, length: float = None, angle_deg: float = 0.0) -> Shape:
        """Generate a line through the center."""
        length = length or (self.safe_zone_m * 2)
        rad = math.radians(angle_deg)
        half = length / 2
        
        points = [
            (-half * math.cos(rad), -half * math.sin(rad)),
            ( half * math.cos(rad),  half * math.sin(rad)),
        ]
        return Shape(name="line", waypoints=self._to_board_points(points), closed=False)

    def circle(self,
               diameter: float = None,
               center: Tuple[float, float] = (0.0, 0.0),
               n_points: int = 22) -> Shape:
        """Generate a circle on the board for smooth-path benchmarking."""
        diameter = diameter or (self.safe_zone_m * 2)
        radius = diameter / 2.0
        cx, cy = center
        points = []
        for theta in np.linspace(-math.pi / 2, 3 * math.pi / 2, max(n_points - 1, 8), endpoint=False):
            points.append((cx + radius * math.cos(theta), cy + radius * math.sin(theta)))
        points.append(points[0])
        return Shape(name=f"circle_{len(points)}wp", waypoints=self._to_board_points(points), closed=True)
    
    def random_triangle(self, min_size: float = 0.03, max_size: float = None) -> Shape:
        """Generate random triangle within safe zone."""
        max_size = max_size or (self.safe_zone_m * 1.5)
        size = np.random.uniform(min_size, max_size)
        height = size * np.sqrt(3) / 2
        
        margin_x = size / 2 + 0.005
        margin_y = height / 2 + 0.005
        
        max_offset = self.safe_zone_m - max(margin_x, margin_y)
        max_offset = max(0.001, max_offset)
        
        cx = np.random.uniform(-max_offset, max_offset)
        cy = np.random.uniform(-max_offset, max_offset)
        
        return self.equilateral_triangle(size=size, center=(cx, cy))


def test_shape_generator():
    """Test shape generation in board-local coordinates."""
    print("=" * 60)
    print("Testing Board-Local Shape Generator")
    print("=" * 60)
    
    gen = ShapeGenerator(safe_zone_m=0.035)
    
    # Test triangle
    tri = gen.equilateral_triangle(size=0.06)
    print(f"\n{tri.name}:")
    print(f"  Waypoints: {tri.num_waypoints}")
    for i, wp in enumerate(tri.waypoints):
        print(f"  P{i}: ({wp[0]:.4f}, {wp[1]:.4f}, {wp[2]:.4f}, {wp[3]:.1f})")
    
    # Verify: all Z use the configured board offset, all W=1
    assert np.allclose(tri.waypoints[:, 2], -0.005), "Z should use board offset"
    assert np.allclose(tri.waypoints[:, 3], 1.0), "W should be 1"
    
    # Test square
    sq = gen.square(size=0.05)
    print(f"\n{sq.name}:")
    print(f"  Waypoints: {sq.num_waypoints}")
    
    # Test dense triangle
    dense = gen.dense_triangle(size=0.06, points_per_edge=7)
    print(f"\n{dense.name}:")
    print(f"  Waypoints: {dense.num_waypoints}")

    rounded = gen.rounded_triangle(size=0.06, points_per_edge=7)
    print(f"\n{rounded.name}:")
    print(f"  Waypoints: {rounded.num_waypoints}")

    circle = gen.circle(diameter=0.06, n_points=22)
    print(f"\n{circle.name}:")
    print(f"  Waypoints: {circle.num_waypoints}")
    
    print("\n" + "=" * 60)
    print("Board-local shape generator OK!")


if __name__ == '__main__':
    test_shape_generator()
