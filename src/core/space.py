"""
3D/2D Spatial Geometry, Vector Utilities, and Coordinate Transforms.
Implements transformation matrices as formulated in ACACT (Paper 1, Eq 17).
"""
import numpy as np
from typing import Union, Tuple, Sequence

def normalize_vector(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Returns unit vector in the direction of v. If norm is near zero, returns zero vector."""
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v, dtype=float)
    return v / norm

def local_to_global_transform(pos_local: np.ndarray, theta: float, translation: np.ndarray) -> np.ndarray:
    """
    Transforms coordinates from Local frame to Global frame (ENU).
    Inverse of Paper 1 Eq (17).
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0]
    ])
    return R @ pos_local + translation

def global_to_local_transform(pos_global: np.ndarray, theta: float, translation: np.ndarray) -> np.ndarray:
    """
    Paper 1 Eq (17):
    [xl, yl, zl, 1]^T = [R, T; 0, 1] * [xg, yg, zg, 1]^T
    Transforms coordinates from Global frame to Local frame.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,   1.0]
    ])
    # Relative position before rotation
    diff = pos_global - translation
    return R.T @ diff

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Computes Euclidean distance between two points."""
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))

def clamp_magnitude(v: np.ndarray, max_val: float) -> np.ndarray:
    """Clamps the magnitude of a vector to max_val."""
    norm = np.linalg.norm(v)
    if norm > max_val and norm > 1e-9:
        return (v / norm) * max_val
    return v
