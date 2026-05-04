# Copyright (c) 2026 Teerth Sharma. All rights reserved.

"""
topoflow.topology — Persistent homology computation via ripser.
"""

import numpy as np
from ripser import ripser
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress ripser verbose output
warnings.filterwarnings("ignore")


def generate_random_point_cloud(
    n_points: int = 50,
    n_dims: int = 2,
    seed: Optional[int] = None,
    distribution: str = "uniform",
) -> np.ndarray:
    """
    Generate a random point cloud for PH computation.

    Parameters
    ----------
    n_points : int
        Number of points in the cloud.
    n_dims : int
        Dimensionality of the embedding space.
    seed : int, optional
        Random seed for reproducibility.
    distribution : str
        'uniform', 'circle', 'sphere', or 'torus'.

    Returns
    -------
    np.ndarray
        (n_points, n_dims) array of points.
    """
    if seed is not None:
        np.random.seed(seed)

    if distribution == "uniform":
        return np.random.rand(n_points, n_dims)

    elif distribution == "circle":
        angles = np.random.uniform(0, 2 * np.pi, n_points)
        r = 0.5 + 0.3 * np.random.randn(n_points)
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        return np.column_stack([x, y])

    elif distribution == "sphere":
        # Uniform sampling on sphere surface
        u = np.random.rand(n_points)
        v = np.random.rand(n_points)
        theta = 2 * np.pi * u
        phi = np.arccos(2 * v - 1)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.column_stack([x, y, z])

    elif distribution == "torus":
        u = np.random.rand(n_points) * 2 * np.pi
        v = np.random.rand(n_points) * 2 * np.pi
        R, r = 2.0, 0.8
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        return np.column_stack([x, y, z])

    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def read_point_cloud(filepath: str) -> np.ndarray:
    """
    Read point cloud from CSV or TXT file.

    Parameters
    ----------
    filepath : str
        Path to file with point coordinates.

    Returns
    -------
    np.ndarray
        (n_points, n_dims) array.
    """
    try:
        data = np.loadtxt(filepath, delimiter=",")
    except ValueError:
        try:
            data = np.loadtxt(filepath, delimiter=" ")
        except ValueError:
            data = np.genfromtxt(filepath, delimiter=",")

    return data


def compute_persistent_homology(
    points: np.ndarray,
    max_dim: int = 1,
    thresh: float = 2.0,
    n_perm: Optional[int] = None,
) -> Dict:
    """
    Compute Vietoris-Rips persistent homology using ripser.

    Parameters
    ----------
    points : np.ndarray
        Point cloud data (n_points, n_dims).
    max_dim : int
        Maximum homology dimension to compute (0, 1, 2).
    thresh : float
        Maximum edge length for Rips filtration.
    n_perm : int, optional
        Number of points to subsample via greedy permutation.

    Returns
    -------
    dict
        Dictionary with 'dgms' (list of persistence diagrams),
        'num_points', and 'point_cloud'.
    """
    result = ripser(
        points,
        maxdim=max_dim,
        thresh=thresh,
        n_perm=n_perm,
        do_cocycles=False,
    )

    return {
        "dgms": result["dgms"],
        "num_points": len(points),
        "point_cloud": points,
        "num_edges": result.get("num_edges", None),
        "epoch": result.get("epoch", None),
    }


def extract_barcodes(
    dgms: List[np.ndarray], max_cardinality: int = 50
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Extract birth-death intervals (barcodes) from persistence diagrams.

    Parameters
    ----------
    dgms : list of np.ndarray
        List of persistence diagrams from ripser.
    max_cardinality : int
        Maximum number of intervals to keep per dimension.

    Returns
    -------
    dict
        Dictionary with 'betti_0', 'betti_1', etc. as lists of (birth, death) tuples.
    """
    barcodes = {}
    dim_names = ["betti_0", "betti_1", "betti_2", "betti_3"]

    for dim, dgm in enumerate(dgms):
        if dim > 3:
            continue
        # Filter out infinite death (never dies)
        finite_pairs = dgm[np.isfinite(dgm[:, 1])]
        # Also filter pairs where birth == death (noise)
        lifetimes = finite_pairs[:, 1] - finite_pairs[:, 0]
        # Sort by lifetime descending
        idx = np.argsort(lifetimes)[::-1][:max_cardinality]
        sorted_pairs = finite_pairs[idx]

        barcodes[dim_names[dim]] = [
            (float(b), float(d)) for b, d in sorted_pairs
        ]

    return barcodes


def compute_betti_curves(
    barcodes: Dict[str, List[Tuple[float, float]]],
    n_steps: int = 100,
    t_max: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute Betti curves (Betti number as function of threshold t).

    Parameters
    ----------
    barcodes : dict
        Barcodes from extract_barcodes.
    n_steps : int
        Number of sampling points for the curve.
    t_max : float, optional
        Maximum threshold value. Defaults to max death value.

    Returns
    -------
    dict
        Dictionary with 't' (threshold values) and 'betti_0', 'betti_1', etc.
    """
    betti_curves = {}

    # Find global t_max
    all_deaths = []
    for key, pairs in barcodes.items():
        for birth, death in pairs:
            all_deaths.append(death)
    global_t_max = t_max if t_max is not None else max(all_deaths) * 1.1
    t_values = np.linspace(0, global_t_max, n_steps)

    betti_curves["t"] = t_values

    for dim in range(4):
        key = f"betti_{dim}"
        if key not in barcodes:
            continue

        pairs = barcodes[key]
        betti_curve = np.zeros(n_steps)

        for birth, death in pairs:
            for i, t in enumerate(t_values):
                if birth <= t < death:
                    betti_curve[i] += 1

        betti_curves[key] = betti_curve

    return betti_curves


def compute_persistence_landscape(
    barcodes: Dict[str, List[Tuple[float, float]]],
    n_functions: int = 5,
    n_samples: int = 100,
    t_max: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute persistence landscape as a 2D array (landscape functions × samples).

    Parameters
    ----------
    barcodes : dict
        Barcodes from extract_barcodes.
    n_functions : int
        Number of landscape functions to compute.
    n_samples : int
        Number of sample points.
    t_max : float, optional
        Maximum threshold.

    Returns
    -------
    dict
        Dictionary with 't', 'landscape_0', 'landscape_1', etc.
    """
    landscapes = {}

    # Find global t_max
    all_deaths = []
    for key, pairs in barcodes.items():
        for birth, death in pairs:
            all_deaths.append(death)
    global_t_max = t_max if t_max is not None else max(all_deaths) * 1.1
    t_values = np.linspace(0, global_t_max, n_samples)

    landscapes["t"] = t_values

    for dim in range(4):
        key = f"betti_{dim}"
        if key not in barcodes:
            continue

        pairs = barcodes[key]
        landscape_matrix = np.zeros((n_functions, n_samples))

        for birth, death in pairs:
            # Persistence = death - birth
            persistence = death - birth
            if persistence <= 0:
                continue

            # Birth-death interval
            for k in range(n_functions):
                # Lambda function for this landscape basis
                midpoint = (birth + death) / 2
                half_life = persistence / 2

                # Compute landscape value at each t
                for i, t in enumerate(t_values):
                    if birth <= t <= death:
                        # Triangular function
                        dist_from_center = abs(t - midpoint)
                        val = max(0, half_life - dist_from_center)
                        landscape_matrix[k, i] = max(landscape_matrix[k, i], val)

        landscapes[f"landscape_{dim}"] = landscape_matrix

    return landscapes


def get_topological_summary(dgms: List[np.ndarray]) -> Dict:
    """
    Compute summary statistics from persistence diagrams.

    Returns
    -------
    dict
        Statistics including total persistence, max persistence, etc.
    """
    summary = {}

    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            summary[f"betti_{dim}"] = 0
            summary[f"max_persistence_{dim}"] = 0.0
            summary[f"total_persistence_{dim}"] = 0.0
            continue

        finite = dgm[np.isfinite(dgm[:, 1])]
        lifetimes = finite[:, 1] - finite[:, 0]

        summary[f"betti_{dim}"] = len(finite)
        summary[f"max_persistence_{dim}"] = float(np.max(lifetimes)) if len(lifetimes) > 0 else 0.0
        summary[f"total_persistence_{dim}"] = float(np.sum(lifetimes)) if len(lifetimes) > 0 else 0.0

    return summary
