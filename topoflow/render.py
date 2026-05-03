"""
topoflow.render — 3D visualization of persistent homology.
Renders Betti barcodes as 3D bar charts, persistence landscapes as 3D surfaces,
VR complexes as 3D meshes, and produces animated GIFs.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from PIL import Image
import io
import warnings
from typing import Dict, List, Optional, Tuple, Callable

# Suppress specific warnings
warnings.filterwarnings("ignore", ".*GUI is implemented.*")
warnings.filterwarnings("ignore", ".*animation.*")


# Beautiful custom colormaps
TORUS_CMAP = LinearSegmentedColormap.from_list(
    "torus", ["#0a0a1a", "#1a1a4a", "#4a1a6a", "#6a2a9a", "#9a4aca", "#ca6af0"]
)

ELECTRIC_CMAP = LinearSegmentedColormap.from_list(
    "electric", ["#000000", "#1a0033", "#4d0099", "#8000cc", "#b347ff", "#ff80ff"]
)

SUNSET_CMAP = LinearSegmentedColormap.from_list(
    "sunset", ["#0d0221", "#3d1a78", "#9b2fc0", "#f72585", "#ff9e00", "#fee440"]
)

NEON_CMAP = LinearSegmentedColormap.from_list(
    "neon", ["#000000", "#1a0a2e", "#2d1b69", "#5c2d91", "#8b5cf6", "#c084fc"]
)


def create_3d_barcode_chart(
    barcodes: Dict[str, List[Tuple[float, float]]],
    title: str = "Betti Barcodes as 3D Bars",
    birth_label: str = "Birth",
    persistence_label: str = "Persistence",
    color_scheme: str = "auto",
    elevation: float = 25,
    azimuth: float = 45,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Render Betti_0 and Betti_1 persistence as 3D bar chart.

    Parameters
    ----------
    barcodes : dict
        Dictionary with 'betti_0', 'betti_1' etc. as lists of (birth, death) tuples.
    title : str
        Plot title.
    birth_label : str
        Label for birth axis.
    persistence_label : str
        Label for persistence (death-birth) axis.
    color_scheme : str
        'auto', 'betti0', 'betti1', 'lifetime', 'depth'.
    elevation : float
        Camera elevation angle.
    azimuth : float
        Camera azimuth angle.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure object.
    """
    fig = plt.figure(figsize=figsize, facecolor="#0a0a1a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0a0a1a")

    colors_betti0 = ["#00ffff", "#00cccc", "#009999", "#006666"]
    colors_betti1 = ["#ff00ff", "#cc00cc", "#990099", "#660066"]

    bar_index = 0

    for dim in [0, 1]:
        key = f"betti_{dim}"
        if key not in barcodes:
            continue

        pairs = barcodes[key]
        if len(pairs) == 0:
            continue

        # Sort by persistence (lifetime)
        sorted_pairs = sorted(pairs, key=lambda x: x[1] - x[0], reverse=True)
        n_bars = min(len(sorted_pairs), 20)  # Limit for visual clarity
        sorted_pairs = sorted_pairs[:n_bars]

        for i, (birth, death) in enumerate(sorted_pairs):
            persistence = death - birth

            if color_scheme == "lifetime":
                color = plt.cm.plasma(np.clip(persistence / max(p[1] - p[0] for p in pairs), 0, 1))
            elif color_scheme == "depth":
                color = plt.cm.viridis(i / n_bars)
            elif dim == 0:
                color = colors_betti0[i % len(colors_betti0)]
            else:
                color = colors_betti1[i % len(colors_betti1)]

            if color_scheme in ["lifetime", "depth"]:
                # Bar as 3D rectangle
                x = birth
                y = bar_index
                z = 0

                ax.bar3d(
                    x,
                    y,
                    z,
                    dx=persistence,
                    dy=0.4,
                    dz=persistence * 0.5 + 0.01,
                    color=color,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.3,
                )
            else:
                x = bar_index
                y = birth
                z = 0

                ax.bar3d(
                    x,
                    y,
                    z,
                    dx=0.4,
                    dy=persistence,
                    dz=persistence * 0.5 + 0.01,
                    color=color,
                    alpha=0.85,
                    edgecolor="white",
                    linewidth=0.3,
                )

            bar_index += 0.6

        bar_index += 2  # Gap between dimensions

    ax.set_xlabel(birth_label, color="white", fontsize=11, labelpad=10)
    ax.set_ylabel("Feature Index", color="white", fontsize=11, labelpad=10)
    ax.set_zlabel(persistence_label, color="white", fontsize=11, labelpad=10)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=20)

    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333333")
    ax.yaxis.pane.set_edgecolor("#333333")
    ax.zaxis.pane.set_edgecolor("#333333")
    ax.grid(True, color="#222222", alpha=0.5)

    ax.view_init(elev=elevation, azim=azimuth)

    fig.tight_layout()
    return fig


def create_3d_persistence_surface(
    landscapes: Dict[str, np.ndarray],
    title: str = "Persistence Landscape as 3D Surface",
    cmap_name: str = "torus",
    elevation: float = 30,
    azimuth: float = 135,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """
    Render persistence landscape as 3D surface.

    Parameters
    ----------
    landscapes : dict
        Dictionary from compute_persistence_landscape with 't' and 'landscape_N' arrays.
    title : str
        Plot title.
    cmap_name : str
        Colormap name ('torus', 'electric', 'sunset', 'neon').
    elevation : float
        Camera elevation.
    azimuth : float
        Camera azimuth.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig = plt.figure(figsize=figsize, facecolor="#0a0a1a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0a0a1a")

    cmap = {"torus": TORUS_CMAP, "electric": ELECTRIC_CMAP, "sunset": SUNSET_CMAP, "neon": NEON_CMAP}.get(
        cmap_name, TORUS_CMAP
    )

    # Use betti_1 landscape (loops)
    key = "landscape_1"
    if key not in landscapes:
        key = list(landscapes.keys())[1] if len(landscapes) > 1 else None

    if key is None:
        ax.text(0, 0, 0, "No landscape data", color="white", fontsize=16)
        return fig

    t = landscapes["t"]
    landscape = landscapes[key]
    n_functions = landscape.shape[0]

    # Create mesh for surface plot
    T, K = np.meshgrid(t, np.arange(n_functions))

    # Sum all landscape functions for surface
    Z = np.sum(landscape, axis=0)

    # Plot surface
    surf = ax.plot_surface(
        T,
        K,
        Z,
        cmap=cmap,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )

    # Plot individual landscape curves behind surface
    for k in range(n_functions):
        ax.plot(
            t,
            [k] * len(t),
            landscape[k],
            color=cmap(k / n_functions),
            linewidth=2,
            alpha=0.7,
        )

    ax.set_xlabel("Threshold t", color="white", fontsize=11, labelpad=10)
    ax.set_ylabel("Landscape Function k", color="white", fontsize=11, labelpad=10)
    ax.set_zlabel("Persistence", color="white", fontsize=11, labelpad=10)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=20)

    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333333")
    ax.yaxis.pane.set_edgecolor("#333333")
    ax.zaxis.pane.set_edgecolor("#333333")

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1, label="Persistence")

    ax.view_init(elev=elevation, azim=azimuth)

    fig.tight_layout()
    return fig


def create_vr_complex_visualization(
    points: np.ndarray,
    edges: List[Tuple[int, int]] = None,
    simplices: List[List[int]] = None,
    point_cloud_only: bool = False,
    title: str = "Vietoris-Rips Complex",
    elevation: float = 25,
    azimuth: float = 45,
    figsize: Tuple[int, int] = (12, 10),
) -> plt.Figure:
    """
    Render VR complex as 3D mesh/scatter plot.

    Parameters
    ----------
    points : np.ndarray
        Point cloud coordinates.
    edges : list of tuples
        List of (i, j) edge indices.
    simplices : list
        List of simplex vertex indices.
    point_cloud_only : bool
        If True, only show scatter plot.
    title : str
        Plot title.
    elevation : float
        Camera elevation.
    azimuth : float
        Camera azimuth.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig = plt.figure(figsize=figsize, facecolor="#0a0a1a")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0a0a1a")

    n_points = len(points)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, n_points))

    # Scatter plot
    if points.shape[1] == 2:
        x = points[:, 0]
        y = points[:, 1]
        z = np.zeros(n_points)
        scatter = ax.scatter(x, y, z, c=range(n_points), cmap=TORUS_CMAP, s=80, depthshade=True)
    else:
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        scatter = ax.scatter(x, y, z, c=range(n_points), cmap=TORUS_CMAP, s=80, depthshade=True)

    # Add edges
    if edges is not None and not point_cloud_only:
        for i, j in edges:
            if points.shape[1] == 2:
                ax.plot(
                    [points[i, 0], points[j, 0]],
                    [points[i, 1], points[j, 1]],
                    [0, 0],
                    color="#00ffff",
                    alpha=0.4,
                    linewidth=0.8,
                )
            else:
                ax.plot(
                    [points[i, 0], points[j, 0]],
                    [points[i, 1], points[j, 1]],
                    [points[i, 2], points[j, 2]],
                    color="#00ffff",
                    alpha=0.4,
                    linewidth=0.8,
                )

    # Add triangles (2-simplices)
    if simplices is not None and not point_cloud_only:
        for simplex in simplices:
            if len(simplex) == 3:
                verts = points[simplex]
                triangle = Poly3DCollection(
                    [verts],
                    alpha=0.15,
                    facecolor="#ff00ff",
                    edgecolor="#ff66ff",
                    linewidth=0.5,
                )
                ax.add_collection3d(triangle)

    ax.set_xlabel("X", color="white", fontsize=11)
    ax.set_ylabel("Y", color="white", fontsize=11)
    ax.set_zlabel("Z", color="white", fontsize=11)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=15)

    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333333")
    ax.yaxis.pane.set_edgecolor("#333333")
    ax.zaxis.pane.set_edgecolor("#333333")
    ax.grid(True, color="#222222", alpha=0.5)

    ax.view_init(elev=elevation, azim=azimuth)

    fig.tight_layout()
    return fig


def create_betti_curves_plot(
    betti_curves: Dict[str, np.ndarray],
    title: str = "Betti Curves Over Threshold",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """
    Render Betti curves as 2D line plot.

    Parameters
    ----------
    betti_curves : dict
        Dictionary from compute_betti_curves.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0a0a1a")
    ax.set_facecolor("#0a0a1a")

    t = betti_curves["t"]

    colors = ["#00ffff", "#ff00ff", "#ffff00", "#00ff00"]
    labels = ["H₀ (Components)", "H₁ (Loops)", "H₂ (Voids)", "H₃"]

    for dim in range(4):
        key = f"betti_{dim}"
        if key not in betti_curves:
            continue
        ax.plot(
            t,
            betti_curves[key],
            color=colors[dim],
            linewidth=2.5,
            label=labels[dim],
            alpha=0.9,
        )
        ax.fill_between(t, 0, betti_curves[key], color=colors[dim], alpha=0.15)

    ax.set_xlabel("Threshold t", color="white", fontsize=12)
    ax.set_ylabel("Betti Number", color="white", fontsize=12)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_color("#333333")
    ax.spines["right"].set_color("#333333")
    ax.grid(True, color="#222222", alpha=0.5)
    ax.legend(facecolor="#1a1a3a", edgecolor="#333333", labelcolor="white", fontsize=10)

    fig.tight_layout()
    return fig


def create_persistence_diagram(
    dgms: List[np.ndarray],
    title: str = "Persistence Diagram",
    figsize: Tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    Render persistence diagram (birth vs death scatter).

    Parameters
    ----------
    dgms : list of np.ndarray
        Persistence diagrams from ripser.
    title : str
        Plot title.
    figsize : tuple
        Figure size.

    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0a0a1a")
    ax.set_facecolor("#0a0a1a")

    colors = ["#00ffff", "#ff00ff", "#ffff00", "#00ff00"]
    labels = ["H₀", "H₁", "H₂", "H₃"]

    max_val = 0

    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            continue
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite) == 0:
            continue

        birth = finite[:, 0]
        death = finite[:, 1]
        persistence = death - birth

        max_val = max(max_val, np.max(death))

        ax.scatter(
            birth,
            death,
            c=colors[dim],
            s=persistence * 100 + 10,
            alpha=0.7,
            label=f"{labels[dim]} ({len(finite)} features)",
            edgecolors="white",
            linewidth=0.5,
        )

    # Diagonal line (birth == death)
    if max_val > 0:
        ax.plot([0, max_val], [0, max_val], color="#333333", linestyle="--", linewidth=1, alpha=0.7)

    ax.set_xlabel("Birth", color="white", fontsize=12)
    ax.set_ylabel("Death", color="white", fontsize=12)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_color("#333333")
    ax.spines["right"].set_color("#333333")
    ax.grid(True, color="#222222", alpha=0.5)
    ax.legend(facecolor="#1a1a3a", edgecolor="#333333", labelcolor="white", fontsize=10)

    fig.tight_layout()
    return fig


def create_collapse_animation(
    barcodes: Dict[str, List[Tuple[float, float]]],
    betti_curves: Dict[str, np.ndarray],
    n_frames: int = 30,
    output_path: Optional[str] = None,
    title: str = "Topological Feature Collapse",
    figsize: Tuple[int, int] = (14, 6),
) -> Optional[str]:
    """
    Create animated GIF showing topological features "collapsing" as threshold increases.

    Parameters
    ----------
    barcodes : dict
        Barcodes from extract_barcodes.
    betti_curves : dict
        Betti curves from compute_betti_curves.
    n_frames : int
        Number of frames in the animation.
    output_path : str, optional
        Path to save the GIF.
    title : str
        Animation title.
    figsize : tuple
        Figure size per frame.

    Returns
    -------
    str or None
        Path to saved GIF if output_path provided.
    """
    if output_path is None:
        output_path = "/tmp/topoflow_collapse.gif"

    t = betti_curves["t"]
    max_t = np.max(t)

    frames = []

    # Color scheme
    cmap = NEON_CMAP

    for frame_idx in range(n_frames):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor="#0a0a1a")

        current_t = (frame_idx / (n_frames - 1)) * max_t

        # Left: Betti curve with current position
        ax1.set_facecolor("#0a0a1a")
        for dim in range(2):
            key = f"betti_{dim}"
            if key not in betti_curves:
                continue
            color = "#00ffff" if dim == 0 else "#ff00ff"
            ax1.plot(
                t,
                betti_curves[key],
                color=color,
                linewidth=2.5,
                label=f"H_{dim}",
            )
            ax1.fill_between(t, 0, betti_curves[key], color=color, alpha=0.1)

        ax1.axvline(x=current_t, color="white", linewidth=2, linestyle="--", alpha=0.8)
        ax1.set_xlabel("Threshold t", color="white", fontsize=11)
        ax1.set_ylabel("Betti Number", color="white", fontsize=11)
        ax1.set_title(f"t = {current_t:.3f}", color="white", fontsize=12)
        ax1.tick_params(colors="white")
        ax1.spines["bottom"].set_color("#333333")
        ax1.spines["left"].set_color("#333333")
        ax1.spines["top"].set_color("#333333")
        ax1.spines["right"].set_color("#333333")
        ax1.grid(True, color="#222222", alpha=0.3)
        ax1.legend(facecolor="#1a1a3a", edgecolor="#333333", labelcolor="white")

        # Right: 3D bar representation at current threshold
        ax2.set_facecolor("#0a0a1a")
        ax2 = fig.add_subplot(122, projection="3d", facecolor="#0a0a1a")

        colors_betti0 = ["#00ffff", "#00cccc", "#009999", "#006666", "#003333"]
        colors_betti1 = ["#ff00ff", "#cc00cc", "#990099", "#660066", "#330033"]

        bar_idx = 0
        for dim in [0, 1]:
            key = f"betti_{dim}"
            if key not in barcodes:
                continue
            pairs = barcodes[key]

            for birth, death in pairs:
                if birth <= current_t < death:
                    persistence = death - birth
                    x = bar_idx
                    y = birth
                    z = 0
                    dz = persistence * 0.5 + 0.01
                    color = colors_betti0[bar_idx % len(colors_betti0)] if dim == 0 else colors_betti1[bar_idx % len(colors_betti1)]

                    if isinstance(color, str):
                        ax2.bar3d(
                            x,
                            y,
                            z,
                            dx=0.4,
                            dy=persistence,
                            dz=dz,
                            color=color,
                            alpha=0.85,
                            edgecolor="white",
                            linewidth=0.2,
                        )
                    bar_idx += 0.5

        ax2.set_xlabel("Feature", color="white", fontsize=9)
        ax2.set_ylabel("Birth", color="white", fontsize=9)
        ax2.set_zlabel("Persistence", color="white", fontsize=9)
        ax2.tick_params(colors="white", labelsize=7)
        ax2.xaxis.pane.fill = False
        ax2.yaxis.pane.fill = False
        ax2.zaxis.pane.fill = False
        ax2.xaxis.pane.set_edgecolor("#333333")
        ax2.yaxis.pane.set_edgecolor("#333333")
        ax2.zaxis.pane.set_edgecolor("#333333")
        ax2.view_init(elev=25, azim=45)

        fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.98)

        # Save frame to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, facecolor="#0a0a1a", bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)

    # Save as GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
    )

    return output_path


def save_animation(
    fig: plt.Figure,
    output_path: str,
    n_frames: int = 30,
    fps: int = 10,
    func: Optional[Callable] = None,
) -> str:
    """
    Save matplotlib figure as animated GIF with optional rotation.

    Parameters
    ----------
    fig : plt.Figure
        Figure to animate.
    output_path : str
        Output file path.
    n_frames : int
        Number of frames.
    fps : int
        Frames per second.
    func : callable, optional
        Function to call between frames (e.g., for updating view).
    """
    frames = []

    for i in range(n_frames):
        if func is not None:
            func(fig, i, n_frames)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, facecolor="#0a0a1a", bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )

    return output_path


def create_rotating_animation(
    render_func: Callable,
    output_path: str,
    n_frames: int = 36,
    fps: int = 10,
    **kwargs,
) -> str:
    """
    Create rotating animation of a 3D plot.

    Parameters
    ----------
    render_func : callable
        Function that takes (fig, azimuth) and returns a Figure.
    output_path : str
        Output GIF path.
    n_frames : int
        Number of rotation frames.
    fps : int
        Frames per second.
    **kwargs
        Passed to render_func.

    Returns
    -------
    str
        Path to saved GIF.
    """
    frames = []

    fig = plt.figure(figsize=(14, 10), facecolor="#0a0a1a")

    for i in range(n_frames):
        azimuth = (i / n_frames) * 360
        elevation = 25 + 10 * np.sin(2 * np.pi * i / n_frames)

        # Clear previous axes
        fig.clear()
        fig.set_facecolor("#0a0a1a")

        ax = fig.add_subplot(111, projection="3d", facecolor="#0a0a1a")

        # Re-render the plot with new view
        render_func(ax, azimuth, elevation, **kwargs)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, facecolor="#0a0a1a", bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )

    plt.close(fig)
    return output_path


def save_fig(fig: plt.Figure, output_path: str, dpi: int = 150) -> str:
    """
    Save figure to file.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure.
    output_path : str
        Output file path.
    dpi : int
        Resolution.

    Returns
    -------
    str
        Path to saved file.
    """
    fig.savefig(output_path, dpi=dpi, facecolor="#0a0a1a", bbox_inches="tight")
    return output_path


def create_demo_animation(
    barcodes: Dict[str, List[Tuple[float, float]]],
    landscapes: Dict[str, np.ndarray],
    output_path: str = "/tmp/agentic-dream-topoflow/demo.gif",
    n_frames: int = 40,
) -> str:
    """
    Create the main demo animation showcasing all visualizations.

    Parameters
    ----------
    barcodes : dict
        Barcodes dictionary.
    landscapes : dict
        Landscapes dictionary.
    output_path : str
        Output GIF path.
    n_frames : int
        Number of frames.

    Returns
    -------
    str
        Path to saved GIF.
    """
    frames = []

    for i in range(n_frames):
        fig = plt.figure(figsize=(16, 9), facecolor="#0a0a1a")

        # Create a multi-panel layout
        phase = i % 4

        if phase == 0:
            # Panel 1: 3D barcode
            ax = fig.add_subplot(121, projection="3d", facecolor="#0a0a1a")
            _render_3d_barcode_panel(ax, barcodes, i, n_frames)

            # Panel 2: Betti curves
            ax2 = fig.add_subplot(222, facecolor="#0a0a1a")
            _render_betti_panel(ax2, landscapes)

            # Panel 3: Landscape surface
            ax3 = fig.add_subplot(224, projection="3d", facecolor="#0a0a1a")
            _render_landscape_panel(ax3, landscapes, i, n_frames)

        elif phase == 1:
            # Full 3D barcode
            ax = fig.add_subplot(111, projection="3d", facecolor="#0a0a1a")
            _render_3d_barcode_panel(ax, barcodes, i, n_frames, fullscreen=True)

        elif phase == 2:
            # Full landscape surface
            ax = fig.add_subplot(111, projection="3d", facecolor="#0a0a1a")
            _render_landscape_panel(ax, landscapes, i, n_frames, fullscreen=True)

        else:
            # Persistence diagram + info
            ax = fig.add_subplot(121, facecolor="#0a0a1a")
            _render_diagram_panel(ax, barcodes)

            ax2 = fig.add_subplot(122, facecolor="#0a0a1a")
            _render_info_panel(ax2, barcodes)

        # Title with progress
        progress = i / n_frames
        fig.suptitle(
            f"topoflow — Persistent Homology Visualizer  [{int(progress*100)}%]",
            color="white",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, facecolor="#0a0a1a", bbox_inches="tight")
        buf.seek(0)
        frames.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
    )

    return output_path


def _render_3d_barcode_panel(ax, barcodes, frame_idx, n_frames, fullscreen=False):
    """Helper to render 3D barcode panel."""
    colors_betti0 = ["#00ffff", "#00e5e5", "#00cccc", "#00b3b3", "#009999"]
    colors_betti1 = ["#ff00ff", "#e500e5", "#cc00cc", "#b300b3", "#990099"]

    azimuth = (frame_idx / n_frames) * 360
    elevation = 20 + 15 * np.sin(2 * np.pi * frame_idx / n_frames)

    bar_idx = 0
    for dim in [0, 1]:
        key = f"betti_{dim}"
        if key not in barcodes:
            continue
        pairs = barcodes[key]
        sorted_pairs = sorted(pairs, key=lambda x: x[1] - x[0], reverse=True)[:15]

        for j, (birth, death) in enumerate(sorted_pairs):
            persistence = death - birth
            x = bar_idx
            y = birth
            z = 0
            dz = persistence * 0.4 + 0.01
            color = colors_betti0[j % len(colors_betti0)] if dim == 0 else colors_betti1[j % len(colors_betti1)]
            ax.bar3d(x, y, z, dx=0.5, dy=persistence, dz=dz, color=color, alpha=0.85, edgecolor="white", linewidth=0.2)
            bar_idx += 0.5
        bar_idx += 2

    ax.set_xlabel("Feature", color="white", fontsize=9)
    ax.set_ylabel("Birth", color="white", fontsize=9)
    ax.set_zlabel("Persistence", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333333")
    ax.yaxis.pane.set_edgecolor("#333333")
    ax.zaxis.pane.set_edgecolor("#333333")
    ax.view_init(elev=elevation, azim=azimuth)


def _render_betti_panel(ax, landscapes):
    """Helper to render Betti curves panel."""
    ax.set_facecolor("#0a0a1a")
    t = landscapes["t"]
    cmap = plt.cm.cool

    for dim in range(3):
        key = f"landscape_{dim}"
        if key not in landscapes:
            continue
        color = plt.cm.cool(dim / 3)
        ax.plot(t, np.sum(landscapes[key], axis=0), color=color, linewidth=2, label=f"H_{dim}", alpha=0.9)
        ax.fill_between(t, 0, np.sum(landscapes[key], axis=0), color=color, alpha=0.1)

    ax.set_xlabel("t", color="white", fontsize=9)
    ax.set_ylabel("Sum", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_color("#333333")
    ax.spines["right"].set_color("#333333")
    ax.grid(True, color="#222222", alpha=0.3)
    ax.legend(facecolor="#1a1a3a", edgecolor="#333333", labelcolor="white", fontsize=7)


def _render_landscape_panel(ax, landscapes, frame_idx, n_frames, fullscreen=False):
    """Helper to render landscape surface panel."""
    t = landscapes["t"]
    key = "landscape_1"
    if key not in landscapes:
        key = list(landscapes.keys())[1] if len(landscapes) > 1 else None
    if key is None:
        return

    landscape = landscapes[key]
    n_functions = min(landscape.shape[0], 5)

    # Make sure we have proper 2D arrays for plot_surface
    t_2d = np.tile(t, (n_functions, 1))
    k_2d = np.tile(np.arange(n_functions).reshape(-1, 1), (1, len(t)))
    Z = landscape[:n_functions]  # Already (n_functions, n_samples)

    azimuth = (frame_idx / n_frames) * 360
    elevation = 25 + 10 * np.sin(2 * np.pi * frame_idx / n_frames)

    surf = ax.plot_surface(t_2d, k_2d, Z, cmap=TORUS_CMAP, linewidth=0, antialiased=True, alpha=0.85)

    for k in range(n_functions):
        ax.plot(t, [k] * len(t), landscape[k], color=TORUS_CMAP(k / n_functions), linewidth=1.5, alpha=0.7)

    ax.set_xlabel("t", color="white", fontsize=9)
    ax.set_ylabel("k", color="white", fontsize=9)
    ax.set_zlabel("λ", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333333")
    ax.yaxis.pane.set_edgecolor("#333333")
    ax.zaxis.pane.set_edgecolor("#333333")
    ax.view_init(elev=elevation, azim=azimuth)


def _render_diagram_panel(ax, barcodes):
    """Helper to render persistence diagram panel."""
    ax.set_facecolor("#0a0a1a")

    colors = ["#00ffff", "#ff00ff", "#ffff00"]
    for dim in range(3):
        key = f"betti_{dim}"
        if key not in barcodes:
            continue
        pairs = barcodes[key]
        births = [p[0] for p in pairs]
        deaths = [p[1] for p in pairs]
        pers = [p[1] - p[0] for p in pairs]

        if len(births) > 0:
            ax.scatter(
                births,
                deaths,
                c=colors[dim],
                s=[max(p * 50, 10) for p in pers],
                alpha=0.7,
                label=f"H_{dim}",
                edgecolors="white",
                linewidth=0.3,
            )

    max_val = 2.0
    ax.plot([0, max_val], [0, max_val], color="#333333", linestyle="--", linewidth=1)
    ax.set_xlabel("Birth", color="white", fontsize=9)
    ax.set_ylabel("Death", color="white", fontsize=9)
    ax.tick_params(colors="white", labelsize=7)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.spines["top"].set_color("#333333")
    ax.spines["right"].set_color("#333333")
    ax.grid(True, color="#222222", alpha=0.3)
    ax.legend(facecolor="#1a1a3a", edgecolor="#333333", labelcolor="white", fontsize=7)


def _render_info_panel(ax, barcodes):
    """Helper to render info panel."""
    ax.set_facecolor("#0a0a1a")
    ax.axis("off")

    info_text = "╔══════════════════════════╗\n"
    info_text += "║   TOPOFLOW STATISTICS     ║\n"
    info_text += "╠══════════════════════════╣\n"

    for dim in range(3):
        key = f"betti_{dim}"
        if key in barcodes:
            pairs = barcodes[key]
            n_features = len(pairs)
            total_persistence = sum(p[1] - p[0] for p in pairs)
            max_persistence = max((p[1] - p[0] for p in pairs), default=0)
            info_text += f"║ H_{dim}: {n_features:3d} features       ║\n"
            info_text += f"║   Total: {total_persistence:6.2f}          ║\n"
            info_text += f"║   Max:   {max_persistence:6.2f}          ║\n"

    info_text += "╚══════════════════════════╝"

    ax.text(
        0.5,
        0.5,
        info_text,
        transform=ax.transAxes,
        color="#00ffff",
        fontsize=10,
        fontfamily="monospace",
        ha="center",
        va="center",
    )
