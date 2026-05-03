"""
topoflow — 3D Persistent Homology Visualizer

A visually stunning tool for rendering topological landscapes as animated GIFs.
Compute Vietoris-Rips persistent homology and visualize:
- Betti barcodes as 3D bar charts
- Persistence landscapes as 3D surfaces
- VR complexes as 3D meshes
- Animated collapse showing topological features dying

Usage:
    topoflow render --input data.csv --output animate.gif
    topoflow demo
    topoflow random --n-points 100 --distribution torus
"""

__version__ = "0.1.0"
__author__ = "Teerth Sharma"

from .topology import (
    generate_random_point_cloud,
    read_point_cloud,
    compute_persistent_homology,
    extract_barcodes,
    compute_betti_curves,
    compute_persistence_landscape,
    get_topological_summary,
)

from .render import (
    create_3d_barcode_chart,
    create_3d_persistence_surface,
    create_vr_complex_visualization,
    create_betti_curves_plot,
    create_persistence_diagram,
    create_collapse_animation,
    create_demo_animation,
    save_fig,
)

__all__ = [
    # Version
    "__version__",
    # Topology
    "generate_random_point_cloud",
    "read_point_cloud",
    "compute_persistent_homology",
    "extract_barcodes",
    "compute_betti_curves",
    "compute_persistence_landscape",
    "get_topological_summary",
    # Render
    "create_3d_barcode_chart",
    "create_3d_persistence_surface",
    "create_vr_complex_visualization",
    "create_betti_curves_plot",
    "create_persistence_diagram",
    "create_collapse_animation",
    "create_demo_animation",
    "save_fig",
]
