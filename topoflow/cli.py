"""
topoflow.cli — Command-line interface for topoflow.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np

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


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="topoflow",
        description="topoflow — 3D Persistent Homology Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  topoflow render --input data.csv --output visualize.gif
  topoflow demo
  topoflow random --n-points 100 --distribution circle
  topoflow barcode --input points.txt --output barcode.png
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Render command
    render_parser = subparsers.add_parser(
        "render",
        help="Render persistent homology visualization from input data",
    )
    render_parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input point cloud file (CSV, TXT)",
    )
    render_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="output.gif",
        help="Output file (GIF, PNG)",
    )
    render_parser.add_argument(
        "--max-dim",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Maximum homology dimension",
    )
    render_parser.add_argument(
        "--max-edge",
        type=float,
        default=2.0,
        help="Maximum edge length for Rips filtration",
    )
    render_parser.add_argument(
        "--n-frames",
        type=int,
        default=30,
        help="Number of animation frames",
    )
    render_parser.add_argument(
        "--type",
        "-t",
        type=str,
        default="auto",
        choices=["auto", "barcode", "landscape", "diagram", "collapse", "vr"],
        help="Visualization type",
    )

    # Demo command
    demo_parser = subparsers.add_parser(
        "demo",
        help="Generate demo animation showcasing topoflow capabilities",
    )
    demo_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="demo.gif",
        help="Output file path",
    )
    demo_parser.add_argument(
        "--n-frames",
        type=int,
        default=40,
        help="Number of animation frames",
    )
    demo_parser.add_argument(
        "--n-points",
        type=int,
        default=60,
        help="Number of random points to generate",
    )

    # Random command
    random_parser = subparsers.add_parser(
        "random",
        help="Generate random point cloud and compute PH",
    )
    random_parser.add_argument(
        "--n-points",
        type=int,
        default=50,
        help="Number of points",
    )
    random_parser.add_argument(
        "--n-dims",
        type=int,
        default=2,
        help="Dimensionality of embedding space",
    )
    random_parser.add_argument(
        "--distribution",
        type=str,
        default="uniform",
        choices=["uniform", "circle", "sphere", "torus"],
        help="Point distribution",
    )
    random_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    random_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="random.png",
        help="Output file",
    )
    random_parser.add_argument(
        "--viz",
        type=str,
        default="barcode",
        choices=["barcode", "landscape", "diagram", "vr", "all"],
        help="Visualization type",
    )

    # Barcode command
    barcode_parser = subparsers.add_parser(
        "barcode",
        help="Generate Betti barcode visualization",
    )
    barcode_parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input file (optional, generates random if not provided)",
    )
    barcode_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="barcode.png",
        help="Output file",
    )
    barcode_parser.add_argument(
        "--elevation",
        type=float,
        default=25,
        help="Camera elevation",
    )
    barcode_parser.add_argument(
        "--azimuth",
        type=float,
        default=45,
        help="Camera azimuth",
    )

    # Collapse command
    collapse_parser = subparsers.add_parser(
        "collapse",
        help="Generate topological collapse animation",
    )
    collapse_parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input file (optional, generates random if not provided)",
    )
    collapse_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="collapse.gif",
        help="Output file",
    )
    collapse_parser.add_argument(
        "--n-frames",
        type=int,
        default=30,
        help="Number of frames",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        print("\n[topoflow] Use 'topoflow demo' to generate a demo animation!")
        return 0

    try:
        if args.command == "render":
            return handle_render(args)
        elif args.command == "demo":
            return handle_demo(args)
        elif args.command == "random":
            return handle_random(args)
        elif args.command == "barcode":
            return handle_barcode(args)
        elif args.command == "collapse":
            return handle_collapse(args)
        else:
            parser.print_help()
            return 1
    except Exception as e:
        print(f"[topoflow] Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def handle_render(args):
    """Handle the render command."""
    print(f"[topoflow] Loading data from {args.input}...")

    # Load or generate data
    if os.path.exists(args.input):
        points = read_point_cloud(args.input)
    else:
        print(f"[topoflow] File not found: {args.input}")
        print("[topoflow] Generating random point cloud instead...")
        points = generate_random_point_cloud(n_points=50, seed=42)

    print(f"[topoflow] Computing persistent homology on {len(points)} points...")

    # Compute PH
    result = compute_persistent_homology(
        points,
        max_dim=args.max_dim,
        thresh=args.max_edge,
    )

    dgms = result["dgms"]
    barcodes = extract_barcodes(dgms)
    betti_curves = compute_betti_curves(barcodes)
    landscapes = compute_persistence_landscape(barcodes)

    # Summary
    summary = get_topological_summary(dgms)
    print(f"[topoflow] Betti numbers: H_0={summary.get('betti_0', 0)}, H_1={summary.get('betti_1', 0)}")

    # Determine output type
    output_ext = Path(args.output).suffix.lower()

    if output_ext == ".gif":
        print(f"[topoflow] Creating {args.n_frames}-frame animation...")
        create_demo_animation(
            barcodes,
            landscapes,
            output_path=args.output,
            n_frames=args.n_frames,
        )
    else:
        print(f"[topoflow] Creating static visualization...")
        if args.type == "auto":
            viz_type = "barcode"
        else:
            viz_type = args.type

        if viz_type == "barcode":
            fig = create_3d_barcode_chart(barcodes)
        elif viz_type == "landscape":
            fig = create_3d_persistence_surface(landscapes)
        elif viz_type == "diagram":
            fig = create_persistence_diagram(dgms)
        elif viz_type == "collapse":
            path = create_collapse_animation(
                barcodes,
                betti_curves,
                n_frames=args.n_frames,
                output_path=args.output,
            )
            print(f"[topoflow] Saved: {path}")
            return 0
        elif viz_type == "vr":
            fig = create_vr_complex_visualization(points)
        else:
            fig = create_3d_barcode_chart(barcodes)

        save_fig(fig, args.output)

    print(f"[topoflow] Saved: {args.output}")
    return 0


def handle_demo(args):
    """Handle the demo command."""
    print("[topoflow] Generating demo animation...")
    print("[topoflow] Creating random point cloud...")

    # Generate interesting point cloud (torus)
    points = generate_random_point_cloud(
        n_points=args.n_points,
        distribution="torus",
        seed=42,
    )

    print(f"[topoflow] Computing persistent homology on {len(points)} points...")

    # Compute PH
    result = compute_persistent_homology(
        points,
        max_dim=2,
        thresh=3.0,
    )

    dgms = result["dgms"]
    barcodes = extract_barcodes(dgms)
    landscapes = compute_persistence_landscape(barcodes, n_functions=5, n_samples=100)

    # Summary
    summary = get_topological_summary(dgms)
    print(f"[topoflow] Betti numbers: H_0={summary.get('betti_0', 0)}, H_1={summary.get('betti_1', 0)}")

    print(f"[topoflow] Creating {args.n_frames}-frame demo animation...")
    print("[topoflow] This showcases:")
    print("  • 3D Betti barcode visualization")
    print("  • Persistence landscape surface")
    print("  • Persistence diagram")
    print("  • Topological statistics")

    output_path = args.output
    create_demo_animation(
        barcodes,
        landscapes,
        output_path=output_path,
        n_frames=args.n_frames,
    )

    print(f"[topoflow] Demo saved: {output_path}")
    return 0


def handle_random(args):
    """Handle the random command."""
    print(f"[topoflow] Generating random point cloud...")
    print(f"  Points: {args.n_points}")
    print(f"  Dims: {args.n_dims}")
    print(f"  Distribution: {args.distribution}")
    print(f"  Seed: {args.seed}")

    points = generate_random_point_cloud(
        n_points=args.n_points,
        n_dims=args.n_dims,
        distribution=args.distribution,
        seed=args.seed,
    )

    print(f"[topoflow] Computing persistent homology...")
    result = compute_persistent_homology(points, max_dim=1)

    dgms = result["dgms"]
    barcodes = extract_barcodes(dgms)
    betti_curves = compute_betti_curves(barcodes)
    landscapes = compute_persistence_landscape(barcodes)

    summary = get_topological_summary(dgms)
    print(f"[topoflow] Results:")
    print(f"  H_0 features: {summary.get('betti_0', 0)}")
    print(f"  H_1 features: {summary.get('betti_1', 0)}")
    print(f"  Total persistence H_0: {summary.get('total_persistence_0', 0):.3f}")
    print(f"  Total persistence H_1: {summary.get('total_persistence_1', 0):.3f}")

    print(f"[topoflow] Creating visualization...")
    output_ext = Path(args.output).suffix.lower()

    if args.viz == "all":
        # Create multi-panel figure
        fig = plt.figure(figsize=(16, 12), facecolor="#0a0a1a")
        fig.add_subplot(221, projection="3d", facecolor="#0a0a1a")
        fig.add_subplot(222, projection="3d", facecolor="#0a0a1a")
        fig.add_subplot(223, facecolor="#0a0a1a")
        fig.add_subplot(224, facecolor="#0a0a1a")

        # This is simplified - just create barcode for 'all'
        fig = create_3d_barcode_chart(barcodes)
    elif args.viz == "barcode":
        fig = create_3d_barcode_chart(barcodes)
    elif args.viz == "landscape":
        fig = create_3d_persistence_surface(landscapes)
    elif args.viz == "diagram":
        fig = create_persistence_diagram(dgms)
    elif args.viz == "vr":
        fig = create_vr_complex_visualization(points)

    save_fig(fig, args.output)
    print(f"[topoflow] Saved: {args.output}")
    return 0


def handle_barcode(args):
    """Handle the barcode command."""
    if args.input and os.path.exists(args.input):
        print(f"[topoflow] Loading data from {args.input}...")
        points = read_point_cloud(args.input)
    else:
        print("[topoflow] Generating random point cloud...")
        points = generate_random_point_cloud(n_points=50, seed=42)

    print("[topoflow] Computing persistent homology...")
    result = compute_persistent_homology(points, max_dim=1)
    dgms = result["dgms"]
    barcodes = extract_barcodes(dgms)

    print("[topoflow] Rendering 3D barcode chart...")
    fig = create_3d_barcode_chart(
        barcodes,
        elevation=args.elevation,
        azimuth=args.azimuth,
    )

    save_fig(fig, args.output)
    print(f"[topoflow] Saved: {args.output}")
    return 0


def handle_collapse(args):
    """Handle the collapse command."""
    if args.input and os.path.exists(args.input):
        print(f"[topoflow] Loading data from {args.input}...")
        points = read_point_cloud(args.input)
    else:
        print("[topoflow] Generating random point cloud...")
        points = generate_random_point_cloud(n_points=50, seed=42)

    print("[topoflow] Computing persistent homology...")
    result = compute_persistent_homology(points, max_dim=1)
    dgms = result["dgms"]
    barcodes = extract_barcodes(dgms)
    betti_curves = compute_betti_curves(barcodes)

    print(f"[topoflow] Creating {args.n_frames}-frame collapse animation...")
    path = create_collapse_animation(
        barcodes,
        betti_curves,
        n_frames=args.n_frames,
        output_path=args.output,
    )

    print(f"[topoflow] Saved: {path}")
    return 0


# Need to import matplotlib for random command
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


if __name__ == "__main__":
    sys.exit(main())
