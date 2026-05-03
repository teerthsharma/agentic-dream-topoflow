# topoflow 🌊

### 3D Persistent Homology Visualizer

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/downloads/)

**topoflow** renders topological landscapes as stunning animated GIFs. Compute Vietoris-Rips persistent homology from point clouds and visualize Betti barcodes, persistence landscapes, and VR complexes in 3D.

> *"Topology is the study of properties preserved under continuous deformations."*
> topoflow makes these abstract properties viscerally beautiful.

---

## ✨ Features

- 🎨 **3D Betti Barcodes** — Persistence intervals rendered as colorful 3D bar charts
- 🌊 **Persistence Landscapes** — Topological features shown as flowing 3D surfaces
- 🔮 **VR Complex Visualization** — Vietoris-Rips simplicial complex rendered as 3D mesh
- 🎬 **Animated Collapse** — Watch topological features "die" as threshold increases
- 🎞️ **GIF Export** — Beautiful animated outputs perfect for Twitter/social media
- 🖼️ **PNG Export** — High-resolution static visualizations

---

## 🚀 Quick Start

```bash
# Install
pip install -e .

# Generate demo animation
topoflow demo

# Render from your data
topoflow render --input points.csv --output visualize.gif

# Generate random point cloud
topoflow random --n-points 100 --distribution torus --output torus.png

# Create collapse animation
topoflow collapse --output collapse.gif
```

---

## 📊 Example Outputs

### 3D Betti Barcode
![Betti Barcode](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExbzB5dWxwYWRmbzBqYW1xbndlYnE4dWN4ZGticWRsdDltdWN2ZSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/demo_betti.gif)

### Persistence Landscape Surface
Stunning flowing surfaces representing the "landscape" of topological features across scales.

### Topological Collapse Animation
Watch connected components merge and loops collapse as the filtration threshold increases.

---

## 🧮 How It Works

### Input: Point Cloud
```
x1, y1
x2, y2
x3, y3
...
```

### Vietoris-Rips Filtration
Construct a simplicial complex where vertices are points and higher simplices exist when pairwise distances are below threshold τ.

### Persistent Homology
Track birth and death of topological features (H₀: components, H₁: loops, H₂: voids) as τ increases.

### Output: Beautiful Visualizations
- **Betti numbers**: Count of features at each threshold
- **Barcodes**: Birth-death intervals for each feature
- **Landscapes**: Functional summary of persistence

---

## 🎛️ CLI Options

```bash
topoflow render --input data.csv --output animate.gif [options]

Options:
  --input, -i        Input point cloud file (CSV, TXT)
  --output, -o       Output file (GIF, PNG) [default: output.gif]
  --max-dim          Maximum homology dimension [default: 1]
  --max-edge         Maximum edge length for Rips [default: 2.0]
  --n-frames         Number of animation frames [default: 30]
  --type, -t         Visualization type:
                     auto, barcode, landscape, diagram, collapse, vr
```

### Visualization Types

| Type | Description |
|------|-------------|
| `barcode` | 3D bar chart of Betti_0 and Betti_1 persistence |
| `landscape` | 3D surface of persistence landscape |
| `diagram` | 2D scatter plot (birth vs death) |
| `collapse` | Animated GIF showing feature collapse |
| `vr` | VR complex point cloud |
| `all` | Multi-panel combined view |

---

## 🔬 Point Distributions

```bash
# Uniform random in unit square/cube
topoflow random --distribution uniform

# Points on a circle
topoflow random --distribution circle

# Points on a sphere
topoflow random --distribution sphere --n-dims 3

# Points on a torus (best for H_1 features!)
topoflow random --distribution torus --n-points 100 --output torus.png
```

---

## 📦 Package Structure

```
topoflow/
├── __init__.py       # Package exports
├── topology.py       # PH computation via ripser
├── render.py         # 3D visualization functions
└── cli.py            # Command-line interface
```

---

## 🛠️ API Usage

```python
from topoflow import (
    generate_random_point_cloud,
    compute_persistent_homology,
    extract_barcodes,
    create_3d_barcode_chart,
    create_demo_animation,
)

# Generate data
points = generate_random_point_cloud(n_points=100, distribution="torus")

# Compute PH
result = compute_persistent_homology(points, max_dim=2)
dgms = result["dgms"]

# Extract barcodes
barcodes = extract_barcodes(dgms)

# Visualize
fig = create_3d_barcode_chart(barcodes)
fig.savefig("barcode.png")

# Or create animation
create_demo_animation(barcodes, landscapes, output_path="demo.gif")
```

---

## 📚 Dependencies

- **ripser** — Fast Ripser-based persistent homology
- **numpy** — Numerical computing
- **matplotlib** — 3D plotting
- **pillow** — GIF generation
- **scipy** — Scientific utilities

---

## 🎨 Color Schemes

topoflow uses custom neon/cyberpunk colormaps:

| Scheme | Colors |
|--------|--------|
| `torus` | Deep purple → magenta → cyan |
| `electric` | Black → purple → violet → white |
| `sunset` | Dark blue → purple → pink → orange |
| `neon` | Black → purple → violet → lavender |

---

## ⭐ Star History

If topoflow makes topology beautiful for you, please star this repo!

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*"Persistent homology reveals the topological structure hidden in data."*
