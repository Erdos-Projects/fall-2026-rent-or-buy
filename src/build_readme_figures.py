"""Draw the README's Ohio map and project workflow from local inputs."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch
import numpy as np


# =============================================================================
# 1. PATHS AND A SHARED VISUAL STYLE
# =============================================================================
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "figures/readme"
OUTPUT.mkdir(parents=True, exist_ok=True)
boundary = json.loads((ROOT / "data/metadata/ohio_county_boundaries_2024.geojson").read_text())
COUNTIES = {"Butler", "Cuyahoga", "Franklin", "Hamilton", "Lucas",
            "Montgomery", "Stark", "Summit", "Warren"}
TEAL, INK, PAPER, GREY = "#1d6b66", "#1c2826", "#fbf8f1", "#e6e6df"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK,
                     "svg.fonttype": "none", "svg.hashsalt": "ohio-housing"})


# =============================================================================
# 2. OHIO COUNTY MAP: SOURCE BOUNDARIES, WITH ALL NINE STUDY COUNTIES LABELED
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 7.4), facecolor=PAPER)
ax.set_facecolor(PAPER)
centers = {}
for feature in boundary["features"]:
    name = feature["properties"]["county_name"]
    polygons = feature["geometry"]["coordinates"]
    largest_area = 0
    for rings in polygons:
        points = np.array(rings[0])
        # A local equirectangular display scales longitude at latitude 40°N.
        # These are display coordinates, not a measurement of county area.
        points[:, 0] *= np.cos(np.deg2rad(40))
        cross = points[:-1, 0] * points[1:, 1] - points[1:, 0] * points[:-1, 1]
        signed_area = cross.sum() / 2
        ax.add_patch(Polygon(points, facecolor=TEAL if name in COUNTIES else GREY,
                             edgecolor="white", linewidth=0.8, zorder=2 if name in COUNTIES else 1))
        if abs(signed_area) > largest_area:
            largest_area = abs(signed_area)
            centers[name] = ((points[:-1] + points[1:]) * cross[:, None]).sum(axis=0) / (6 * signed_area)

# Callouts avoid crowding the adjacent counties around Cincinnati and Akron.
label_positions = {
    "Lucas": (-84.60, 41.90), "Cuyahoga": (-79.85, 41.65),
    "Summit": (-79.85, 41.05), "Stark": (-79.85, 40.50),
    "Franklin": (-81.15, 39.75), "Montgomery": (-85.40, 40.15),
    "Butler": (-85.40, 39.55), "Hamilton": (-85.40, 38.95),
    "Warren": (-82.70, 38.80),
}
for name, (lon, lat) in label_positions.items():
    x = lon * np.cos(np.deg2rad(40))
    cx, cy = centers[name]
    ax.plot(cx, cy, "o", color="white", markersize=4, zorder=4)
    ax.annotate(name, xy=(cx, cy), xytext=(x, lat), fontsize=13, weight="bold",
                ha="left" if x > cx else "right", va="center", color=TEAL,
                arrowprops={"arrowstyle": "-", "color": TEAL, "lw": 1.1,
                            "connectionstyle": "angle,angleA=0,angleB=90,rad=5"},
                bbox={"facecolor": PAPER, "edgecolor": "none", "pad": 3}, zorder=5)
ax.set_xlim(-66.45, -60.30)
ax.set_ylim(38.25, 42.30)
ax.set_aspect("equal")
ax.axis("off")
fig.text(0.075, 0.94, "Nine counties. One Ohio housing study.", fontsize=23, weight="bold")
fig.text(0.075, 0.895, "Highlighted counties form the modeling panel; other counties provide context.", fontsize=12)
fig.text(0.075, 0.045, "Boundary source: U.S. Census Bureau · 2024 cartographic boundaries · 1:20,000,000", fontsize=10, color="#626b66")
fig.subplots_adjust(left=0.04, right=0.96, top=0.87, bottom=0.08)
fig.savefig(OUTPUT / "ohio_study_counties.svg", metadata={"Date": None}, facecolor=PAPER)
fig.savefig(OUTPUT / "ohio_study_counties.png", dpi=170, facecolor=PAPER)
plt.close(fig)


# =============================================================================
# 3. A COMPACT OVERVIEW OF HOW DATA BECOMES A FINANCIAL SCENARIO
# =============================================================================
fig, ax = plt.subplots(figsize=(13, 2.0), facecolor=PAPER)
ax.set_xlim(0, 13)
ax.set_ylim(0, 2)
ax.axis("off")
steps = [
    ("01", "COLLECT", "Housing, people & economy"),
    ("02", "PREDICT", "One-year price & rent growth"),
    ("03", "COMPARE", "Buy vs. rent + invest"),
    ("04", "EXPLORE", "Your inputs, different outcomes"),
]
for i, (number, title, detail) in enumerate(steps):
    x = 0.12 + i * 3.25
    ax.add_patch(FancyBboxPatch((x, 0.30), 2.95, 1.40,
                               boxstyle="round,pad=0.02,rounding_size=0.10",
                               facecolor="#e7efeb", edgecolor="#c7d8d0"))
    ax.text(x + 0.17, 1.30, number, fontsize=17, color=TEAL, weight="bold")
    ax.text(x + 0.77, 1.30, title, fontsize=19, weight="bold")
    ax.text(x + 0.17, 0.78, detail, fontsize=13.5)
    if i < 3:
        ax.annotate("", (x + 3.23, 1.0), (x + 3.00, 1.0),
                    arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.5})
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
fig.savefig(OUTPUT / "project_workflow.svg", metadata={"Date": None}, facecolor=PAPER)
plt.close(fig)
for svg_path in OUTPUT.glob("*.svg"):
    # Matplotlib's path output contains harmless trailing spaces; normalize
    # them so regenerated figures also pass the repository whitespace check.
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
print("Saved README map and workflow to figures/readme/.")
