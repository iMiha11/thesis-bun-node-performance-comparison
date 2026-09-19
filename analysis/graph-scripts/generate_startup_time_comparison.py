from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


# ---------------------------------------------------------
# POTI
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

CSV_FILE = (
    ROOT_DIR
    / "startup-benchmark"
    / "results"
    / "startup_measurements.csv"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "figures"
    / "startup-time-comparison.png"
)


# ---------------------------------------------------------
# NASTAVITVE GRAFA
# ---------------------------------------------------------

NUMBER_OF_MEASUREMENTS = 100

NODE_COLOR = "#2ca02c"
BUN_COLOR = "#fb7a24"

NODE_POINT_COLOR = "#1b5e20"
BUN_POINT_COLOR = "#a84300"


# ---------------------------------------------------------
# BRANJE PODATKOV
# ---------------------------------------------------------

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Datoteka z meritvami ne obstaja: {CSV_FILE}"
    )

data = pd.read_csv(CSV_FILE)

# Odstranimo morebitne presledke in podpičja iz imen stolpcev.
data.columns = (
    data.columns
    .str.strip()
    .str.replace(";", "", regex=False)
)

required_columns = {
    "node_startup_ms",
    "bun_startup_ms"
}

missing_columns = required_columns.difference(data.columns)

if missing_columns:
    raise ValueError(
        "V CSV-datoteki manjkajo stolpci: "
        + ", ".join(sorted(missing_columns))
    )


# ---------------------------------------------------------
# ČIŠČENJE PODATKOV
# ---------------------------------------------------------

for column in required_columns:
    data[column] = (
        data[column]
        .astype(str)
        .str.replace(";", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.strip()
    )

    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

data = data.dropna(
    subset=["node_startup_ms", "bun_startup_ms"]
)

if len(data) < NUMBER_OF_MEASUREMENTS:
    raise ValueError(
        f"Najdenih je samo {len(data)} veljavnih meritev, "
        f"potrebnih pa je {NUMBER_OF_MEASUREMENTS}."
    )

# Uporabimo prvih 100 veljavnih meritev.
data = data.iloc[:NUMBER_OF_MEASUREMENTS].copy()


# ---------------------------------------------------------
# IZRAČUN STATISTIKE
# ---------------------------------------------------------

node_values = data["node_startup_ms"].to_numpy()
bun_values = data["bun_startup_ms"].to_numpy()

node_mean = np.mean(node_values)
bun_mean = np.mean(bun_values)

node_stdev = np.std(node_values, ddof=1)
bun_stdev = np.std(bun_values, ddof=1)

# 95-% interval zaupanja za aritmetično sredino.
node_ci_95 = (
    t.ppf(0.975, df=len(node_values) - 1)
    * node_stdev
    / np.sqrt(len(node_values))
)

bun_ci_95 = (
    t.ppf(0.975, df=len(bun_values) - 1)
    * bun_stdev
    / np.sqrt(len(bun_values))
)

means = [
    node_mean,
    bun_mean
]

confidence_intervals = [
    node_ci_95,
    bun_ci_95
]

labels = [
    "Node.js",
    "Bun.js"
]

colors = [
    NODE_COLOR,
    BUN_COLOR
]

positions = np.arange(len(labels))


# ---------------------------------------------------------
# RISANJE GRAFA
# ---------------------------------------------------------

figure, axis = plt.subplots(
    figsize=(10, 7)
)

bars = axis.bar(
    positions,
    means,
    yerr=confidence_intervals,
    capsize=8,
    color=colors,
    edgecolor="black",
    linewidth=1.1,
    alpha=0.85,
    width=0.45,
    error_kw={
        "elinewidth": 1.5,
        "capthick": 1.5
    },
    zorder=2
)


# ---------------------------------------------------------
# PRIKAZ POSAMEZNIH MERITEV
# ---------------------------------------------------------

# Fiksno seme zagotovi enako razporeditev točk pri vsakem zagonu.
random_generator = np.random.default_rng(seed=42)

node_jitter = random_generator.normal(
    loc=positions[0],
    scale=0.035,
    size=len(node_values)
)

bun_jitter = random_generator.normal(
    loc=positions[1],
    scale=0.035,
    size=len(bun_values)
)

axis.scatter(
    node_jitter,
    node_values,
    color=NODE_POINT_COLOR,
    alpha=0.65,
    s=22,
    zorder=3
)

axis.scatter(
    bun_jitter,
    bun_values,
    color=BUN_POINT_COLOR,
    alpha=0.65,
    s=22,
    zorder=3
)


# ---------------------------------------------------------
# NASLOVI IN OZNAKE
# ---------------------------------------------------------

axis.set_title(
    "Primerjava časa zagona aplikacije ($n = 100$)",
    fontsize=15,
    fontweight="bold"
)

axis.set_xlabel(
    "Izvajalno okolje",
    fontsize=12,
    fontweight="bold"
)

axis.set_ylabel(
    "Čas zagona (ms)",
    fontsize=12,
    fontweight="bold"
)

axis.set_xticks(positions)
axis.set_xticklabels(labels)

axis.set_ylim(0, 150)

axis.grid(
    axis="y",
    linestyle="--",
    alpha=0.4,
    zorder=1
)

axis.set_axisbelow(True)


# ---------------------------------------------------------
# IZPIS POVPREČIJ NAD STOLPCI
# ---------------------------------------------------------

for bar, mean, confidence_interval in zip(
    bars,
    means,
    confidence_intervals
):
    axis.text(
        bar.get_x() + bar.get_width() / 2,
        mean + confidence_interval + 2,
        f"{mean:.2f} ms".replace(".", ","),
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
            "pad": 1.5
        },
        zorder=5
    )


# ---------------------------------------------------------
# SHRANJEVANJE GRAFA
# ---------------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

figure.tight_layout()

figure.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close(figure)


# ---------------------------------------------------------
# IZPIS REZULTATOV
# ---------------------------------------------------------

print(f"Graf je shranjen v: {OUTPUT_FILE}")

print("\nIzračunane vrednosti:")

print(
    f"Node.js: povprečje = {node_mean:.3f} ms, "
    f"standardni odklon = {node_stdev:.3f} ms, "
    f"95-% IZ = ±{node_ci_95:.3f} ms"
)

print(
    f"Bun.js: povprečje = {bun_mean:.3f} ms, "
    f"standardni odklon = {bun_stdev:.3f} ms, "
    f"95-% IZ = ±{bun_ci_95:.3f} ms"
)