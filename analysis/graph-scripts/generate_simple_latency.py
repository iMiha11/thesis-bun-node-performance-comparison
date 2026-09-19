from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import t


# ---------------------------------------------------------
# POTI
# ---------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]

CSV_FILE = (
    ROOT_DIR
    / "results"
    / "load-tests-20260904-052953"
    / "summary-runs.csv"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "figures"
    / "simple-latency-comparison.png"
)


# ---------------------------------------------------------
# NASTAVITVE
# ---------------------------------------------------------

SCENARIO = "simple"
EXPECTED_REPETITIONS = 25

RUNTIMES = [
    "node",
    "bun"
]

RUNTIME_COLORS = {
    "node": "#2ca02c",
    "bun": "#ff7f0e"
}

RUNTIME_MARKERS = {
    "node": "o",
    "bun": "^"
}

RUNTIME_POSITIONS = {
    "node": 0,
    "bun": 1
}


# ---------------------------------------------------------
# BRANJE PODATKOV
# ---------------------------------------------------------

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Datoteka ne obstaja: {CSV_FILE}"
    )

data = pd.read_csv(
    CSV_FILE,
    decimal=",",
    encoding="utf-8-sig"
)

required_columns = {
    "scenario",
    "runtime",
    "repetition",
    "latency_avg_ms"
}

missing_columns = required_columns.difference(
    data.columns
)

if missing_columns:
    raise ValueError(
        "Manjkajo stolpci: "
        + ", ".join(sorted(missing_columns))
    )

data["scenario"] = (
    data["scenario"]
    .astype(str)
    .str.strip()
    .str.replace(r"^/", "", regex=True)
)

data["runtime"] = (
    data["runtime"]
    .astype(str)
    .str.strip()
    .str.lower()
)

data["repetition"] = pd.to_numeric(
    data["repetition"],
    errors="coerce"
)

data["latency_avg_ms"] = pd.to_numeric(
    data["latency_avg_ms"],
    errors="coerce"
)

data = data.dropna(
    subset=[
        "scenario",
        "runtime",
        "repetition",
        "latency_avg_ms"
    ]
)

data = data[
    data["scenario"] == SCENARIO
].copy()


# ---------------------------------------------------------
# STATISTIKA
# ---------------------------------------------------------

def calculate_statistics(
    values: np.ndarray
) -> tuple[float, float]:
    count = len(values)

    mean = float(
        np.mean(values)
    )

    standard_deviation = float(
        np.std(values, ddof=1)
    )

    critical_value = t.ppf(
        0.975,
        df=count - 1
    )

    confidence_interval = (
        critical_value
        * standard_deviation
        / np.sqrt(count)
    )

    return mean, confidence_interval


# ---------------------------------------------------------
# PREVERJANJE MERITEV
# ---------------------------------------------------------

for runtime in RUNTIMES:
    measurement_count = len(
        data[
            data["runtime"] == runtime
        ]
    )

    if measurement_count != EXPECTED_REPETITIONS:
        raise ValueError(
            f"Okolje {runtime}: najdenih je "
            f"{measurement_count} meritev, "
            f"pričakovanih pa je "
            f"{EXPECTED_REPETITIONS}."
        )


# ---------------------------------------------------------
# RISANJE GRAFA
# ---------------------------------------------------------

figure, axis = plt.subplots(
    figsize=(11.5, 6.4)
)

random_generator = np.random.default_rng(
    seed=42
)

for runtime in RUNTIMES:
    selected_data = data[
        data["runtime"] == runtime
    ].sort_values("repetition")

    values = selected_data[
        "latency_avg_ms"
    ].to_numpy()

    mean, confidence_interval = (
        calculate_statistics(values)
    )

    position = RUNTIME_POSITIONS[runtime]

    jitter = random_generator.normal(
        loc=position,
        scale=0.035,
        size=len(values)
    )

    axis.scatter(
        jitter,
        values,
        color=RUNTIME_COLORS[runtime],
        marker=RUNTIME_MARKERS[runtime],
        edgecolor="black",
        linewidth=0.3,
        alpha=0.60,
        s=34,
        zorder=2
    )

    axis.errorbar(
        position,
        mean,
        yerr=confidence_interval,
        fmt="D",
        color=RUNTIME_COLORS[runtime],
        markerfacecolor=RUNTIME_COLORS[runtime],
        markeredgecolor="black",
        markeredgewidth=0.9,
        markersize=9,
        capsize=7,
        elinewidth=1.6,
        capthick=1.6,
        zorder=4
    )

    formatted_mean = (
        f"{mean:.3f} ms"
        .replace(".", ",")
    )

    axis.text(
        position,
        mean + confidence_interval + 0.04,
        formatted_mean,
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        zorder=5
    )


# ---------------------------------------------------------
# NASLOV IN OSI
# ---------------------------------------------------------

axis.set_title(
    "Primerjava povprečne latence "
    "pri scenariju /simple",
    fontsize=14,
    fontweight="bold",
    pad=8
)

axis.set_xlabel(
    "Izvajalno okolje",
    fontsize=11,
    fontweight="bold"
)

axis.set_ylabel(
    "Povprečna latenca posamezne ponovitve (ms)",
    fontsize=11,
    fontweight="bold"
)

axis.set_xticks([
    0,
    1
])

axis.set_xticklabels([
    "Node.js",
    "Bun.js"
])

axis.set_xlim(
    -0.15,
    1.15
)

axis.set_ylim(
    3.8,
    4.94
)

axis.set_yticks(
    np.arange(
        4.0,
        4.81,
        0.2
    )
)

axis.tick_params(
    axis="both",
    labelsize=10
)

axis.grid(
    axis="y",
    linestyle="--",
    linewidth=0.8,
    alpha=0.30,
    zorder=1
)

axis.set_axisbelow(True)


# ---------------------------------------------------------
# LEGENDA
# ---------------------------------------------------------

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor=RUNTIME_COLORS["node"],
        markeredgecolor="black",
        markersize=7,
        label="Node.js"
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        linestyle="none",
        markerfacecolor=RUNTIME_COLORS["bun"],
        markeredgecolor="black",
        markersize=7,
        label="Bun.js"
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        linestyle="-",
        color="black",
        markerfacecolor="black",
        markersize=6,
        label="Povprečje ± 95 % IZ"
    )
]

axis.legend(
    handles=legend_handles,
    title="Izvajalno okolje",
    loc="upper right",
    fontsize=10,
    title_fontsize=10,
    frameon=True
)


# ---------------------------------------------------------
# SHRANJEVANJE
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

print(f"Graf je shranjen v: {OUTPUT_FILE}")

plt.show()
plt.close(figure)