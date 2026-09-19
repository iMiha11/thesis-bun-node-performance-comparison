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
    / "results"
    / "load-tests-20260904-052953"
    / "summary-runs.csv"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "figures"
    / "selected-scenarios-latency.png"
)


# ---------------------------------------------------------
# NASTAVITVE
# ---------------------------------------------------------

SCENARIOS = [
    "file-read",
    "file-write",
    "json",
    "auth"
]

RUNTIMES = [
    "node",
    "bun"
]

EXPECTED_REPETITIONS = 25

RUNTIME_LABELS = {
    "node": "Node.js",
    "bun": "Bun.js"
}

RUNTIME_COLORS = {
    "node": "#2ca02c",
    "bun": "#ff7f0e"
}

RUNTIME_MARKERS = {
    "node": "o",
    "bun": "^"
}

RUNTIME_OFFSETS = {
    "node": -0.18,
    "bun": 0.18
}


# ---------------------------------------------------------
# BRANJE PODATKOV
# ---------------------------------------------------------

if not CSV_FILE.exists():
    raise FileNotFoundError(
        f"Datoteka z rezultati ne obstaja: {CSV_FILE}"
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
        "V datoteki manjkajo stolpci: "
        + ", ".join(sorted(missing_columns))
    )


# ---------------------------------------------------------
# ČIŠČENJE PODATKOV
# ---------------------------------------------------------

data["scenario"] = (
    data["scenario"]
    .astype(str)
    .str.strip()
    .str.removeprefix("/")
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
    data["scenario"].isin(SCENARIOS)
    & data["runtime"].isin(RUNTIMES)
].copy()


# ---------------------------------------------------------
# PREVERJANJE ŠTEVILA MERITEV
# ---------------------------------------------------------

for scenario in SCENARIOS:
    for runtime in RUNTIMES:
        selected_data = data[
            (data["scenario"] == scenario)
            & (data["runtime"] == runtime)
        ]

        number_of_measurements = len(
            selected_data
        )

        if (
            number_of_measurements
            != EXPECTED_REPETITIONS
        ):
            raise ValueError(
                f"Scenarij /{scenario}, "
                f"okolje {runtime}: najdenih je "
                f"{number_of_measurements} meritev, "
                f"pričakovanih pa je "
                f"{EXPECTED_REPETITIONS}."
            )


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
# PRIPRAVA GRAFA
# ---------------------------------------------------------

figure, axis = plt.subplots(
    figsize=(11.5, 6.4)
)

positions = np.arange(
    len(SCENARIOS)
)

random_generator = np.random.default_rng(
    seed=42
)

runtime_legend_added = {
    "node": False,
    "bun": False
}

mean_legend_added = False


# ---------------------------------------------------------
# RISANJE MERITEV
# ---------------------------------------------------------

for scenario_index, scenario in enumerate(
    SCENARIOS
):
    for runtime in RUNTIMES:
        selected_data = data[
            (data["scenario"] == scenario)
            & (data["runtime"] == runtime)
        ].sort_values("repetition")

        values = selected_data[
            "latency_avg_ms"
        ].to_numpy()

        mean, confidence_interval = (
            calculate_statistics(values)
        )

        center_position = (
            positions[scenario_index]
            + RUNTIME_OFFSETS[runtime]
        )

        jitter = random_generator.normal(
            loc=center_position,
            scale=0.035,
            size=len(values)
        )

        runtime_label = None

        if not runtime_legend_added[runtime]:
            runtime_label = (
                RUNTIME_LABELS[runtime]
            )

            runtime_legend_added[runtime] = True

        axis.scatter(
            jitter,
            values,
            color=RUNTIME_COLORS[runtime],
            marker=RUNTIME_MARKERS[runtime],
            edgecolor="black",
            linewidth=0.3,
            alpha=0.60,
            s=34,
            label=runtime_label,
            zorder=2
        )

        mean_label = None

        if not mean_legend_added:
            mean_label = (
                "Povprečje ± 95 % IZ"
            )

            mean_legend_added = True

        axis.errorbar(
            center_position,
            mean,
            yerr=confidence_interval,
            fmt="D",
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=6,
            capsize=6,
            elinewidth=1.2,
            capthick=1.2,
            label=mean_label,
            zorder=4
        )

        formatted_mean = (
            f"{mean:.2f} ms"
            .replace(".", ",")
        )

        axis.text(
            center_position,
            mean + confidence_interval + 5,
            formatted_mean,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            zorder=5
        )


# ---------------------------------------------------------
# NASLOV IN OSI
# ---------------------------------------------------------

axis.set_title(
    "Primerjava povprečne latence pri "
    "izbranih testnih scenarijih",
    fontsize=13,
    fontweight="bold",
    pad=8
)

axis.set_xlabel(
    "Testni scenarij",
    fontsize=10,
    fontweight="bold"
)

axis.set_ylabel(
    "Povprečna latenca posamezne "
    "ponovitve (ms)",
    fontsize=10,
    fontweight="bold"
)

axis.set_xticks(
    positions
)

axis.set_xticklabels([
    f"/{scenario}"
    for scenario in SCENARIOS
])

axis.tick_params(
    axis="both",
    labelsize=9
)

axis.set_ylim(
    50,
    445
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

axis.legend(
    title="Izvajalno okolje",
    loc="upper right",
    fontsize=9,
    title_fontsize=9,
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

print(
    f"Graf je shranjen v: {OUTPUT_FILE}"
)

plt.show()
plt.close(figure)