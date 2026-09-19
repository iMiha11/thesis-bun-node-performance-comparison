from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats


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
    / "simple-throughput-comparison.png"
)

SCENARIO = "/simple"
RUNTIMES = ["node", "bun"]

COLORS = {
    "node": "#2ca02c",
    "bun": "#ff7f0e"
}

MARKERS = {
    "node": "o",
    "bun": "^"
}

EXPECTED_REPETITIONS = 25


def confidence_interval(
    values: np.ndarray
) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    standard_error = stats.sem(values)

    margin = stats.t.ppf(
        0.975,
        df=len(values) - 1
    ) * standard_error

    return mean, mean - margin, mean + margin


def format_number(value: float) -> str:
    formatted = f"{value:,.2f}"

    return (
        formatted
        .replace(",", "TEMP")
        .replace(".", ",")
        .replace("TEMP", ".")
    )


def load_data() -> pd.DataFrame:
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
        "requests_avg_sec"
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "Manjkajo stolpci: "
            + ", ".join(sorted(missing_columns))
        )

    data["runtime"] = (
        data["runtime"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "node.js": "node",
            "nodejs": "node",
            "node-server": "node",
            "bun.js": "bun",
            "bunjs": "bun",
            "bun-server": "bun"
        })
    )

    data["scenario"] = (
        data["scenario"]
        .astype(str)
        .str.strip()
        .str.lower()
        .apply(
            lambda value: value
            if value.startswith("/")
            else f"/{value}"
        )
    )

    data["requests_avg_sec"] = pd.to_numeric(
        data["requests_avg_sec"],
        errors="raise"
    )

    return data


def verify_measurements(data: pd.DataFrame) -> None:
    print(
        "Najdena izvajalna okolja:",
        sorted(data["runtime"].unique())
    )

    print(
        "Najdeni scenariji:",
        sorted(data["scenario"].unique())
    )

    for runtime in RUNTIMES:
        count = len(
            data[
                (data["scenario"] == SCENARIO)
                & (data["runtime"] == runtime)
            ]
        )

        if count != EXPECTED_REPETITIONS:
            raise ValueError(
                f"{SCENARIO}, {runtime}: najdenih "
                f"{count} meritev, pričakovanih pa "
                f"{EXPECTED_REPETITIONS}."
            )


def create_graph(data: pd.DataFrame) -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axis = plt.subplots(
        figsize=(11.5, 6.4)
    )

    random_generator = np.random.default_rng(42)

    positions = {
        "node": 0,
        "bun": 1
    }

    for runtime in RUNTIMES:
        values = data.loc[
            (data["scenario"] == SCENARIO)
            & (data["runtime"] == runtime),
            "requests_avg_sec"
        ].to_numpy(dtype=float)

        x_position = positions[runtime]

        jitter = random_generator.uniform(
            -0.06,
            0.06,
            size=len(values)
        )

        axis.scatter(
            np.full(
                len(values),
                x_position
            ) + jitter,
            values,
            color=COLORS[runtime],
            marker=MARKERS[runtime],
            s=40,
            alpha=0.60,
            edgecolor="black",
            linewidth=0.3,
            zorder=2
        )

        mean, lower, upper = confidence_interval(
            values
        )

        axis.errorbar(
            x_position,
            mean,
            yerr=[
                [mean - lower],
                [upper - mean]
            ],
            fmt="D",
            markersize=10,
            color=COLORS[runtime],
            markeredgecolor="black",
            markeredgewidth=1,
            capsize=7,
            elinewidth=2,
            zorder=4
        )

        axis.text(
            x_position,
            upper + 90,
            format_number(mean),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    axis.set_title(
        "Primerjava povprečne prepustnosti "
        "pri scenariju /simple",
        fontsize=13,
        fontweight="bold",
        pad=12
    )

    axis.set_xlabel(
        "Izvajalno okolje",
        fontsize=10,
        fontweight="bold"
    )

    axis.set_ylabel(
        "Povprečna prepustnost posamezne "
        "ponovitve (zahtev/s)",
        fontsize=10,
        fontweight="bold"
    )

    axis.set_xticks([0, 1])

    axis.set_xticklabels(
        ["Node.js", "Bun.js"],
        fontsize=10
    )

    axis.tick_params(
        axis="y",
        labelsize=9
    )

    axis.set_ylim(18350, 22950)

    axis.set_yticks([
        19000,
        20000,
        21000,
        22000
    ])

    axis.grid(
        axis="y",
        linestyle="--",
        alpha=0.30
    )

    axis.grid(
        axis="x",
        visible=False
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["node"],
            markeredgecolor="black",
            markersize=7,
            label="Node.js"
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=COLORS["bun"],
            markeredgecolor="black",
            markersize=8,
            label="Bun.js"
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="black",
            markerfacecolor="black",
            markersize=7,
            label="Povprečje ± 95 % IZ"
        )
    ]

    axis.legend(
        handles=legend_elements,
        title="Izvajalno okolje",
        loc="lower right",
        fontsize=9,
        title_fontsize=9
    )

    for spine in axis.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.8)

    figure.tight_layout()

    figure.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Graf je shranjen v: {OUTPUT_FILE}")

    plt.show()


def main() -> None:
    data = load_data()
    verify_measurements(data)
    create_graph(data)


if __name__ == "__main__":
    main()