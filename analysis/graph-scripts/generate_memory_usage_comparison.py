import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats


ROOT_DIR = Path(__file__).resolve().parents[2]

DOCKER_STATS_DIR = (
    ROOT_DIR
    / "results"
    / "load-tests-20260904-052953"
    / "docker-stats"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "figures"
    / "memory-usage-comparison.png"
)

SCENARIOS_WITHOUT_FILE_READ = [
    "simple",
    "compute",
    "file-write",
    "json",
    "auth"
]

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
SAMPLES_PER_REPETITION = 13

FILE_PATTERN = re.compile(
    r"(?P<scenario>.+)_(?P<runtime>node|bun)_run"
    r"(?P<repetition>\d+)\.csv"
)


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


def load_memory_measurements() -> pd.DataFrame:
    if not DOCKER_STATS_DIR.exists():
        raise FileNotFoundError(
            f"Mapa ne obstaja: {DOCKER_STATS_DIR}"
        )

    results = []

    for file_path in sorted(
        DOCKER_STATS_DIR.glob("*.csv")
    ):
        match = FILE_PATTERN.fullmatch(
            file_path.name
        )

        if match is None:
            continue

        scenario = match.group("scenario")
        runtime = match.group("runtime")
        repetition = int(
            match.group("repetition")
        )

        data = pd.read_csv(
            file_path,
            decimal=",",
            encoding="utf-8-sig"
        )

        if "memory_mib" not in data.columns:
            raise ValueError(
                f"V datoteki {file_path.name} manjka "
                "stolpec memory_mib."
            )

        if len(data) < SAMPLES_PER_REPETITION:
            raise ValueError(
                f"Datoteka {file_path.name} vsebuje "
                f"samo {len(data)} vzorcev."
            )

        selected_samples = data.iloc[
            :SAMPLES_PER_REPETITION
        ].copy()

        selected_samples["memory_mib"] = (
            pd.to_numeric(
                selected_samples["memory_mib"],
                errors="raise"
            )
        )

        mean_memory = float(
            selected_samples["memory_mib"].mean()
        )

        results.append({
            "scenario": scenario,
            "runtime": runtime,
            "repetition": repetition,
            "memory_mib": mean_memory
        })

    if not results:
        raise ValueError(
            "V mapi docker-stats ni bilo najdenih "
            "ustreznih datotek."
        )

    return pd.DataFrame(results)


def verify_measurements(
    data: pd.DataFrame
) -> None:
    all_scenarios = (
        SCENARIOS_WITHOUT_FILE_READ
        + ["file-read"]
    )

    for scenario in all_scenarios:
        for runtime in RUNTIMES:
            selected = data[
                (data["scenario"] == scenario)
                & (data["runtime"] == runtime)
            ]

            count = len(selected)

            if count != EXPECTED_REPETITIONS:
                raise ValueError(
                    f"{scenario}, {runtime}: najdenih "
                    f"{count} ponovitev, pričakovanih pa "
                    f"{EXPECTED_REPETITIONS}."
                )

            repetitions = sorted(
                selected["repetition"].tolist()
            )

            expected_repetitions = list(
                range(
                    1,
                    EXPECTED_REPETITIONS + 1
                )
            )

            if repetitions != expected_repetitions:
                raise ValueError(
                    f"{scenario}, {runtime}: številke "
                    "ponovitev niso popolne."
                )


def plot_runtime_values(
    axis,
    values: np.ndarray,
    x_position: float,
    runtime: str,
    random_generator: np.random.Generator
) -> None:
    jitter = random_generator.uniform(
        -0.055,
        0.055,
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
        s=42,
        alpha=0.55,
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
        markersize=9,
        color=COLORS[runtime],
        markeredgecolor="white",
        markeredgewidth=0.9,
        capsize=7,
        elinewidth=2,
        zorder=4
    )


def configure_axis(axis) -> None:
    axis.grid(
        axis="y",
        linestyle="-",
        linewidth=0.7,
        alpha=0.35
    )

    axis.grid(
        axis="x",
        visible=False
    )

    axis.tick_params(
        axis="both",
        labelsize=10
    )

    for spine in axis.spines.values():
        spine.set_color("#6f7f8f")
        spine.set_linewidth(0.8)

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def create_graph(
    data: pd.DataFrame
) -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(13, 10.5),
        gridspec_kw={
            "height_ratios": [1.55, 1],
            "hspace": 0.20
        }
    )

    top_axis = axes[0]
    bottom_axis = axes[1]

    random_generator = np.random.default_rng(42)

    offsets = {
        "node": -0.18,
        "bun": 0.18
    }

    scenario_positions = np.arange(
        len(SCENARIOS_WITHOUT_FILE_READ)
    )

    for scenario_index, scenario in enumerate(
        SCENARIOS_WITHOUT_FILE_READ
    ):
        for runtime in RUNTIMES:
            values = data.loc[
                (data["scenario"] == scenario)
                & (data["runtime"] == runtime),
                "memory_mib"
            ].to_numpy(dtype=float)

            x_position = (
                scenario_positions[scenario_index]
                + offsets[runtime]
            )

            plot_runtime_values(
                axis=top_axis,
                values=values,
                x_position=x_position,
                runtime=runtime,
                random_generator=random_generator
            )

    top_axis.set_title(
        "Scenariji brez /file-read",
        fontsize=13,
        fontweight="bold",
        pad=10
    )

    top_axis.set_ylabel(
        "Povprečna poraba delovnega "
        "pomnilnika (MiB)",
        fontsize=11,
        fontweight="bold"
    )

    top_axis.set_xticks(
        scenario_positions
    )

    top_axis.set_xticklabels(
        [
            f"/{scenario}"
            for scenario
            in SCENARIOS_WITHOUT_FILE_READ
        ]
    )

    top_axis.set_ylim(22, 99)

    configure_axis(top_axis)

    file_read_position = 0

    for runtime in RUNTIMES:
        values = data.loc[
            (data["scenario"] == "file-read")
            & (data["runtime"] == runtime),
            "memory_mib"
        ].to_numpy(dtype=float)

        x_position = (
            file_read_position
            + offsets[runtime]
        )

        plot_runtime_values(
            axis=bottom_axis,
            values=values,
            x_position=x_position,
            runtime=runtime,
            random_generator=random_generator
        )

    bottom_axis.set_title(
        "Scenarij /file-read na ločeni lestvici",
        fontsize=13,
        fontweight="bold",
        pad=10
    )

    bottom_axis.set_xlabel(
        "Izvajalno okolje",
        fontsize=11,
        fontweight="bold"
    )

    bottom_axis.set_ylabel(
        "Povprečna poraba delovnega "
        "pomnilnika (MiB)",
        fontsize=11,
        fontweight="bold"
    )

    bottom_axis.set_xticks([
        offsets["node"],
        offsets["bun"]
    ])

    bottom_axis.set_xticklabels([
        "Node.js",
        "Bun.js"
    ])

    bottom_axis.set_xlim(-0.55, 0.55)
    bottom_axis.set_ylim(70, 295)

    configure_axis(bottom_axis)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLORS["node"],
            markeredgecolor="black",
            markersize=8,
            label="Node.js"
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=COLORS["bun"],
            markeredgecolor="black",
            markersize=9,
            label="Bun.js"
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="black",
            markerfacecolor="black",
            markersize=8,
            label="Povprečje ± 95 % IZ"
        )
    ]

    top_axis.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=10,
        frameon=False
    )

    figure.suptitle(
        "Poraba delovnega pomnilnika "
        "po testnih scenarijih ($n = 25$)",
        fontsize=16,
        fontweight="bold",
        y=0.985
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    figure.savefig(
        OUTPUT_FILE,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Graf je shranjen v: {OUTPUT_FILE}")

    plt.show()


def main() -> None:
    data = load_memory_measurements()
    verify_measurements(data)
    create_graph(data)


if __name__ == "__main__":
    main()