import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    ROOT_DIR
    / "results"
    / "load-tests-20260904-052953"
)

SUMMARY_FILE = (
    RESULTS_DIR
    / "summary-runs.csv"
)

DOCKER_STATS_DIR = (
    RESULTS_DIR
    / "docker-stats"
)

OUTPUT_FILE = (
    ROOT_DIR
    / "figures"
    / "cpu-time-relative-reduction.png"
)

SCENARIOS = [
    "simple",
    "compute",
    "file-read",
    "file-write",
    "json",
    "auth"
]

RUNTIMES = ["node", "bun"]

EXPECTED_REPETITIONS = 25
SAMPLES_PER_REPETITION = 13

FILE_PATTERN = re.compile(
    r"(?P<scenario>.+)_(?P<runtime>node|bun)_run"
    r"(?P<repetition>\d+)\.csv"
)


def load_summary() -> pd.DataFrame:
    if not SUMMARY_FILE.exists():
        raise FileNotFoundError(
            f"Datoteka ne obstaja: {SUMMARY_FILE}"
        )

    data = pd.read_csv(
        SUMMARY_FILE,
        decimal=",",
        encoding="utf-8-sig"
    )

    required_columns = {
        "scenario",
        "runtime",
        "repetition",
        "duration_seconds",
        "responses_2xx"
    }

    missing_columns = required_columns - set(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "V summary-runs.csv manjkajo stolpci: "
            + ", ".join(sorted(missing_columns))
        )

    data["scenario"] = (
        data["scenario"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.removeprefix("/")
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

    numeric_columns = [
        "repetition",
        "duration_seconds",
        "responses_2xx"
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="raise"
        )

    return data


def load_cpu_measurements() -> pd.DataFrame:
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

        if "cpu_percent" not in data.columns:
            raise ValueError(
                f"V datoteki {file_path.name} manjka "
                "stolpec cpu_percent."
            )

        if len(data) < SAMPLES_PER_REPETITION:
            raise ValueError(
                f"Datoteka {file_path.name} vsebuje "
                f"samo {len(data)} vzorcev."
            )

        selected_samples = data.iloc[
            :SAMPLES_PER_REPETITION
        ].copy()

        selected_samples["cpu_percent"] = (
            pd.to_numeric(
                selected_samples["cpu_percent"],
                errors="raise"
            )
        )

        average_cpu_percent = float(
            selected_samples["cpu_percent"].mean()
        )

        results.append({
            "scenario": scenario,
            "runtime": runtime,
            "repetition": repetition,
            "cpu_avg_percent_13_samples":
                average_cpu_percent
        })

    if not results:
        raise ValueError(
            "V mapi docker-stats ni bilo najdenih "
            "ustreznih datotek."
        )

    return pd.DataFrame(results)


def combine_measurements(
    summary: pd.DataFrame,
    cpu_measurements: pd.DataFrame
) -> pd.DataFrame:
    combined = cpu_measurements.merge(
        summary[
            [
                "scenario",
                "runtime",
                "repetition",
                "duration_seconds",
                "responses_2xx"
            ]
        ],
        on=[
            "scenario",
            "runtime",
            "repetition"
        ],
        how="inner",
        validate="one_to_one"
    )

    if len(combined) != len(cpu_measurements):
        raise ValueError(
            "Vseh meritev CPU ni bilo mogoče povezati "
            "s summary-runs.csv."
        )

    if (combined["responses_2xx"] <= 0).any():
        raise ValueError(
            "Število uspešnih zahtev mora biti večje "
            "od nič."
        )

    return combined


def verify_measurements(
    data: pd.DataFrame
) -> None:
    for scenario in SCENARIOS:
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
                selected["repetition"]
                .astype(int)
                .tolist()
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


def calculate_normalized_cpu_time(
    data: pd.DataFrame
) -> pd.DataFrame:
    calculated = data.copy()

    calculated["normalized_cpu_time_per_1000"] = (
        (
            calculated[
                "cpu_avg_percent_13_samples"
            ] / 100
        )
        * calculated["duration_seconds"]
        / calculated["responses_2xx"]
        * 1000
    )

    return calculated


def calculate_relative_reduction(
    data: pd.DataFrame
) -> pd.DataFrame:
    means = (
        data.groupby(
            ["scenario", "runtime"]
        )["normalized_cpu_time_per_1000"]
        .mean()
        .unstack()
    )

    missing_runtimes = set(RUNTIMES) - set(
        means.columns
    )

    if missing_runtimes:
        raise ValueError(
            "Manjkajo rezultati za izvajalna okolja: "
            + ", ".join(sorted(missing_runtimes))
        )

    means["relative_reduction_percent"] = (
        (
            means["node"] - means["bun"]
        )
        / means["node"]
        * 100
    )

    return means


def format_percentage(value: float) -> str:
    return f"{value:.1f} %".replace(".", ",")


def print_results(
    results: pd.DataFrame
) -> None:
    print(
        "\nRelativno zmanjšanje normaliziranega "
        "procesorskega časa pri Bun.js:"
    )

    for scenario in SCENARIOS:
        node_value = results.loc[
            scenario,
            "node"
        ]

        bun_value = results.loc[
            scenario,
            "bun"
        ]

        reduction = results.loc[
            scenario,
            "relative_reduction_percent"
        ]

        print(
            f"/{scenario}: "
            f"Node.js = {node_value:.6f}, "
            f"Bun.js = {bun_value:.6f}, "
            f"zmanjšanje = {reduction:.2f} %"
        )


def create_graph(
    results: pd.DataFrame
) -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    labels = [
        f"/{scenario}"
        for scenario in SCENARIOS
    ]

    reductions = [
        results.loc[
            scenario,
            "relative_reduction_percent"
        ]
        for scenario in SCENARIOS
    ]

    plt.style.use("seaborn-v0_8-whitegrid")

    figure, axis = plt.subplots(
        figsize=(11.5, 6.5)
    )

    bars = axis.bar(
        labels,
        reductions,
        color="#2878b5",
        width=0.8
    )

    axis.set_title(
        "Relativno zmanjšanje normaliziranega "
        "procesorskega časa pri Bun.js",
        fontsize=15,
        fontweight="bold",
        pad=12
    )

    axis.set_xlabel(
        "Testni scenariji",
        fontsize=11,
        fontweight="bold"
    )

    axis.set_ylabel(
        "Zmanjšanje glede na Node.js (%)",
        fontsize=11,
        fontweight="bold"
    )

    axis.tick_params(
        axis="both",
        labelsize=10
    )

    axis.grid(
        axis="y",
        linestyle="-",
        linewidth=0.7,
        alpha=0.30
    )

    axis.grid(
        axis="x",
        visible=False
    )

    maximum_value = max(reductions)

    axis.set_ylim(
        0,
        maximum_value + 4
    )

    for bar, value in zip(
        bars,
        reductions
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.6,
            format_percentage(value),
            ha="center",
            va="bottom",
            fontsize=11
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

    print(f"\nGraf je shranjen v: {OUTPUT_FILE}")

    plt.show()


def main() -> None:
    summary = load_summary()
    cpu_measurements = load_cpu_measurements()

    combined = combine_measurements(
        summary=summary,
        cpu_measurements=cpu_measurements
    )

    verify_measurements(combined)

    calculated = calculate_normalized_cpu_time(
        combined
    )

    results = calculate_relative_reduction(
        calculated
    )

    print_results(results)
    create_graph(results)


if __name__ == "__main__":
    main()