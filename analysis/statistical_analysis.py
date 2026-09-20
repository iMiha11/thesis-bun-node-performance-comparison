#!/usr/bin/env python3
"""Ponovljiva statistična analiza meritev iz diplomske naloge.

Skripta reproducira opisno in sklepno statistiko za:

* čas zagona aplikacije (100 parov),
* latenco in prepustnost (25 parov na scenarij),
* ocenjeni procesorski čas na 1000 uspešnih zahtev,
* povprečno porabo delovnega pomnilnika.

Privzeto je namenjena lokaciji ``analysis/statistical_analysis.py`` v korenu
repozitorija. Poti je mogoče spremeniti z argumenti ukazne vrstice.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats


SCENARIOS = [
    "simple",
    "compute",
    "file-read",
    "file-write",
    "json",
    "auth",
]
RUNTIMES = ["node", "bun"]
EXPECTED_LOAD_REPETITIONS = 25
EXPECTED_STARTUP_REPETITIONS = 100
RESOURCE_SAMPLES_PER_RUN = 13
ALPHA = 0.05

RESOURCE_FILE_PATTERN = re.compile(
    r"(?P<scenario>.+)_(?P<runtime>node|bun)_run(?P<repetition>\d+)\.csv"
)


def parse_arguments() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    default_root = script_path.parent.parent

    parser = argparse.ArgumentParser(
        description="Izvede statistično analizo meritev Node.js in Bun.js."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=default_root,
        help="Koren repozitorija (privzeto: nadrejena mapa mape analysis).",
    )
    parser.add_argument(
        "--load-results-dir",
        type=Path,
        default=None,
        help=(
            "Mapa rezultatov obremenitvenih testov. Privzeto: "
            "<repo-root>/results/load-tests-20260904-052953"
        ),
    )
    parser.add_argument(
        "--startup-file",
        type=Path,
        default=None,
        help=(
            "CSV meritev časa zagona. Privzeto: "
            "<repo-root>/startup-benchmark/results/startup_measurements.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Izhodna mapa. Privzeto: <repo-root>/analysis/output",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    repo_root = args.repo_root.resolve()
    load_results_dir = (
        args.load_results_dir.resolve()
        if args.load_results_dir
        else repo_root / "results" / "load-tests-20260904-052953"
    )
    startup_file = (
        args.startup_file.resolve()
        if args.startup_file
        else repo_root
        / "startup-benchmark"
        / "results"
        / "startup_measurements.csv"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else repo_root / "analysis" / "output"
    )
    return repo_root, load_results_dir, startup_file, output_dir


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Datoteka ne obstaja: {path}")


def require_directory(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Mapa ne obstaja: {path}")


def sample_sd(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.std(array, ddof=1))


def mean_ci95(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    mean = float(np.mean(array))
    standard_error = float(stats.sem(array))
    critical_value = float(stats.t.ppf(0.975, len(array) - 1))
    margin = critical_value * standard_error
    return mean - margin, mean + margin


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Vrne Holmovo prilagoditev p-vrednosti v prvotnem vrstnem redu."""

    values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.empty(len(values), dtype=float)
    running_maximum = 0.0

    for rank, original_index in enumerate(order):
        candidate = (len(values) - rank) * values[original_index]
        running_maximum = max(running_maximum, candidate)
        adjusted_sorted[rank] = min(running_maximum, 1.0)

    adjusted = np.empty(len(values), dtype=float)
    for rank, original_index in enumerate(order):
        adjusted[original_index] = adjusted_sorted[rank]
    return adjusted


def paired_statistics(node: pd.Series, bun: pd.Series) -> dict[str, float]:
    node_values = node.to_numpy(dtype=float)
    bun_values = bun.to_numpy(dtype=float)
    differences = bun_values - node_values

    if len(node_values) != len(bun_values):
        raise ValueError("Število meritev Node.js in Bun.js ni enako.")
    if np.allclose(differences, 0.0):
        wilcoxon_w = 0.0
        p_value = 1.0
    else:
        result = stats.wilcoxon(
            bun_values,
            node_values,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )
        wilcoxon_w = float(result.statistic)
        p_value = float(result.pvalue)

    node_ci_lower, node_ci_upper = mean_ci95(node_values)
    bun_ci_lower, bun_ci_upper = mean_ci95(bun_values)
    difference_ci_lower, difference_ci_upper = mean_ci95(differences)
    node_mean = float(np.mean(node_values))
    bun_mean = float(np.mean(bun_values))

    return {
        "n_pairs": len(node_values),
        "node_mean": node_mean,
        "node_sd": sample_sd(node_values),
        "node_ci95_lower": node_ci_lower,
        "node_ci95_upper": node_ci_upper,
        "bun_mean": bun_mean,
        "bun_sd": sample_sd(bun_values),
        "bun_ci95_lower": bun_ci_lower,
        "bun_ci95_upper": bun_ci_upper,
        "mean_difference_bun_minus_node": float(np.mean(differences)),
        "relative_change_percent": (bun_mean - node_mean) / node_mean * 100.0,
        "difference_ci95_lower": difference_ci_lower,
        "difference_ci95_upper": difference_ci_upper,
        "wilcoxon_w": wilcoxon_w,
        "p_value": p_value,
        "bun_lower_pairs": int(np.sum(bun_values < node_values)),
        "bun_higher_pairs": int(np.sum(bun_values > node_values)),
        "equal_pairs": int(np.sum(bun_values == node_values)),
    }


def validate_pairs(
    data: pd.DataFrame,
    expected_repetitions: int,
    source_name: str,
) -> None:
    expected = set(range(1, expected_repetitions + 1))

    for scenario in SCENARIOS:
        for runtime in RUNTIMES:
            subset = data[
                (data["scenario"] == scenario) & (data["runtime"] == runtime)
            ]
            repetitions = set(subset["repetition"].astype(int))
            if repetitions != expected or len(subset) != expected_repetitions:
                raise ValueError(
                    f"{source_name}: {scenario}, {runtime}: pričakovanih "
                    f"{expected_repetitions} ponovitev, najdenih {len(subset)}."
                )


def read_startup_measurements(path: Path) -> pd.DataFrame:
    require_file(path)
    data = pd.read_csv(path, encoding="utf-8-sig")
    required_columns = {
        "repetition",
        "node_startup_ms",
        "bun_startup_ms",
    }
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"V {path} manjkajo stolpci: {sorted(missing)}")

    expected = set(range(1, EXPECTED_STARTUP_REPETITIONS + 1))
    repetitions = set(data["repetition"].astype(int))
    if len(data) != EXPECTED_STARTUP_REPETITIONS or repetitions != expected:
        raise ValueError(
            f"Čas zagona: pričakovanih {EXPECTED_STARTUP_REPETITIONS} "
            f"ponovitev, najdenih {len(data)}."
        )
    return data.sort_values("repetition").reset_index(drop=True)


def read_summary(path: Path) -> pd.DataFrame:
    require_file(path)
    data = pd.read_csv(path, encoding="utf-8-sig", decimal=",")
    required_columns = {
        "scenario",
        "runtime",
        "repetition",
        "latency_avg_ms",
        "latency_p50_ms",
        "latency_p99_ms",
        "requests_avg_sec",
        "responses_2xx",
        "responses_non2xx",
        "errors",
        "timeouts",
    }
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"V {path} manjkajo stolpci: {sorted(missing)}")

    data["scenario"] = data["scenario"].astype(str).str.strip().str.lower()
    data["runtime"] = data["runtime"].astype(str).str.strip().str.lower()
    data["repetition"] = data["repetition"].astype(int)
    validate_pairs(data, EXPECTED_LOAD_REPETITIONS, path.name)

    if int(data[["responses_non2xx", "errors", "timeouts"]].to_numpy().sum()) != 0:
        raise ValueError("V povzetku so bili najdeni neuspešni odzivi, napake ali časovne prekoračitve.")
    return data


def read_resource_measurements(load_results_dir: Path) -> pd.DataFrame:
    docker_stats_dir = load_results_dir / "docker-stats"
    autocannon_dir = load_results_dir / "autocannon"
    require_directory(docker_stats_dir)
    require_directory(autocannon_dir)

    rows: list[dict[str, float | int | str]] = []
    for csv_path in sorted(docker_stats_dir.glob("*.csv")):
        match = RESOURCE_FILE_PATTERN.fullmatch(csv_path.name)
        if not match:
            continue

        scenario = match.group("scenario")
        runtime = match.group("runtime")
        repetition = int(match.group("repetition"))
        if scenario not in SCENARIOS:
            continue

        samples = pd.read_csv(csv_path, encoding="utf-8-sig", decimal=",")
        required_columns = {"cpu_percent", "memory_mib"}
        missing = required_columns - set(samples.columns)
        if missing:
            raise ValueError(f"V {csv_path} manjkajo stolpci: {sorted(missing)}")
        if len(samples) < RESOURCE_SAMPLES_PER_RUN:
            raise ValueError(
                f"{csv_path}: pričakovanih najmanj {RESOURCE_SAMPLES_PER_RUN} "
                f"vzorcev, najdenih {len(samples)}."
            )

        selected = samples.iloc[:RESOURCE_SAMPLES_PER_RUN]
        cpu_mean_percent = float(selected["cpu_percent"].mean())
        memory_mean_mib = float(selected["memory_mib"].mean())

        autocannon_path = autocannon_dir / csv_path.with_suffix(".json").name
        require_file(autocannon_path)
        with autocannon_path.open(encoding="utf-8-sig") as file:
            autocannon = json.load(file)

        duration_seconds = float(autocannon["duration"])
        successful_requests = int(autocannon["2xx"])
        non_2xx = int(autocannon.get("non2xx", 0))
        errors = int(autocannon.get("errors", 0))
        timeouts = int(autocannon.get("timeouts", 0))
        if non_2xx or errors or timeouts:
            raise ValueError(
                f"{autocannon_path}: najdeni so neuspešni odzivi, napake ali "
                "časovne prekoračitve."
            )
        if successful_requests <= 0:
            raise ValueError(f"{autocannon_path}: število odzivov 2xx mora biti pozitivno.")

        normalized_cpu_seconds = (
            (cpu_mean_percent / 100.0)
            * duration_seconds
            * (1000.0 / successful_requests)
        )
        rows.append(
            {
                "scenario": scenario,
                "runtime": runtime,
                "repetition": repetition,
                "resource_samples_used": RESOURCE_SAMPLES_PER_RUN,
                "cpu_mean_percent": cpu_mean_percent,
                "memory_mean_mib": memory_mean_mib,
                "duration_seconds": duration_seconds,
                "successful_requests_2xx": successful_requests,
                "normalized_cpu_seconds_per_1000_requests": normalized_cpu_seconds,
            }
        )

    data = pd.DataFrame(rows)
    validate_pairs(data, EXPECTED_LOAD_REPETITIONS, "docker-stats/autocannon")
    return data.sort_values(["scenario", "repetition", "runtime"]).reset_index(drop=True)


def verify_cross_source_counts(summary: pd.DataFrame, resources: pd.DataFrame) -> None:
    merged = summary.merge(
        resources,
        on=["scenario", "runtime", "repetition"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        raise ValueError("Ponovitve v summary-runs.csv in surovih datotekah se ne ujemajo.")

    summary_counts = merged["responses_2xx"].astype(int)
    raw_counts = merged["successful_requests_2xx"].astype(int)
    if not summary_counts.equals(raw_counts):
        raise ValueError("Število odzivov 2xx v CSV-povzetku in Autocannon JSON se ne ujema.")


def paired_series(
    data: pd.DataFrame,
    scenario: str,
    value_column: str,
) -> tuple[pd.Series, pd.Series]:
    pivot = data[data["scenario"] == scenario].pivot(
        index="repetition",
        columns="runtime",
        values=value_column,
    )
    pivot = pivot.sort_index()
    if pivot.isna().any().any() or set(pivot.columns) != set(RUNTIMES):
        raise ValueError(f"Neveljavni pari za {scenario}, metrika {value_column}.")
    return pivot["node"], pivot["bun"]


def startup_analysis(data: pd.DataFrame) -> pd.DataFrame:
    node = data["node_startup_ms"].astype(float)
    bun = data["bun_startup_ms"].astype(float)
    comparison = paired_statistics(node, bun)

    row: dict[str, float | int] = {
        "n_pairs": len(data),
        "node_mean_ms": float(node.mean()),
        "node_median_ms": float(node.median()),
        "node_sd_ms": sample_sd(node),
        "node_min_ms": float(node.min()),
        "node_max_ms": float(node.max()),
        "node_p95_ms": float(np.percentile(node, 95)),
        "node_p99_ms": float(np.percentile(node, 99)),
        "bun_mean_ms": float(bun.mean()),
        "bun_median_ms": float(bun.median()),
        "bun_sd_ms": sample_sd(bun),
        "bun_min_ms": float(bun.min()),
        "bun_max_ms": float(bun.max()),
        "bun_p95_ms": float(np.percentile(bun, 95)),
        "bun_p99_ms": float(np.percentile(bun, 99)),
        "mean_difference_bun_minus_node_ms": comparison[
            "mean_difference_bun_minus_node"
        ],
        "relative_change_percent": comparison["relative_change_percent"],
        "difference_ci95_lower_ms": comparison["difference_ci95_lower"],
        "difference_ci95_upper_ms": comparison["difference_ci95_upper"],
        "wilcoxon_w": comparison["wilcoxon_w"],
        "p_value": comparison["p_value"],
        "bun_faster_pairs": comparison["bun_lower_pairs"],
    }
    return pd.DataFrame([row])


def latency_analysis(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        node, bun = paired_series(summary, scenario, "latency_avg_ms")
        row = {"scenario": f"/{scenario}", **paired_statistics(node, bun)}

        for runtime in RUNTIMES:
            subset = summary[
                (summary["scenario"] == scenario) & (summary["runtime"] == runtime)
            ]
            row[f"{runtime}_p50_mean_ms"] = float(subset["latency_p50_ms"].mean())
            row[f"{runtime}_p99_mean_ms"] = float(subset["latency_p99_ms"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def throughput_analysis(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        node, bun = paired_series(summary, scenario, "requests_avg_sec")
        rows.append(
            {
                "scenario": f"/{scenario}",
                **paired_statistics(node, bun),
            }
        )
    return pd.DataFrame(rows)


def successful_requests_analysis(resources: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        row: dict[str, str | int | float] = {"scenario": f"/{scenario}"}
        for runtime in RUNTIMES:
            values = resources[
                (resources["scenario"] == scenario)
                & (resources["runtime"] == runtime)
            ]["successful_requests_2xx"]
            row[f"{runtime}_mean_2xx"] = float(values.mean())
            row[f"{runtime}_sd_2xx"] = sample_sd(values)
        rows.append(row)
    return pd.DataFrame(rows)


def cpu_analysis(resources: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metric = "normalized_cpu_seconds_per_1000_requests"
    for scenario in SCENARIOS:
        node, bun = paired_series(resources, scenario, metric)
        row = {"scenario": f"/{scenario}", **paired_statistics(node, bun)}

        for runtime in RUNTIMES:
            values = resources[
                (resources["scenario"] == scenario)
                & (resources["runtime"] == runtime)
            ]["cpu_mean_percent"]
            row[f"{runtime}_raw_cpu_mean_percent"] = float(values.mean())
            row[f"{runtime}_raw_cpu_sd_percent"] = sample_sd(values)
        rows.append(row)

    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_value"])
    return result


def memory_analysis(resources: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario in SCENARIOS:
        node, bun = paired_series(resources, scenario, "memory_mean_mib")
        rows.append(
            {
                "scenario": f"/{scenario}",
                **paired_statistics(node, bun),
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_value"])
    return result


def apply_h2_holm(latency: pd.DataFrame, throughput: pd.DataFrame) -> None:
    latency_index = latency.index[latency["scenario"] == "/simple"][0]
    throughput_index = throughput.index[throughput["scenario"] == "/simple"][0]
    adjusted = holm_adjust(
        [
            latency.loc[latency_index, "p_value"],
            throughput.loc[throughput_index, "p_value"],
        ]
    )
    latency["p_holm_h2"] = np.nan
    throughput["p_holm_h2"] = np.nan
    latency.loc[latency_index, "p_holm_h2"] = adjusted[0]
    throughput.loc[throughput_index, "p_holm_h2"] = adjusted[1]


def sl_number(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def p_text(value: float) -> str:
    if value < 0.001:
        return "< 0,001"
    return sl_number(value, 3)


def row_for(data: pd.DataFrame, scenario: str) -> pd.Series:
    return data.loc[data["scenario"] == scenario].iloc[0]


def create_report(
    startup: pd.DataFrame,
    latency: pd.DataFrame,
    throughput: pd.DataFrame,
    cpu: pd.DataFrame,
    memory: pd.DataFrame,
) -> str:
    startup_row = startup.iloc[0]
    simple_latency = row_for(latency, "/simple")
    simple_throughput = row_for(throughput, "/simple")

    h1_confirmed = (
        startup_row["p_value"] < ALPHA
        and startup_row["mean_difference_bun_minus_node_ms"] < 0
    )
    h2_confirmed = (
        simple_latency["p_holm_h2"] < ALPHA
        and simple_throughput["p_holm_h2"] < ALPHA
        and simple_latency["mean_difference_bun_minus_node"] < 0
        and simple_throughput["mean_difference_bun_minus_node"] > 0
    )
    cpu_part_supported = bool(
        ((cpu["p_holm"] < ALPHA) & (cpu["mean_difference_bun_minus_node"] < 0)).all()
    )
    memory_part_supported = bool(
        (
            (memory["p_holm"] < ALPHA)
            & (memory["mean_difference_bun_minus_node"] < 0)
        ).all()
    )
    h3_confirmed = cpu_part_supported and memory_part_supported

    lines = [
        "STATISTIČNA ANALIZA MERITEV NODE.JS IN BUN.JS",
        "=" * 48,
        "",
        "Metodološka pravila",
        "- čas zagona: 100 parov meritev",
        "- obremenitveni testi: 25 parov na scenarij",
        "- viri: prvih 13 vzorcev docker stats na ponovitev",
        "- test: dvostranski Wilcoxonov test predznačenih rangov",
        "- raven statistične značilnosti: 0,05",
        "- H2: Holmova prilagoditev dveh primerjav za /simple",
        "- H3: ločeni Holmovi prilagoditvi šestih CPU in šestih RAM primerjav",
        "",
        "H1 – čas zagona",
        f"Node.js: {sl_number(startup_row['node_mean_ms'])} ms",
        f"Bun.js: {sl_number(startup_row['bun_mean_ms'])} ms",
        (
            "Povprečna parna razlika (Bun − Node): "
            f"{sl_number(startup_row['mean_difference_bun_minus_node_ms'])} ms"
        ),
        (
            "95-% IZ razlike: ["
            f"{sl_number(startup_row['difference_ci95_lower_ms'])}; "
            f"{sl_number(startup_row['difference_ci95_upper_ms'])}] ms"
        ),
        (
            f"W = {startup_row['wilcoxon_w']:.0f}; "
            f"p {p_text(startup_row['p_value'])}"
        ),
        f"Sklep H1: {'potrjena' if h1_confirmed else 'ni potrjena'}",
        "",
        "H2 – /simple",
        (
            "Latenca, Bun − Node: "
            f"{sl_number(simple_latency['mean_difference_bun_minus_node'], 3)} ms "
            f"({sl_number(simple_latency['relative_change_percent'])} %); "
            f"W = {simple_latency['wilcoxon_w']:.0f}; "
            f"pHolm = {p_text(simple_latency['p_holm_h2'])}"
        ),
        (
            "Prepustnost, Bun − Node: "
            f"{sl_number(simple_throughput['mean_difference_bun_minus_node'])} zahtev/s "
            f"({sl_number(simple_throughput['relative_change_percent'])} %); "
            f"W = {simple_throughput['wilcoxon_w']:.0f}; "
            f"pHolm = {p_text(simple_throughput['p_holm_h2'])}"
        ),
        f"Sklep H2: {'potrjena' if h2_confirmed else 'ni potrjena'}",
        "",
        "H3 – procesor in delovni pomnilnik",
        (
            "Procesorski del: "
            f"{'podprt' if cpu_part_supported else 'ni podprt'}"
        ),
        (
            "Pomnilniški del: "
            f"{'podprt' if memory_part_supported else 'ni podprt'}"
        ),
        f"Sklep H3: {'potrjena' if h3_confirmed else 'ni potrjena'}",
        "",
        "Podrobne vrednosti so zapisane v spremljajočih datotekah CSV.",
    ]
    return "\n".join(lines) + "\n"


def format_p_value_for_csv(value: float) -> str:
    if pd.isna(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def save_csv(
    data: pd.DataFrame,
    path: Path,
    default_decimals: int = 2,
    column_decimals: dict[str, int] | None = None,
) -> None:
    """Shrani pregleden CSV z zaokroženimi prikaznimi vrednostmi."""

    output = data.copy()
    decimals = column_decimals or {}

    for column in output.columns:
        if column in {"p_value", "p_holm", "p_holm_h2"}:
            output[column] = output[column].map(format_p_value_for_csv)
            continue

        if pd.api.types.is_float_dtype(output[column]):
            places = decimals.get(column, default_decimals)
            output[column] = output[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{places}f}"
            )

    output.to_csv(path, index=False, encoding="utf-8")


def main() -> None:
    args = parse_arguments()
    _, load_results_dir, startup_file, output_dir = resolve_paths(args)
    require_directory(load_results_dir)

    summary_file = load_results_dir / "summary-runs.csv"
    startup_measurements = read_startup_measurements(startup_file)
    summary = read_summary(summary_file)
    resources = read_resource_measurements(load_results_dir)
    verify_cross_source_counts(summary, resources)

    startup = startup_analysis(startup_measurements)
    latency = latency_analysis(summary)
    throughput = throughput_analysis(summary)
    apply_h2_holm(latency, throughput)
    successful_requests = successful_requests_analysis(resources)
    cpu = cpu_analysis(resources)
    memory = memory_analysis(resources)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(startup, output_dir / "startup-statistics.csv")
    save_csv(
        latency,
        output_dir / "latency-statistics.csv",
        default_decimals=2,
        column_decimals={"mean_difference_bun_minus_node": 3},
    )
    save_csv(throughput, output_dir / "throughput-statistics.csv")
    save_csv(
        cpu,
        output_dir / "cpu-statistics.csv",
        default_decimals=4,
        column_decimals={
            "relative_change_percent": 2,
            "wilcoxon_w": 0,
            "node_raw_cpu_mean_percent": 2,
            "node_raw_cpu_sd_percent": 2,
            "bun_raw_cpu_mean_percent": 2,
            "bun_raw_cpu_sd_percent": 2,
        },
    )
    save_csv(memory, output_dir / "memory-statistics.csv")
    save_csv(
        successful_requests,
        output_dir / "successful-requests.csv",
        default_decimals=1,
    )
    save_csv(
        resources,
        output_dir / "resource-measurements.csv",
        default_decimals=2,
        column_decimals={
            "normalized_cpu_seconds_per_1000_requests": 4,
        },
    )

    report = create_report(startup, latency, throughput, cpu, memory)
    report_path = output_dir / "analysis-report.txt"
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Rezultati so shranjeni v: {output_dir}")


if __name__ == "__main__":
    main()
