import csv
import os
import random
import selectors
import statistics
import subprocess
import time
from pathlib import Path


# ---------------------------------------------------------
# NASTAVITVE TESTA
# ---------------------------------------------------------

REPETITIONS = 100
PILOT_REPETITIONS = 5

PAUSE_BETWEEN_TESTS_SECONDS = 0.20
STARTUP_TIMEOUT_SECONDS = 5

BASE_DIR = Path(__file__).resolve().parent
SERVER_FILE = BASE_DIR / "app" / "server.mjs"
RESULTS_DIR = BASE_DIR / "results"

NODE_COMMAND = ["node", str(SERVER_FILE)]
BUN_COMMAND = ["bun", str(SERVER_FILE)]

NODE_PORT = 3100
BUN_PORT = 3101


# ---------------------------------------------------------
# PREVERJANJE RAZLIČIC
# ---------------------------------------------------------

def get_runtime_version(command: str) -> str:
    try:
        result = subprocess.run(
            [command, "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        version = result.stdout.strip()

        if not version:
            version = result.stderr.strip()

        return version

    except FileNotFoundError as error:
        raise RuntimeError(
            f"Ukaz '{command}' ni bil najden. "
            "Preveri namestitev in spremenljivko PATH."
        ) from error


# ---------------------------------------------------------
# USTAVLJANJE PROCESA
# ---------------------------------------------------------

def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return

    process.terminate()

    try:
        process.wait(timeout=2)

    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=2)


# ---------------------------------------------------------
# MERJENJE ENEGA ZAGONA
# ---------------------------------------------------------

def measure_startup(
    command: list[str],
    port: int
) -> float:
    """
    Meri čas od sprožitve novega procesa do izpisa
    SERVER_READY.

    SERVER_READY se izpiše v callback funkciji listen(),
    kar pomeni, da je strežnik začel poslušati na vratih.

    Rezultat je vrnjen v milisekundah.
    """

    environment = os.environ.copy()
    environment["PORT"] = str(port)

    start_time = time.perf_counter_ns()

    process = subprocess.Popen(
        command,
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=environment
    )

    if process.stdout is None:
        stop_process(process)
        raise RuntimeError("Standardnega izhoda ni mogoče prebrati.")

    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)

    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS

    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                error_output = ""

                if process.stderr is not None:
                    error_output = process.stderr.read()

                raise RuntimeError(
                    "Proces se je končal pred pripravljenostjo.\n"
                    f"{error_output}"
                )

            events = selector.select(timeout=0.01)

            for key, _ in events:
                line = key.fileobj.readline().strip()

                if line == "SERVER_READY":
                    end_time = time.perf_counter_ns()

                    return (end_time - start_time) / 1_000_000

        raise TimeoutError(
            f"Proces se ni zagnal v "
            f"{STARTUP_TIMEOUT_SECONDS} sekundah."
        )

    finally:
        selector.close()
        stop_process(process)


# ---------------------------------------------------------
# STATISTIKA
# ---------------------------------------------------------

def percentile(
    values: list[float],
    proportion: float
) -> float:
    ordered = sorted(values)

    position = (len(ordered) - 1) * proportion
    lower_index = int(position)
    upper_index = min(
        lower_index + 1,
        len(ordered) - 1
    )

    fraction = position - lower_index

    return (
        ordered[lower_index] * (1 - fraction)
        + ordered[upper_index] * fraction
    )


def calculate_summary(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99)
    }


# ---------------------------------------------------------
# POSKUSNI ZAGONI
# ---------------------------------------------------------

def perform_pilot_runs() -> None:
    """
    Izvede pet nezabeleženih poskusnih zagonov.

    Ti zagoni niso vključeni v končne rezultate.
    Namenjeni so preverjanju delovanja testa in zmanjšanju
    vpliva popolnoma prvega zagona po namestitvi.
    """

    print(
        f"\nIzvajam {PILOT_REPETITIONS} "
        "nezabeleženih poskusnih zagonov ..."
    )

    for repetition in range(1, PILOT_REPETITIONS + 1):
        measure_startup(
            command=NODE_COMMAND,
            port=NODE_PORT
        )

        time.sleep(PAUSE_BETWEEN_TESTS_SECONDS)

        measure_startup(
            command=BUN_COMMAND,
            port=BUN_PORT
        )

        time.sleep(PAUSE_BETWEEN_TESTS_SECONDS)

        print(
            f"Poskusni zagon "
            f"{repetition}/{PILOT_REPETITIONS}"
        )


# ---------------------------------------------------------
# GLAVNI TEST
# ---------------------------------------------------------

def perform_benchmark() -> tuple[list[float], list[float]]:
    node_results = []
    bun_results = []

    print("\nZačenjam glavne meritve zagona ...")

    for repetition in range(1, REPETITIONS + 1):
        order = ["node", "bun"]
        random.shuffle(order)

        current_results = {}

        for runtime in order:
            if runtime == "node":
                startup_time = measure_startup(
                    command=NODE_COMMAND,
                    port=NODE_PORT
                )

                node_results.append(startup_time)
                current_results["node"] = startup_time

            else:
                startup_time = measure_startup(
                    command=BUN_COMMAND,
                    port=BUN_PORT
                )

                bun_results.append(startup_time)
                current_results["bun"] = startup_time

            time.sleep(PAUSE_BETWEEN_TESTS_SECONDS)

        print(
            f"Ponovitev {repetition:3}/{REPETITIONS} | "
            f"Node.js: {current_results['node']:.3f} ms | "
            f"Bun.js: {current_results['bun']:.3f} ms"
        )

    return node_results, bun_results


# ---------------------------------------------------------
# SHRANJEVANJE MERITEV
# ---------------------------------------------------------

def save_measurements(
    node_results: list[float],
    bun_results: list[float]
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    file_path = RESULTS_DIR / "startup_measurements.csv"

    with file_path.open(
        mode="w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "repetition",
            "node_startup_ms",
            "bun_startup_ms"
        ])

        for index in range(REPETITIONS):
            writer.writerow([
                index + 1,
                f"{node_results[index]:.6f}",
                f"{bun_results[index]:.6f}"
            ])


def save_summary(
    node_results: list[float],
    bun_results: list[float]
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    file_path = RESULTS_DIR / "startup_summary.csv"

    summaries = {
        "Node.js": calculate_summary(node_results),
        "Bun.js": calculate_summary(bun_results)
    }

    with file_path.open(
        mode="w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "runtime",
            "count",
            "mean_ms",
            "median_ms",
            "stdev_ms",
            "minimum_ms",
            "maximum_ms",
            "p95_ms",
            "p99_ms"
        ])

        for runtime, summary in summaries.items():
            writer.writerow([
                runtime,
                summary["count"],
                f"{summary['mean']:.6f}",
                f"{summary['median']:.6f}",
                f"{summary['stdev']:.6f}",
                f"{summary['minimum']:.6f}",
                f"{summary['maximum']:.6f}",
                f"{summary['p95']:.6f}",
                f"{summary['p99']:.6f}"
            ])


# ---------------------------------------------------------
# SHRANJEVANJE PODATKOV O OKOLJU
# ---------------------------------------------------------

def save_environment_info(
    node_version: str,
    bun_version: str
) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    file_path = RESULTS_DIR / "test_environment.txt"

    commands = {
        "Fedora": ["cat", "/etc/fedora-release"],
        "Kernel": ["uname", "-r"],
        "Architecture": ["uname", "-m"],
        "CPU count": ["nproc"],
        "Memory": ["free", "-h"],
        "Python": ["python3", "--version"]
    }

    with file_path.open(
        mode="w",
        encoding="utf-8"
    ) as file:
        file.write(f"Node.js: {node_version}\n")
        file.write(f"Bun.js: {bun_version}\n\n")

        for label, command in commands.items():
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )

            file.write(f"{label}:\n")
            file.write(result.stdout.strip())
            file.write("\n\n")


# ---------------------------------------------------------
# IZPIS POVZETKA
# ---------------------------------------------------------

def print_summary(
    runtime: str,
    values: list[float]
) -> None:
    summary = calculate_summary(values)

    print(f"\n{runtime}")
    print(f"Število meritev: {summary['count']}")
    print(f"Povprečje: {summary['mean']:.3f} ms")
    print(f"Mediana: {summary['median']:.3f} ms")
    print(f"Standardni odklon: {summary['stdev']:.3f} ms")
    print(f"Minimum: {summary['minimum']:.3f} ms")
    print(f"Maksimum: {summary['maximum']:.3f} ms")
    print(f"P95: {summary['p95']:.3f} ms")
    print(f"P99: {summary['p99']:.3f} ms")


# ---------------------------------------------------------
# GLAVNI PROGRAM
# ---------------------------------------------------------

def main() -> None:
    if not SERVER_FILE.exists():
        raise FileNotFoundError(
            f"Datoteka ne obstaja: {SERVER_FILE}"
        )

    node_version = get_runtime_version("node")
    bun_version = get_runtime_version("bun")

    print("RV1 – primerjava časa zagona aplikacije")
    print(f"Node.js: {node_version}")
    print(f"Bun.js: {bun_version}")
    print(f"Število meritev: {REPETITIONS}")

    save_environment_info(
        node_version=node_version,
        bun_version=bun_version
    )

    perform_pilot_runs()

    node_results, bun_results = perform_benchmark()

    save_measurements(
        node_results=node_results,
        bun_results=bun_results
    )

    save_summary(
        node_results=node_results,
        bun_results=bun_results
    )

    print_summary(
        runtime="Node.js",
        values=node_results
    )

    print_summary(
        runtime="Bun.js",
        values=bun_results
    )

    print("\nRezultati so shranjeni v mapi results.")


if __name__ == "__main__":
    main()