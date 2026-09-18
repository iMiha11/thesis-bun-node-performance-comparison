import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP

scenarios = [
    "/simple",
    "/compute",
    "/file-read",
    "/file-write",
    "/json",
    "/auth"
]

reduction = [
    3.03,
    30.67,
    19.78,
    32.65,
    28.97,
    1.61
]

fig, ax = plt.subplots(figsize=(11.5, 6.5))

bars = ax.bar(scenarios, reduction)

ax.set_xlabel("Testni scenarij", fontsize=12)
ax.set_ylabel("Zmanjšanje glede na Node.js (%)", fontsize=12)

ax.set_title(
    "Relativno zmanjšanje ocenjenega procesorskega časa pri Bun.js",
    fontsize=15
)

ax.tick_params(axis="both", labelsize=11)
ax.grid(True, axis="y", alpha=0.25)
ax.set_ylim(0, 36)

for bar, value in zip(bars, reduction):
    rounded = Decimal(str(value)).quantize(
        Decimal("0.1"),
        rounding=ROUND_HALF_UP
    )

    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.6,
        f"{rounded} %".replace(".", ","),
        ha="center",
        va="bottom",
        fontsize=11
    )

fig.tight_layout()

plt.savefig(
    "Slika_4_7_CPU_relativna_razlika.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()