"""Regenerate Gaussian replay figures from the saved population summaries."""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT = Path(__file__).resolve().parents[2] / "experiments" / "output"
ENVIRONMENTS = {
    "mountaincar_continuous": "Mountain Car",
    "pendulum": "Pendulum",
    "acrobot": "Acrobot",
    "reacher": "Reacher",
    "halfcheetah": "Half-Cheetah",
}
RATES = {"swap": "Swap", "delete": "Delete", "add": "Add", "mutate": "One-point"}
MEAN = "mean_abs_dp"
SD = "sd_seed_mean_abs_dp"


def read_population(folder):
    with (folder / "population.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, stem):
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=180)
    plt.close(fig)


def plot_generation(source_folder, figure_folder, *, single_column=False):
    # Average channels within each champion, then give each seed equal weight.
    by_seed = defaultdict(list)
    with (source_folder / "summary_all_seeds.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["generation"]) != 5000:
                continue
            key = (row["environment"], row["rate"], float(row["alpha"]), row["seed"])
            by_seed[key].append(float(row[MEAN]))
    rate_values = defaultdict(list)
    for (env, rate, alpha, _seed), channels in by_seed.items():
        rate_values[env, rate, alpha].append(statistics.fmean(channels))
    environments = [env for env in ENVIRONMENTS if any(key[0] == env for key in rate_values)]
    shape = (len(environments), 1) if single_column else (1, len(environments))
    figsize = (3.35, 5.5) if single_column else (14, 3.7)
    fig, axes = plt.subplots(*shape, figsize=figsize, sharex=single_column,
                             squeeze=False, layout="constrained")
    for ax, env in zip(axes.flat, environments):
        for rate, label in RATES.items():
            series = sorted(
                (alpha, statistics.fmean(values))
                for (environment, operation, alpha), values in rate_values.items()
                if environment == env and operation == rate
            )
            alphas = [alpha for alpha, _mean in series]
            means = [100 * mean for _alpha, mean in series]
            ax.plot(alphas, means, label=label, marker="o", markersize=3)
        if single_column:
            ax.set_title(ENVIRONMENTS[env], loc="left", fontsize=8)
            ax.tick_params(labelsize=7, length=2)
            ax.locator_params(axis="y", nbins=3)
            ax.margins(x=.03)
        else:
            ax.set(title=ENVIRONMENTS[env], xlabel="Noise scale α")
        ax.grid(alpha=.2)
    if single_column:
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncols=4,
                   fontsize=7, frameon=False, columnspacing=1, handlelength=1.4)
        axes[-1, 0].set(xlabel="Noise scale α", xticks=alphas)
        axes[-1, 0].set_xticklabels([f"{alpha:g}" for alpha in alphas])
        axes[-1, 0].xaxis.label.set_size(8)
        fig.supylabel("Mean absolute rate change (percentage points)", fontsize=8)
        save(fig, figure_folder / "population_single_column")
        return
    axes[0, 0].set_ylabel("Mean absolute rate change\n(percentage points)")
    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Gaussian replay at generation 5000\nMean across observation channels and seeds")
    save(fig, figure_folder / "population")


def plot_observations(folder):
    by_alpha = defaultdict(list)
    for row in read_population(folder):
        by_alpha[float(row["alpha"])].append(row)

    # Marker area is proportional to the statistic in both panels and at every alpha.
    area_scale = 3200
    for alpha, rows in sorted(by_alpha.items()):
        environments = [env for env in ENVIRONMENTS if any(r["environment"] == env for r in rows)]
        fig, axes = plt.subplots(len(environments), 2, figsize=(13, 11), squeeze=False)
        fig.subplots_adjust(left=.085, right=.985, top=.89, bottom=.10, hspace=.9, wspace=.14)
        for i, env in enumerate(environments):
            subset = [r for r in rows if r["environment"] == env]
            observations = sorted({int(r["observation"]) for r in subset})
            rate_index = {rate: j for j, rate in enumerate(RATES)}
            for ax, field, color in zip(axes[i], (MEAN, SD), ("#275580", "#9a5319")):
                ax.scatter([int(r["observation"]) for r in subset],
                           [rate_index[r["rate"]] for r in subset],
                           s=[area_scale * float(r[field]) for r in subset],
                           color=color, linewidths=0, zorder=3)
                ax.set(xticks=observations, yticks=range(len(RATES)),
                       xlim=(observations[0] - .5, observations[-1] + .5),
                       ylim=(len(RATES) - .5, -.5))
                ax.set_xticklabels([fr"$o_{{{obs}}}$" for obs in observations], fontsize=9)
                ax.set_yticklabels(list(RATES.values()) if field == MEAN else [], fontsize=9)
                ax.xaxis.tick_top()
                ax.tick_params(length=0)
                ax.grid(color="0.92")
                for spine in ax.spines.values():
                    spine.set_visible(False)
            left, right = (ax.get_position() for ax in axes[i])
            fig.text((left.x0 + right.x1) / 2, left.y1 + .034, ENVIRONMENTS[env],
                     ha="center", fontsize=12)
        for x, label in ((.29, "Mean"), (.77, "Sample SD across seeds")):
            fig.text(x, .946, label, ha="center", fontsize=12, weight="bold")
        fig.suptitle(f"Gaussian replay by observation · α = {alpha:g}", y=.988)
        labels = [0.001, 0.01, 0.05]
        handles = [axes[0, 0].scatter([], [], s=area_scale * value, color="0.30", linewidths=0)
                   for value in labels]
        fig.legend(handles, [f"{100 * value:g}" for value in labels], ncol=len(labels),
                   loc="lower center", bbox_to_anchor=(.5, .016), frameon=False,
                   title="Absolute rate change (percentage points) · zero has no marker")
        save(fig, folder / f"alpha_{alpha:g}")


if __name__ == "__main__":
    observations = OUTPUT / "rate_sensitivity_gaussian_individual_latest"
    plot_generation(observations, OUTPUT / "rate_sensitivity_gaussian_latest")
    plot_generation(observations, OUTPUT / "rate_sensitivity_gaussian_latest", single_column=True)
    plot_observations(observations)
