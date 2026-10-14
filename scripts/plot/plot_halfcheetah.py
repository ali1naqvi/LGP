"""HalfCheetah training and validation medians with interquartile bands."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_mutation_validation_comparison import ROOT, load_experiment, summarize


GROUPS = {
    "Mutation-rate strategies": (
        ("Modified rates", "halfcheetah_execution_modified_rates"),
        ("Fixed individual rates", "halfcheetah_fixed_individual"),
        ("Fixed rates", "halfcheetah_fixed_rates"),
        ("Inherited rates", "halfcheetah_inherited_rates"),
    ),
    "Modified rates: mutation operators": (
        ("All mutations", "halfcheetah_execution_modified_rates"),
        ("Add", "halfcheetah_execution_modified_rates_add"),
        ("Delete", "halfcheetah_execution_modified_rates_delete"),
        ("Swap", "halfcheetah_execution_modified_rates_swap"),
        ("One-point", "halfcheetah_execution_modified_rates_one_point"),
    ),
}


def main():
    plt.style.use("seaborn-v0_8-whitegrid")
    output = ROOT / "output" / "halfcheetah"
    output.mkdir(parents=True, exist_ok=True)
    for metric, phase in (("best_fitness", "training"), ("validation_fitness", "validation")):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for ax, (title, experiments) in zip(axes, GROUPS.items()):
            loaded = [(label, *load_experiment(name, metric)) for label, name in experiments]
            endpoint = min(min(max(run) for run in runs) for _, runs, _ in loaded)
            if endpoint <= 0:
                raise ValueError(f"No shared range for {title}: {phase}")
            for label, runs, total in loaded:
                generations, median, q25, q75 = summarize(runs, endpoint)
                line, = ax.plot(generations, median, label=f"{label} (n={len(runs)})", linewidth=1.7)
                ax.fill_between(generations, q25, q75, color=line.get_color(), alpha=.12)
                print(f"{phase}: {label}: {len(runs)}/{total} logs, through generation {endpoint}", flush=True)
            ax.set(title=title, xlabel="Generation", ylabel=f"{phase.capitalize()} fitness", xlim=(0, endpoint))
            ax.legend(fontsize=9)
        fig.suptitle(f"HalfCheetah — {phase.capitalize()}\nMedian across runs; shaded band: 25th–75th percentile")
        for suffix in ("png", "pdf"):
            path = output / f"halfcheetah_{phase}.{suffix}"
            fig.savefig(path, dpi=180, bbox_inches="tight")
            print(path, flush=True)
        plt.close(fig)


if __name__ == "__main__":
    main()
