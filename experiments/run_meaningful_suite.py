"""
Run a small, curated suite of experiments (exactly 5) intended to be "most meaningful":

1) LinReg (Diabetes): homogeneous sanity check
2) LogReg (a9a): homogeneous
3) LogReg (a9a): heterogeneous (straggler)
4) LogReg (a9a): dominance setup (bigger)
5) LogReg (a9a): scalability (bigger)
"""

import argparse
import os
import subprocess
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(module, config_rel, experiment, plots_dir_rel):
    cmd = [
        sys.executable,
        "-m",
        module,
        "--config",
        config_rel,
        "--experiment",
        experiment,
        "--plots-dir",
        plots_dir_rel,
    ]
    print("\n" + "=" * 80)
    print("Running:", " ".join(cmd))
    print("cwd:", REPO_ROOT)
    print("=" * 80)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="Run curated 5-experiment suite")
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="../plots",
        help="Directory to save plots (relative to experiments/)",
    )
    args = parser.parse_args()

    # These paths are interpreted by the runners relative to experiments/
    linreg_cfg = "configs/linreg_diabetes_smoke.yaml"
    logreg_a9a_suite_cfg = "configs/logreg_a9a_suite.yaml"

    plots_dir_rel = args.plots_dir

    _run("experiments.run_linreg_experiments", linreg_cfg, "homogeneous", plots_dir_rel)

    _run("experiments.run_logreg_experiments", logreg_a9a_suite_cfg, "homogeneous", plots_dir_rel)
    _run("experiments.run_logreg_experiments", logreg_a9a_suite_cfg, "heterogeneous", plots_dir_rel)
    _run("experiments.run_logreg_experiments", logreg_a9a_suite_cfg, "dominance", plots_dir_rel)
    _run("experiments.run_logreg_experiments", logreg_a9a_suite_cfg, "scalability", plots_dir_rel)

    print("\nDone. Plots saved under:", os.path.normpath(os.path.join(os.path.dirname(__file__), plots_dir_rel)))


if __name__ == "__main__":
    main()


