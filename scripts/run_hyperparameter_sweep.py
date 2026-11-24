"""Run node-count/regularization/learning-rate sweeps for either model."""

import argparse
import sys

from src.analysis.sweeps import (
    add_model_arguments,
    experiment_grid,
    model_settings,
    parse_values,
    report,
    run_sweep,
)


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--models", default="classifier,autoencoder")
    command.add_argument("--cells", default="0")
    command.add_argument("--seeds", default="0")
    command.add_argument("--nodes", default="7,10,20,40,60")
    command.add_argument("--l1", default="0,0.0001,0.0005")
    command.add_argument("--l2", default="0")
    command.add_argument("--sigmas", default="3")
    command.add_argument("--epochs", type=int, default=100)
    command.add_argument("--data", default="data/cell_data_01_NC.mat")
    command.add_argument("--stimuli", default="data/stimulus_data")
    command.add_argument("--sta", default="results/cache/sta")
    command.add_argument("--cache", default="results/cache")
    command.add_argument("--output", default="results/sweeps/hyperparameters")
    command.add_argument("--patience", type=int, default=None,
                         help="stop after this many epochs without validation improvement")
    add_model_arguments(command)
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--fail-fast", action="store_true")
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    learning_rates, batch_sizes = model_settings(args)
    grid = experiment_grid(
        models=parse_values(args.models, str), cells=parse_values(args.cells, int),
        seeds=parse_values(args.seeds, int), node_nums=parse_values(args.nodes, int),
        l1_coefficients=parse_values(args.l1), l2_coefficients=parse_values(args.l2),
        sigmas=parse_values(args.sigmas), fractions=[1.0], num_epochs=args.epochs,
        learning_rates=learning_rates, batch_sizes=batch_sizes,
    )
    rows = run_sweep(
        grid, data_path=args.data, stim_path=args.stimuli, sta_path=args.sta,
        output_dir=args.output, patience=args.patience, cache_dir=args.cache,
        dry_run=args.dry_run, continue_on_error=not args.fail_fast,
    )
    return report(rows, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
