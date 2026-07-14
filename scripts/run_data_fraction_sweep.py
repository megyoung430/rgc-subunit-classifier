"""Measure extraction behavior as the number of training trials changes."""

import argparse

from src.analysis.sweeps import experiment_grid, parse_values, run_sweep


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--models", default="classifier,autoencoder")
    command.add_argument("--cells", default="0")
    command.add_argument("--seeds", default="0,1,2")
    command.add_argument("--fractions", default="0.1,0.25,0.5,0.75,1")
    command.add_argument("--nodes", default="60")
    command.add_argument("--l1", default="0.0001")
    command.add_argument("--l2", default="0")
    command.add_argument("--learning-rates", default="0.001")
    command.add_argument("--sigmas", default="3")
    command.add_argument("--epochs", type=int, default=100)
    command.add_argument("--batch-size", type=int, default=100)
    command.add_argument("--data", default="data/cell_data_01_NC.mat")
    command.add_argument("--stimuli", default="data/stimulus_data")
    command.add_argument("--sta", default="results/sta")
    command.add_argument("--output", default="results/sweeps/data_fractions")
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--fail-fast", action="store_true")
    return command


def main(argv=None):
    args = parser().parse_args(argv)
    grid = experiment_grid(
        models=parse_values(args.models, str), cells=parse_values(args.cells, int),
        seeds=parse_values(args.seeds, int), node_nums=parse_values(args.nodes, int),
        l1_coefficients=parse_values(args.l1), l2_coefficients=parse_values(args.l2),
        learning_rates=parse_values(args.learning_rates),
        sigmas=parse_values(args.sigmas), fractions=parse_values(args.fractions),
        num_epochs=args.epochs,
    )
    rows = run_sweep(
        grid, data_path=args.data, stim_path=args.stimuli, sta_path=args.sta,
        output_dir=args.output, batch_size=args.batch_size,
        dry_run=args.dry_run, continue_on_error=not args.fail_fast,
    )
    print(f"{len(rows)} run(s) {'planned' if args.dry_run else 'processed'}")
    if args.dry_run:
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
