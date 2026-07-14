"""Walk through LN, classifier, and derived subunit-model analysis on one cell."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.io import loadmat

from src.analysis.model_comparison import prediction_metrics
from src.prediction.sta_model_predictions import (
    gen_sta_model, get_sta_predictions,
)
from src.prediction.subunit_model_predictions import (
    gen_subunit_model, get_subunit_predictions,
)
from src.simulation.generate_training_data import crop_sta, gen_sta
from src.simulation.run_pipeline import run_subunit_model
from src.visualization.plotting_fxns import plot_predictions, plot_subunits


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=int, default=0)
    parser.add_argument("--data", default="data/cell_data_01_NC.mat")
    parser.add_argument("--stimuli", default="data/stimulus_data")
    parser.add_argument("--sta", default="results/sta")
    parser.add_argument("--output", default="results/demo_classifier_model_comparison")
    parser.add_argument("--nodes", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--l1", type=float, default=0.0001)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--sigma", type=float, default=3)
    parser.add_argument("--morans-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-trials", type=int, default=None,
        help="Use only the first N trials. Omit for the full 219-trial analysis.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    width, height, refresh_rate = 200, 150, 30
    filter_length = int(0.67 * refresh_rate)

    spike_data = loadmat(args.data)["spk1"]
    spikecounts = spike_data[args.cell]
    num_frames, total_trials = spikecounts.shape
    num_trials = total_trials if args.max_trials is None else min(args.max_trials, total_trials)
    spikecounts = spikecounts[:, :num_trials]

    names = {
        "spatial_sta_filename": f"Cell {args.cell} Uncropped Spatial STA.h5",
        "temp_sta_filename": f"Cell {args.cell} Uncropped Temp STA.h5",
        "sta_filename": f"Cell {args.cell} Uncropped STA.h5",
    }
    sta, spatial_sta, temporal_sta = gen_sta(
        width, height, spikecounts, args.cell, filter_length, num_frames,
        num_trials, sta_path=args.sta, stim_path=args.stimuli, **names,
    )
    cropped_sta, crop, ellipse_mask = crop_sta(spatial_sta, args.sigma)
    crop_shape = cropped_sta.shape
    stac = sta.reshape(width, height, filter_length)[crop].reshape(
        cropped_sta.size, filter_length
    )

    # 1. Fit the conventional linear-nonlinear baseline on running noise.
    print("1/4 Fitting the LN model...")
    ln_parameters = gen_sta_model(
        width, height, spikecounts, crop, *crop_shape, filter_length,
        num_frames, num_trials, stac, args.stimuli,
    )
    ln_predictions, ln_correlation, actual, time = get_sta_predictions(
        args.cell, ln_parameters, stac, crop, *crop_shape,
        args.data, args.stimuli, filter_length=filter_length,
    )
    ln_metrics = prediction_metrics(
        actual, ln_predictions, n_parameters=3 + stac.size,
        filter_length=filter_length,
    )

    # 2. Train the classifier and reconstruct hidden weights spatially.
    print("2/4 Training the subunit classifier...")
    classifier = run_subunit_model(
        args.cell, args.data, args.stimuli, sta_path=args.sta,
        node_num=args.nodes, batch_size=args.batch_size,
        learning_rate=args.learning_rate, num_epochs=args.epochs,
        l1_coefficient=args.l1, l2_coefficient=args.l2,
        sigma=args.sigma, random_state=args.seed, max_trials=args.max_trials,
        morans_threshold=args.morans_threshold,
    )
    np.savez_compressed(
        output / "classifier_weights_and_subunits.npz",
        node_weights=classifier.node_weights, subunits=classifier.subunits,
        ellipse_mask=classifier.ellipse_mask,
    )
    figure, _ = plot_subunits(classifier.node_weights, crop_shape)
    figure.suptitle("All classifier hidden-node weights")
    figure.savefig(output / "classifier_node_weights.png", dpi=180)
    plt.close(figure)
    if len(classifier.subunits):
        figure, _ = plot_subunits(classifier.subunits, crop_shape)
        figure.suptitle("Moran's-I-selected classifier subunits")
        figure.savefig(output / "selected_subunits.png", dpi=180)
        plt.close(figure)
    else:
        raise RuntimeError(
            "the classifier produced no weights above the Moran's-I threshold; "
            "inspect classifier_node_weights.png, train longer, tune the model, "
            "or explicitly choose a different --morans-threshold"
        )

    # 3. Fit the rectified subunit model using the selected spatial filters.
    print("3/4 Fitting the derived subunit model...")
    subunit_parameters, combination_weights, normalized_subunits = gen_subunit_model(
        classifier.subunits, width, height, spikecounts, filter_length,
        num_frames, num_trials, spatial_sta, temporal_sta, crop,
        *crop_shape, args.stimuli,
    )
    subunit_predictions, subunit_correlation, _, _ = get_subunit_predictions(
        args.cell, subunit_parameters, combination_weights,
        normalized_subunits, temporal_sta, crop, *crop_shape,
        args.data, args.stimuli, filter_length=filter_length,
    )
    # Output nonlinearity (3) + one combination weight per selected subunit.
    subunit_metrics = prediction_metrics(
        actual, subunit_predictions,
        n_parameters=3 + len(combination_weights), filter_length=filter_length,
    )

    # 4. Compare both models on exactly the same frozen-noise response.
    print("4/4 Saving frozen-noise comparison...")
    _, axis = plt.subplots(figsize=(10, 4))
    plot_predictions(
        ln_predictions, actual, time, filter_length, label="LN", ax=axis
    )
    axis.plot(
        time[filter_length:], subunit_predictions, label="Subunit model"
    )
    axis.legend()
    axis.figure.tight_layout()
    axis.figure.savefig(output / "frozen_noise_predictions.png", dpi=180)
    plt.close(axis.figure)

    summary = {
        "cell": args.cell, "training_trials": num_trials,
        "classifier": {
            "nodes": args.nodes, "selected_subunits": len(classifier.subunits),
            "test_loss": classifier.test_loss,
            "test_accuracy": classifier.test_accuracy,
        },
        "ln_model": {
            "correlation": ln_correlation, "mse": ln_metrics.mse,
            "aic": ln_metrics.aic, "bic": ln_metrics.bic,
        },
        "subunit_model": {
            "correlation": subunit_correlation, "mse": subunit_metrics.mse,
            "aic": subunit_metrics.aic, "bic": subunit_metrics.bic,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez_compressed(
        output / "model_predictions.npz", actual=actual, time=time,
        ln_predictions=ln_predictions, subunit_predictions=subunit_predictions,
        ln_parameters=ln_parameters, subunit_parameters=subunit_parameters,
        combination_weights=combination_weights,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
