"""Walk through LN, autoencoder, and derived subunit-model analysis on one cell."""

import argparse
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from src.analysis.model_comparison import correlation_ceiling, prediction_metrics
from src.data_cache import cropped_stimulus, load_spike_data
from src.outputs import check_output_dir
from src.prediction.sta_model_predictions import (
    gen_sta_model,
    get_sta_predictions,
    separable_ln_filter,
)
from src.prediction.subunit_model_predictions import (
    gen_subunit_model,
    get_subunit_predictions,
)
from src.simulation.generate_training_data import crop_sta, gen_sta
from src.simulation.run_autoencoder_pipeline import run_autoencoder_model
from src.visualization.plotting_fxns import (
    MODEL_COLORS,
    outside_legend,
    plot_predictions,
    plot_subunits,
    save_figure,
    use_style,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell", type=int, default=0)
    parser.add_argument("--data", default="data/cell_data_01_NC.mat")
    parser.add_argument("--stimuli", default="data/stimulus_data")
    parser.add_argument("--sta", default="results/cache/sta")
    parser.add_argument("--cache", default="results/cache")
    parser.add_argument("--output", default="results/demo_autoencoder_model_comparison")
    parser.add_argument("--nodes", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--l1", type=float, default=0.0001)
    parser.add_argument("--l2", type=float, default=0)
    parser.add_argument("--spatial-regularization", type=float, default=0)
    parser.add_argument("--output-activation", choices=["linear", "sigmoid"], default="linear")
    parser.add_argument("--sigma", type=float, default=3)
    parser.add_argument("--morans-threshold", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-trials", type=int, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    use_style()
    output = check_output_dir(args.output)
    output.mkdir(parents=True, exist_ok=True)
    width, height, refresh_rate = 200, 150, 30
    filter_length = int(0.67 * refresh_rate)

    spikecounts = load_spike_data(args.data, Path(args.cache) / "spikes", keys=("spk1",))["spk1"][args.cell]
    num_frames, total_trials = spikecounts.shape
    num_trials = total_trials if args.max_trials is None else min(args.max_trials, total_trials)
    spikecounts = spikecounts[:, :num_trials]
    sta, spatial_sta, temporal_sta = gen_sta(
        width, height, spikecounts, args.cell, filter_length, num_frames,
        num_trials, sta_path=args.sta, stim_path=args.stimuli,
    )
    cropped_sta, crop, ellipse_mask = crop_sta(spatial_sta, args.sigma)
    crop_shape = cropped_sta.shape
    stac = sta.reshape(width, height, filter_length)[crop].reshape(
        cropped_sta.size, filter_length
    )
    stimuli = cropped_stimulus(
        args.stimuli, crop, width, height, num_frames, num_trials,
        Path(args.cache) / "stimuli",
    )

    # 1. Fit the LN baseline on running noise. The comparison LN filter is
    # the masked spatial STA times the temporal STA, the same structure as
    # the subunit model; the full STA is fitted too for reference.
    print("1/4 Fitting the LN models...")

    def fit_ln(weights):
        parameters = gen_sta_model(
            width, height, spikecounts, crop, *crop_shape, filter_length,
            num_frames, num_trials, weights, args.stimuli, stimuli=stimuli,
        )
        predictions, _, actual, time = get_sta_predictions(
            args.cell, parameters, weights, crop, *crop_shape,
            args.data, args.stimuli, filter_length=filter_length,
            cache_dir=args.cache,
        )
        return parameters, predictions, actual, time

    ln_filter = separable_ln_filter(spatial_sta, temporal_sta, crop, ellipse_mask)
    ln_parameters, ln_predictions, actual, time = fit_ln(ln_filter)
    _, full_sta_predictions, _, _ = fit_ln(stac)
    repeats = load_spike_data(args.data, Path(args.cache) / "spikes", keys=("spk2",))["spk2"]
    ceiling = correlation_ceiling(repeats[args.cell], filter_length, seed=args.seed)
    ln_metrics, full_sta_metrics = (
        prediction_metrics(actual, predictions, filter_length, ceiling=ceiling.max_correlation)
        for predictions in (ln_predictions, full_sta_predictions)
    )

    print("2/4 Training the subunit autoencoder...")
    autoencoder = run_autoencoder_model(
        args.cell, args.data, args.stimuli, sta_path=args.sta,
        node_num=args.nodes, batch_size=args.batch_size,
        learning_rate=args.learning_rate, num_epochs=args.epochs,
        l1_coefficient=args.l1, l2_coefficient=args.l2,
        spatial_coefficient=args.spatial_regularization,
        output_activation=args.output_activation, sigma=args.sigma,
        random_state=args.seed, max_trials=args.max_trials,
        morans_threshold=args.morans_threshold, cache_dir=args.cache,
    )
    np.savez_compressed(
        output / "autoencoder_weights_and_subunits.npz",
        node_weights=autoencoder.node_weights, subunits=autoencoder.subunits,
        ellipse_mask=autoencoder.ellipse_mask,
    )
    figure, _ = plot_subunits(autoencoder.node_weights, crop_shape)
    figure.suptitle("All Autoencoder Encoder Weights")
    save_figure(figure, output / "autoencoder_node_weights", caption=(
        "Spatial weight map of every autoencoder node on the cropped window (red positive, "
        "blue negative)."))
    plt.close(figure)
    if len(autoencoder.subunits):
        figure, _ = plot_subunits(autoencoder.subunits, crop_shape)
        figure.suptitle("Moran's-I-Selected Autoencoder Subunits")
        save_figure(figure, output / "selected_subunits", caption=(
            f"Autoencoder nodes whose Moran's I (inside the ellipse mask) exceeds "
            f"{args.morans_threshold}."))
        plt.close(figure)
    else:
        raise RuntimeError(
            "the autoencoder produced no weights above the Moran's-I threshold; "
            "inspect autoencoder_node_weights.png, train/tune the model, or "
            "explicitly choose a different --morans-threshold"
        )

    print("3/4 Fitting the derived subunit model...")
    subunit_parameters, combination_weights, normalized_subunits = gen_subunit_model(
        autoencoder.subunits, width, height, spikecounts, filter_length,
        num_frames, num_trials, spatial_sta, temporal_sta, crop,
        *crop_shape, args.stimuli, stimuli=stimuli,
    )
    subunit_predictions, subunit_correlation, _, _ = get_subunit_predictions(
        args.cell, subunit_parameters, combination_weights,
        normalized_subunits, temporal_sta, crop, *crop_shape,
        args.data, args.stimuli, filter_length=filter_length,
        cache_dir=args.cache,
    )
    subunit_metrics = prediction_metrics(
        actual, subunit_predictions, filter_length, ceiling=ceiling.max_correlation
    )

    print("4/4 Saving frozen-noise comparison...")
    figure, axis = plt.subplots(figsize=(10, 4))
    plot_predictions(ln_predictions, actual, time, filter_length, label="Separable LN",
        color=MODEL_COLORS["LN"], ax=axis)
    axis.plot(time[filter_length:], subunit_predictions, label="Autoencoder Subunit Model",
        color=MODEL_COLORS["Autoencoder Subunit"])
    outside_legend(axis)
    figure.tight_layout()
    save_figure(figure, output / "frozen_noise_predictions", caption=(
        f"Measured frozen-noise firing rate of cell {args.cell} with the separable LN and "
        f"autoencoder subunit model predictions. LN r = {ln_metrics.correlation:.3f}, subunit r = "
        f"{subunit_metrics.correlation:.3f}, noise ceiling {ceiling.max_correlation:.3f}."))
    plt.close(figure)

    summary = {
        "cell": args.cell, "training_trials": num_trials,
        "autoencoder": {
            "nodes": args.nodes, "selected_subunits": len(autoencoder.subunits),
            "validation_loss": autoencoder.validation_loss,
            "test_loss": autoencoder.test_loss,
            "output_activation": args.output_activation,
        },
        "noise_ceiling": {
            "max_correlation": ceiling.max_correlation,
            "reliability": ceiling.reliability,
        },
        "ln_model": {
            "filter": "masked spatial STA x temporal STA",
            "correlation": ln_metrics.correlation, "mse": ln_metrics.mse,
            "normalized_correlation": ln_metrics.normalized_correlation,
        },
        "ln_model_full_sta": {
            "correlation": full_sta_metrics.correlation, "mse": full_sta_metrics.mse,
            "normalized_correlation": full_sta_metrics.normalized_correlation,
        },
        "subunit_model": {
            "correlation": subunit_metrics.correlation, "mse": subunit_metrics.mse,
            "normalized_correlation": subunit_metrics.normalized_correlation,
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
