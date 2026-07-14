"""Quick real-data smoke run for both extraction methods (not a scientific fit)."""

from src.simulation.run_autoencoder_pipeline import run_autoencoder_model
from src.simulation.run_pipeline import run_subunit_model


def main():
    common = dict(
        cell_num=0,
        data_path="data/cell_data_01_NC.mat",
        stim_path="data/stimulus_data",
        sta_path="results/sta",
        node_num=3,
        num_epochs=2,
        max_trials=2,
        random_state=0,
    )
    classifier = run_subunit_model(
        **common, batch_size=128, learning_rate=0.05
    )
    autoencoder = run_autoencoder_model(
        **common, batch_size=128, learning_rate=0.001,
        output_activation="sigmoid",
    )
    print({
        "classifier_examples": 2 * (1500 - 20),
        "classifier_test_accuracy": classifier.test_accuracy,
        "classifier_weights": classifier.node_weights.shape,
        "classifier_subunits": len(classifier.subunits),
        "autoencoder_examples": autoencoder.n_examples,
        "autoencoder_validation_loss": autoencoder.validation_loss,
        "autoencoder_weights": autoencoder.node_weights.shape,
        "autoencoder_subunits": len(autoencoder.subunits),
    })


if __name__ == "__main__":
    main()
