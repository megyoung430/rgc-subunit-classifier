import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from src.visualization.plotting_fxns import plot_subunits, save_figure  # noqa: E402


def test_save_figure_writes_three_formats_and_a_caption(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    stem = tmp_path / "figs" / "Example Figure"
    save_figure(fig, stem, caption="What the figure shows.")
    for suffix in (".pdf", ".png", ".svg"):
        assert stem.with_name(stem.name + suffix).exists()
    assert stem.with_name(stem.name + ".caption.txt").read_text() == "What the figure shows.\n"
    plt.close(fig)


def test_save_figure_refuses_the_archive(tmp_path, monkeypatch):
    import src.outputs

    archive = tmp_path / "results" / "colab_legacy"
    monkeypatch.setattr(src.outputs, "ARCHIVE", archive)
    fig, _ = plt.subplots()
    with pytest.raises(PermissionError):
        save_figure(fig, archive / "x")
    plt.close(fig)


def test_plot_subunits_draws_into_given_axes_with_titles():
    fig, axes = plt.subplots(1, 3)
    weights = np.random.default_rng(0).normal(size=(2, 12))
    returned_fig, returned_axes = plot_subunits(weights, (3, 4), axes=axes, titles=["A", "B"])
    assert returned_fig is fig
    assert [ax.get_title() for ax in axes[:2]] == ["A", "B"]
    assert not axes[2].get_visible()
    plt.close(fig)
