import numpy as np
import pytest

from src.simulation.splits import split_by_trial


def test_trials_go_to_exactly_one_set_and_are_reproducible():
    trial_ids = np.repeat(np.arange(20), 7)
    train, valid, test = split_by_trial(trial_ids, seed=3)
    for a, b in ((train, valid), (train, test), (valid, test)):
        assert not set(trial_ids[a]) & set(trial_ids[b])
    assert np.array_equal(np.sort(np.concatenate([train, valid, test])), np.arange(140))
    assert len(set(trial_ids[test])) == 5 and len(set(trial_ids[valid])) == 4
    again = split_by_trial(trial_ids, seed=3)
    for x, y in zip((train, valid, test), again):
        np.testing.assert_array_equal(x, y)


def test_too_few_trials_is_an_error():
    with pytest.raises(ValueError, match="at least 3 trials"):
        split_by_trial(np.repeat(np.arange(2), 5), seed=0)
