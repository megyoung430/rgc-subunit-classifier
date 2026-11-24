"""Train/validation/test splits that keep every trial in one set."""

import numpy as np


def split_by_trial(trial_ids, seed, test_fraction=0.25, valid_fraction=0.25):
    """Indices of the training, validation and test examples, split by whole trials.

    Consecutive frames share most of their stimulus history, so splitting
    frame by frame puts near-copies of training examples in the test set.
    Here each trial goes to exactly one set: ``test_fraction`` of the trials
    to test, ``valid_fraction`` of the rest to validation, the remainder to
    training, each set holding at least one trial.

    Args:
        trial_ids: Trial of each example.
        seed: Seed for the random assignment of trials.
        test_fraction: Fraction of trials held out for testing.
        valid_fraction: Fraction of the remaining trials used for validation.

    Returns:
        (train, valid, test) arrays of example indices, each in ascending order.

    Raises:
        ValueError: If there are fewer than 3 trials.
    """
    trial_ids = np.asarray(trial_ids)
    trials = np.unique(trial_ids)
    if len(trials) < 3:
        raise ValueError(f"a train/validation/test split needs at least 3 trials; got {len(trials)}")
    shuffled = np.random.default_rng(seed).permutation(trials)
    n_test = min(max(1, round(test_fraction * len(trials))), len(trials) - 2)
    n_valid = min(max(1, round(valid_fraction * (len(trials) - n_test))), len(trials) - n_test - 1)
    test, valid = shuffled[:n_test], shuffled[n_test:n_test + n_valid]
    in_test, in_valid = np.isin(trial_ids, test), np.isin(trial_ids, valid)
    train = ~(in_test | in_valid)
    return np.flatnonzero(train), np.flatnonzero(in_valid), np.flatnonzero(in_test)
