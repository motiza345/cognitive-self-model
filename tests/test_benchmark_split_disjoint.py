"""Episode-disjoint split contract."""

from src.cognitive_self_model.benchmark import EnvironmentTrack
from src.cognitive_self_model.benchmark.data import (
    collect_episode_track_data,
    flatten_split,
)

TRACKS = [EnvironmentTrack.KNOWN_VALID, EnvironmentTrack.CAUSAL_BREAK]


def _split(seed_start, name, n=4):
    episodes = collect_episode_track_data(TRACKS, num_episodes=n, seed_start=seed_start)
    return flatten_split(episodes, split_name=name)


def test_disjoint_episode_ids_across_splits():
    train = _split(100, "train")
    calib = _split(400, "calibration")
    test = _split(800, "final_test")

    train_ids = set(train["episode_id"].tolist())
    calib_ids = set(calib["episode_id"].tolist())
    test_ids = set(test["episode_id"].tolist())

    assert not (train_ids & calib_ids)
    assert not (train_ids & test_ids)
    assert not (calib_ids & test_ids)


def test_episode_id_is_stable_and_unique():
    train = _split(100, "train", n=3)
    # 2 tracks x 3 episodes = 6 unique episode ids, replicated across 40 timesteps.
    ids = train["episode_id"]
    assert len(set(ids.tolist())) == 6
    assert all("__seed_" in i and "__episode_" in i for i in set(ids.tolist()))
