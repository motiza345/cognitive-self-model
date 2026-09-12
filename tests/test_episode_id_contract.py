import numpy as np


def make_global_episode_key(
    track: np.ndarray,
    episode_id: np.ndarray,
) -> np.ndarray:
    """
    Build a globally unique episode key.

    A local episode id like 'episode_000' is not globally unique
    unless it is combined with the track identity.
    """
    return np.asarray(
        [
            f"{str(track_name)}::{str(local_episode_id)}"
            for track_name, local_episode_id
            in zip(track, episode_id)
        ],
        dtype=str,
    )


def test_same_track_multiple_episodes_remain_distinct():
    track = np.array([
        "KNOWN_VALID",
        "KNOWN_VALID",
        "KNOWN_VALID",
        "KNOWN_VALID",
    ])

    episode_id = np.array([
        "calibration_episode_000",
        "calibration_episode_000",
        "calibration_episode_001",
        "calibration_episode_001",
    ])

    global_episode_keys = make_global_episode_key(
        track=track,
        episode_id=episode_id,
    )

    assert len(np.unique(global_episode_keys)) == 2


def test_same_local_id_in_different_tracks_remains_distinct():
    track = np.array([
        "KNOWN_VALID",
        "CAUSAL_BREAK",
    ])

    episode_id = np.array([
        "episode_000",
        "episode_000",
    ])

    global_episode_keys = make_global_episode_key(
        track=track,
        episode_id=episode_id,
    )

    assert len(np.unique(global_episode_keys)) == 2
