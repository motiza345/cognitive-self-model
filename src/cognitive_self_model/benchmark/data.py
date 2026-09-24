"""
Episode generation and split flattening.

Each episode runs a fresh, seed-controlled environment/encoder pair for a fixed
number of steps. Splits are made **episode-disjoint by seed range**: every
episode is assigned a globally unique, human-readable ``episode_id`` string of
the form ``{split}__{track}__seed_{seed}__episode_{index:03d}``.

The per-row label ``y`` is ``self_model_invalid`` (the environment's
``is_contextually_invalid`` flag).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .environment import ScientificFrozenEnvironment
from .evidence import ScientificFrozenEncoder
from .tracks import EnvironmentTrack


def collect_episode_track_data(
    tracks: List[EnvironmentTrack],
    num_episodes: int,
    seed_start: int,
    *,
    steps_per_episode: int = 40,
    noise_std: float = 0.05,
    probe_measurement_std: float = 0.01,
    probe_offset: float = 0.2,
    window_size: int = 15,
    onset_step: int = 20,
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate episodes for each track, incrementing the seed per episode."""
    episodes_by_track: Dict[str, List[Dict[str, Any]]] = {}
    current_seed = int(seed_start)

    for track in tracks:
        track_episodes: List[Dict[str, Any]] = []

        for episode_index in range(int(num_episodes)):
            env = ScientificFrozenEnvironment(
                track=track,
                noise_std=noise_std,
                probe_measurement_std=probe_measurement_std,
                seed=current_seed,
                onset_step=onset_step,
            )
            encoder = ScientificFrozenEncoder(window_size=window_size)
            env.reset()
            encoder.reset()

            x_frequency = env.rng.uniform(0.05, 0.15)

            X_rows, y_rows, t_rows = [], [], []
            for step in range(int(steps_per_episode)):
                x = np.sin(step * x_frequency)
                u = np.cos(step * x_frequency)
                obs, gt = env.step(
                    x=float(x),
                    u=float(u),
                    probe_inputs=[float(u + probe_offset)],
                )
                X_rows.append(encoder.encode(obs))
                y_rows.append(gt.self_model_invalid)
                t_rows.append(obs.t)

            track_episodes.append(
                {
                    "X": np.asarray(X_rows, dtype=float),
                    "y": np.asarray(y_rows, dtype=int),
                    "t": np.asarray(t_rows, dtype=int),
                    "seed": int(current_seed),
                    "episode_id_within_track": int(episode_index),
                }
            )
            current_seed += 1

        episodes_by_track[track.value] = track_episodes

    return episodes_by_track


def flatten_split(
    episodes_by_track: Dict[str, List[Dict[str, Any]]],
    split_name: str,
) -> Dict[str, np.ndarray]:
    """Flatten per-track episodes into flat arrays with stable episode ids."""
    X_parts, y_parts, track_parts, episode_parts, timestep_parts = [], [], [], [], []

    for track_name, episodes in episodes_by_track.items():
        for episode in episodes:
            X_episode = np.asarray(episode["X"], dtype=float)
            y_episode = np.asarray(episode["y"], dtype=int)
            t_episode = np.asarray(episode["t"], dtype=int)
            episode_seed = int(episode["seed"])
            episode_index = int(episode["episode_id_within_track"])

            stable_episode_id = (
                f"{split_name}__{track_name}"
                f"__seed_{episode_seed}__episode_{episode_index:03d}"
            )
            n = len(y_episode)

            X_parts.append(X_episode)
            y_parts.append(y_episode)
            track_parts.append(np.full(n, track_name, dtype=object))
            episode_parts.append(np.full(n, stable_episode_id, dtype=object))
            timestep_parts.append(t_episode)

    return {
        "X": np.concatenate(X_parts),
        "y": np.concatenate(y_parts).astype(int),
        "track": np.concatenate(track_parts).astype(str),
        "episode_id": np.concatenate(episode_parts).astype(str),
        "timestep": np.concatenate(timestep_parts).astype(int),
    }
