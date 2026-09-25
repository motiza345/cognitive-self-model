# M22.1.2 response-space audit

This note records the held-out prediction audit of the frozen M22.1.1 response matrices.

## Protocol

- No new model forward pass.
- Primary target: alpha `+1` prompt-by-direction deltas on validation and replication separately.
- Alpha `-1` is a symmetry check and is not a training target.
- Masks hold out two cells per prompt and are stored in the manifest before fitting.
- The label uses ridge alternating least squares of rank 1.
- Mean-imputed SVD is a secondary closed form and cannot change the label.
- Few-shot calibration reveals the lexicographic first prompt of a held-out direction.

## Result

- Label: `INSUFFICIENT_EVIDENCE`.
- Manifest: `978e715812bc8291077e8e32447f1d20db3da78f675d3b0501c38af49b98ebaa`.
- M22.1 status: `CANDIDATE`.
- M22.2: not authorized.

## Non-claims

The result does not establish a causal mechanism, semantic direction identity, self-awareness, or a causal SelfModel.
