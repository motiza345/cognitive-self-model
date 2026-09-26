# M22.1.3 preflight

Facts only. No delta prediction, correlation, or label was computed. No intervention forward was run.

## Git

- Starting branch: `cursor/m22-1-2-response-space-audit-c559`
- Starting commit: `63699b7ec91f4912f6c2b464dcd33ae4a91a1e1b`
- Working tree before this preflight: clean
- `main` is `c08f0e6` and does not contain the M22.1.2 commit, so that branch is not merged
- This preflight branch: `m22-1-3-readout-null`, created from the M22.1.2 branch

## Control plane

`STATE`, `MAP` / `FILE_MAP`, and `CLAIMS` are not in the repository. `README.md` names `docs/` and `artifacts/`, and `docs/` is empty. Those control-plane values are UNKNOWN.

## Model

`Qwen/Qwen2.5-0.5B`, revision `060db6499f32faf8b98477b0a26969ef7d8b9987`.

Recorded architecture `Qwen2.5-0.5B`, `n_layers = 24`, `d_model = 896` (`reports/M22_1/manifest.json` lines 31–39 and `reports/M22_1_1/direction_panel_results.json` lines 3–10). The same sizes are in the local Hugging Face `config.json` (`num_hidden_layers` 24, `hidden_size` 896).

Layer 23 is the final block because `24 - 1 = 23`. M22.1 recorded `selected_layer` 23 (`reports/M22_1/intervention_preflight_results.json` line 1858).

## Final norm

The git repository does not define the final norm. The local checkpoint and installed libraries do:

- Class: `Qwen2RMSNorm` (`transformers/models/qwen2/modeling_qwen2.py` lines 237–252)
- Hugging Face module: `model.norm` (same file, line 331)
- Epsilon: `1e-06` (snapshot `config.json` line 18)
- Learnable gain: `self.weight`, an `nn.Parameter`
- Learnable bias: none in that class
- TransformerLens name: `ln_final.w`, copied from `qwen.model.norm.weight` (`transformer_lens/pretrained/weight_conversions/qwen2.py` line 71). That converter does not set `ln_final.b`.

## Loading flags

M22.1 loads with:

```python
load_kwargs = {"device": device, "dtype": dtype}
# revision is added only when the hub sha and the constructor allow it
HookedTransformer.from_pretrained(model_id, **load_kwargs)
```

`src/cognitive_self_model/m22_1/loader.py` lines 78–93. `device` is CUDA when available, otherwise CPU. The saved run used CPU. `dtype` is forced to `float32`.

M22.1.1 loads with:

```python
HookedTransformer.from_pretrained(
    str(config["model_id"]),
    device="cpu",
    dtype=torch.float32,
    revision=revision,
)
```

`src/cognitive_self_model/m22_1_1/run_audit.py` lines 43–48.

Neither call passes `fold_ln`, `center_writing_weights`, `center_unembed`, or `fold_value_biases`. In TransformerLens 3.9.0 those defaults are all `True` (`HookedTransformer.py` lines 1162–1174). The M22 artifacts do not record the flags that were actually applied.

If that default stays true, `weight_processing.py` lines 1008–1034 multiply the unembedding by `ln_final.w` and then replace `ln_final.w` with ones. This preflight did not reload the model to observe the folded weights.

The checkpoint config says `tie_word_embeddings: true` and `torch_dtype: bfloat16` (`config.json` lines 21–22). The intervention runtime recorded `float32`.

## Tokens and margin

Tokenizer decode of the local snapshot, with no model forward:

- id `9834` decodes to ` yes` and the piece `Ġyes`
- id `902` decodes to ` no` and the piece `Ġno`
- `encode(" yes")` is `[9834]` and `encode(" no")` is `[902]`

The saved M22.1 outcome uses those same strings (`reports/M22_1/manifest.json` lines 41–58).

The margin is the last-position logit difference:

```python
last = logits[0, -1]
value = last[int(positive_id)] - last[int(negative_id)]
```

`src/cognitive_self_model/m22_1/outcome.py` lines 42–45.

## Hook

Exact hook: `blocks.23.hook_resid_post`. The template is `blocks.{layer}.hook_resid_post` (`configs/m22_1_preflight.json` lines 5–6).

The addition is last-token only:

```python
step = float(alpha) * direction.to(device=updated.device, dtype=updated.dtype)
updated[:, -1, :] = updated[:, -1, :] + step
```

`src/cognitive_self_model/m22_1/intervention.py` lines 29–30. Alpha `0` returns the residual unchanged (lines 26–27). The direction is cast to float32 before that multiply (`loader.py` lines 187–189).

## Directions

No direction vector file exists. A repository search found no `.npy`, `.npz`, `.pt`, or `.safetensors` files. Each vector is identified by the sha256 of its contiguous float64 bytes (`direction.py` lines 20–22). The recorded norms are 1, or `0.9999999999999999` where the JSON float is not exactly 1. The recorded dimension is 896.

| id | seed | role | norm recorded | content sha256 |
| --- | ---: | --- | ---: | --- |
| D1 | 22101 | M22.1 primary | 1.0 | `5aad7f993311efc1bfcd3fa14141b4b5403d88562c66ec6717770bf70c91e411` |
| D2 | 22103 | M22.1 control | 0.9999999999999999 | `8078ed36ed147c57c2bd723a9523f1b6dc30a6bc6895b34fc142ed8d1c0b400e` |
| D3 | 22111 | new | 1.0 | `546083c4b5c7779725a9544d7fdf4da8d4770b50826d2274b9c566219d4d309d` |
| D4 | 22112 | new | 0.9999999999999999 | `9b2f1b8831336b2b871b7d9d4587adbc427eb4a66434b77073d8c1e2764bd911` |
| D5 | 22113 | new | 1.0 | `10579f559390a7ebf5b25965b8c3487c3dedd4cb8295d020fc74f5beec736f43` |
| D6 | 22114 | new | 0.9999999999999999 | `9aa41426dd3eca94f25ca54e67e6e23555bf1afbcf81240736c0ae411d376ee3` |
| D7 | 22115 | new | 1.0 | `142c1f5f44d36430d89371a231aabe825a116affd847a7fafad80ced72b26c45` |
| D8 | 22116 | new | 1.0 | `6752b59f8f74c2bacb818deac69840ad744144f6dca41b3a4b58a0f59ccc7d60` |

D1 matches the M22.1 primary hash. D2 matches the M22.1 control hash. File sha256 for each vector is UNKNOWN because there is no file. The manifest file itself hashes to `c8b85bffa1ffeec2a86a7911c79c548bf1816623bc9c4b3bd11b0609325753bf`.

Alpha scales the unit vector: the added step is `alpha` times the float32 vector.

## Clean activations

Absent. Path, shape, dtype, and prompt coverage are UNKNOWN.

## Recorded delta table

The table M22.1.2 reads is `reports/M22_1_1/direction_panel_results.json`, key `records`.

- 720 records
- 18 prompts, 8 directions, alphas `-2, -1, 0, +1, +2`
- splits: discovery 240, validation 240, replication 240
- file sha256: `fe167fc92546c5949c5a298065c59d1488371129f977ad14e6750049648a26a1`

M22.1.2 also stores derived matrices in `reports/m22_1_2_response_space_results.json`. Each of discovery, validation, and replication has a `6 x 8` matrix at alpha `+1` and a `6 x 8` matrix at alpha `-1`.

## Anomaly 0.003849

The only printed occurrences are `reports/m22_1_2_response_space_report.md` lines 26–27. Both rank-2 and rank-3 show replication MAE `0.003849`, and both show validation MAE `0.003629`.

The stored means are not equal:

- replication rank-2: `0.003848899355113099`
- replication rank-3: `0.0038488960769474468`
- validation rank-2: `0.0036288875425197807`
- validation rank-3: `0.0036288875956987285`

Six-decimal formatting (`protocol.py` line 435) prints both replication means as `0.003849` and both validation means as `0.003629`. Per-seed rank-2 and rank-3 values are also unequal. The two models call `predict_ridge_als` with rank 2 and rank 3 (`protocol.py` lines 42–48). This is a display collision, not a reused array or a copied cache.

## BLOCKERS

- `STATE`, `MAP` / `FILE_MAP`, and `CLAIMS` are absent.
- D1–D8 are not saved as vectors. Only seeds and content hashes are saved.
- Clean `blocks.23.hook_resid_post` activations are absent.
- The repository never records whether `fold_ln`, `center_writing_weights`, and `center_unembed` stayed at the TransformerLens default `True`.
- The checkpoint declares bfloat16 and the intervention runtime recorded float32.
- The final-norm class and `ln_final` mapping are outside the git tree. They were read from the local Hugging Face snapshot and the installed libraries.
