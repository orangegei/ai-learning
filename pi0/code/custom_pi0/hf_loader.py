from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file as safetensors_load_file


@dataclass
class HFLoadReport:
    loaded_keys: int
    total_model_keys: int
    loaded_params: int
    total_model_params: int
    missing_keys: list[str]
    unexpected_keys: list[str]
    match_ratio_by_keys: float
    match_ratio_by_params: float


def resolve_hf_checkpoint_dir(
    model_id_or_path: str,
    cache_dir: str | None = None,
    local_files_only: bool = False,
) -> Path:
    """
    Resolve a local checkpoint directory:
    - if `model_id_or_path` is already a local dir -> return it
    - else download snapshot from HF Hub (unless local_files_only=True)
    """
    p = Path(model_id_or_path)
    if p.exists() and p.is_dir():
        return p

    if local_files_only:
        raise FileNotFoundError(
            f"`{model_id_or_path}` is not a local directory and local_files_only=True."
        )

    from huggingface_hub import snapshot_download

    local_path = snapshot_download(
        repo_id=model_id_or_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return Path(local_path)


def _load_state_dict_from_checkpoint_dir(ckpt_dir: Path) -> dict[str, torch.Tensor]:
    """
    Load all checkpoint tensors from a HF snapshot directory.
    Supports:
    - model.safetensors
    - sharded *.safetensors
    - pytorch_model.bin / *.pt
    """
    state_dict: dict[str, torch.Tensor] = {}

    # 1) Prefer safetensors (single file).
    single_st = ckpt_dir / "model.safetensors"
    if single_st.exists():
        state_dict.update(safetensors_load_file(str(single_st), device="cpu"))
        return state_dict

    # 2) Sharded safetensors.
    index_st = ckpt_dir / "model.safetensors.index.json"
    if index_st.exists():
        with index_st.open("r", encoding="utf-8") as f:
            index_obj = json.load(f)
        shard_files = sorted(set(index_obj.get("weight_map", {}).values()))
        for name in shard_files:
            shard = ckpt_dir / name
            if shard.exists():
                state_dict.update(safetensors_load_file(str(shard), device="cpu"))
        return state_dict

    # 3) Any safetensors in the snapshot root.
    safetensors_files = sorted(ckpt_dir.glob("*.safetensors"))
    if safetensors_files:
        for st_file in safetensors_files:
            state_dict.update(safetensors_load_file(str(st_file), device="cpu"))
        return state_dict

    # 4) PyTorch bin/pt fallback.
    pt_files = sorted(list(ckpt_dir.glob("*.bin")) + list(ckpt_dir.glob("*.pt")))
    for pt_file in pt_files:
        obj = torch.load(str(pt_file), map_location="cpu")
        if isinstance(obj, dict):
            if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                state_dict.update(obj["state_dict"])
            else:
                # assume already a state dict
                # keep only tensor values
                for k, v in obj.items():
                    if torch.is_tensor(v):
                        state_dict[k] = v
    if state_dict:
        return state_dict

    raise FileNotFoundError(f"No checkpoint tensors found in {ckpt_dir}")


def _token_overlap_score(a: str, b: str) -> int:
    ta = set(a.replace(".", "_").split("_"))
    tb = set(b.replace(".", "_").split("_"))
    return len(ta & tb)


def _normalize_candidate_keys(source_key: str) -> list[str]:
    """
    Try common prefixes from different training wrappers.
    """
    variants = [source_key]
    prefixes = ["model.", "policy.", "policy.model.", "module.", "actor."]
    for p in prefixes:
        if source_key.startswith(p):
            variants.append(source_key[len(p) :])
        variants.append(p + source_key)
    # Keep insertion order while deduplicating.
    seen = set()
    uniq = []
    for x in variants:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def load_hf_weights_into_custom_model(
    model: torch.nn.Module,
    model_id_or_path: str,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    min_match_ratio_by_params: float = 0.05,
) -> HFLoadReport:
    """
    Load HF checkpoint tensors into our self-written model.

    Matching strategy:
    1) exact key match (with common prefix aliases)
    2) same-shape + same tail token
    3) same-shape + best token-overlap score

    This loader is intentionally transparent for inspection:
    it returns explicit coverage stats and does NOT silently hide mismatch.
    """
    ckpt_dir = resolve_hf_checkpoint_dir(
        model_id_or_path=model_id_or_path,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    source_state = _load_state_dict_from_checkpoint_dir(ckpt_dir)
    target_state = model.state_dict()

    assigned: dict[str, torch.Tensor] = {}
    used_source_keys: set[str] = set()

    # Fast index by shape for fallback matching.
    shape_to_source_keys: dict[tuple[int, ...], list[str]] = {}
    for sk, sv in source_state.items():
        shape_to_source_keys.setdefault(tuple(sv.shape), []).append(sk)

    for tk, tv in target_state.items():
        # 1) exact + alias
        matched_key = None
        for candidate in _normalize_candidate_keys(tk):
            if candidate in source_state and tuple(source_state[candidate].shape) == tuple(tv.shape):
                matched_key = candidate
                break

        # 2/3) fallback by shape and lexical similarity
        if matched_key is None:
            candidates = shape_to_source_keys.get(tuple(tv.shape), [])
            if candidates:
                # Prefer same suffix token.
                tail = tk.split(".")[-1]
                tail_hits = [c for c in candidates if c.split(".")[-1] == tail and c not in used_source_keys]
                if len(tail_hits) == 1:
                    matched_key = tail_hits[0]
                else:
                    # Best overlap score among unused candidates.
                    scored = []
                    for c in candidates:
                        if c in used_source_keys:
                            continue
                        scored.append((_token_overlap_score(tk, c), c))
                    if scored:
                        scored.sort(key=lambda x: x[0], reverse=True)
                        if scored[0][0] > 0:
                            matched_key = scored[0][1]

        if matched_key is not None:
            source_tensor = source_state[matched_key]
            # Keep model dtype to avoid precision mismatch surprises.
            assigned[tk] = source_tensor.to(dtype=tv.dtype)
            used_source_keys.add(matched_key)

    missing_keys = [k for k in target_state.keys() if k not in assigned]
    unexpected_keys = [k for k in source_state.keys() if k not in used_source_keys]

    load_result = model.load_state_dict(assigned, strict=False)
    if hasattr(load_result, "missing_keys"):
        # use PyTorch's final missing keys (more accurate after load)
        missing_keys = list(load_result.missing_keys)
    if hasattr(load_result, "unexpected_keys"):
        # should be empty because strict=False with filtered state,
        # but keep here for completeness.
        unexpected_keys = list(load_result.unexpected_keys) + unexpected_keys

    total_model_params = sum(v.numel() for v in target_state.values())
    loaded_params = sum(assigned[k].numel() for k in assigned.keys())
    ratio_params = loaded_params / max(1, total_model_params)
    ratio_keys = len(assigned) / max(1, len(target_state))

    report = HFLoadReport(
        loaded_keys=len(assigned),
        total_model_keys=len(target_state),
        loaded_params=loaded_params,
        total_model_params=total_model_params,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        match_ratio_by_keys=ratio_keys,
        match_ratio_by_params=ratio_params,
    )

    if ratio_params < min_match_ratio_by_params:
        raise RuntimeError(
            "HF weights loaded, but effective match is too low. "
            f"matched params ratio={ratio_params:.4f} < threshold={min_match_ratio_by_params:.4f}. "
            "Likely architecture mismatch between checkpoint and custom model."
        )
    return report


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
