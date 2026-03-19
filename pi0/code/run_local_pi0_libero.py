from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from custom_pi0 import (
    Pi0CustomConfig,
    Pi0CustomModel,
    load_hf_weights_into_custom_model,
    resolve_hf_checkpoint_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Self-written pi0-style model + HF checkpoint loading + LIBERO rollout.\n"
            "No official LeRobot policy class is used."
        )
    )
    parser.add_argument("--policy-path", type=str, default="lerobot/pi0_libero_finetuned")
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="google/paligemma-3b-pt-224",
        help=(
            "Tokenizer source. openpi / lerobot pi0 use PaliGemma tokenizer."
        ),
    )
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--local-files-only", action="store_true")

    parser.add_argument("--suite", type=str, default="libero_object")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--camera-width", type=int, default=128)
    parser.add_argument("--camera-height", type=int, default=128)
    parser.add_argument("--instruction", type=str, default=None)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument(
        "--min-load-ratio",
        type=float,
        default=0.05,
        help="Minimum parameter match ratio required when loading HF weights.",
    )
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def make_libero_env(args: argparse.Namespace):
    """
    Build LIBERO environment from suite/task id.
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.suite not in benchmark_dict:
        raise ValueError(f"Unknown suite `{args.suite}`. Available: {sorted(benchmark_dict.keys())}")

    task_suite = benchmark_dict[args.suite]()
    task = task_suite.get_task(args.task_id)
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=args.camera_height,
        camera_widths=args.camera_width,
    )
    if hasattr(env, "seed"):
        env.seed(args.seed)
    env.reset()
    init_states = task_suite.get_task_init_states(args.task_id)
    return env, task_suite, task, init_states


def find_image_keys(obs: dict[str, Any]) -> list[str]:
    keys = []
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            if v.shape[-1] in (1, 3, 4) or v.shape[0] in (1, 3, 4):
                keys.append(k)
    return sorted(keys)


def find_state_key(obs: dict[str, Any]) -> str | None:
    priority = ["robot0_proprio-state", "proprio", "state", "robot_state"]
    for p in priority:
        for k, v in obs.items():
            if p in k.lower() and isinstance(v, np.ndarray) and v.ndim == 1:
                return k
    for k, v in obs.items():
        if isinstance(v, np.ndarray) and v.ndim == 1 and v.size > 3:
            return k
    return None


def image_to_tensor(x: np.ndarray, target_h: int, target_w: int) -> torch.Tensor:
    """
    Convert HWC or CHW image to float CHW in [0,1], then resize to target shape.
    """
    if x.shape[0] in (1, 3, 4) and x.shape[-1] not in (1, 3, 4):
        chw = x
    else:
        chw = np.transpose(x, (2, 0, 1))
    t = torch.from_numpy(chw).float()
    if t.max() > 1.0:
        t = t / 255.0
    if t.shape[0] > 3:
        t = t[:3]
    t = t.unsqueeze(0)
    t = F.interpolate(t, size=(target_h, target_w), mode="bilinear", align_corners=False)
    return t.squeeze(0)


def pad_or_trim_state(state: np.ndarray, target_dim: int) -> torch.Tensor:
    t = torch.from_numpy(state).float().view(-1)
    if t.numel() > target_dim:
        return t[:target_dim]
    if t.numel() < target_dim:
        return torch.cat([t, torch.zeros(target_dim - t.numel())], dim=0)
    return t


def build_model_inputs(
    obs: dict[str, Any],
    instruction: str,
    tokenizer,
    cfg: Pi0CustomConfig,
    device: torch.device,
    debug: bool = False,
):
    """
    Build tensors consumed by our self-written model.
    """
    image_obs_keys = find_image_keys(obs)
    if not image_obs_keys:
        raise ValueError("No image key found in LIBERO observation.")
    state_obs_key = find_state_key(obs)
    if state_obs_key is None:
        raise ValueError("No state key found in LIBERO observation.")

    cams = []
    for i in range(cfg.n_cams):
        src_key = image_obs_keys[min(i, len(image_obs_keys) - 1)]
        cam = image_to_tensor(obs[src_key], cfg.image_size, cfg.image_size)
        cams.append(cam)
        if debug:
            print(f"[map] obs[{src_key}] -> images[:, {i}] shape={tuple(cam.shape)}")
    images = torch.stack(cams, dim=0).unsqueeze(0).to(device)  # [1, N, 3, H, W]

    state = pad_or_trim_state(obs[state_obs_key], cfg.state_dim).unsqueeze(0).to(device)
    if debug:
        print(f"[map] obs[{state_obs_key}] -> state shape={tuple(state.shape)}")

    encoded = tokenizer(
        [instruction],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=cfg.max_text_len,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    if debug:
        print(f"[map] instruction -> input_ids shape={tuple(input_ids.shape)}")
        print(f"[map] instruction -> attention_mask shape={tuple(attention_mask.shape)}")

    return images, state, input_ids, attention_mask


def get_env_action_bounds_and_dim(env) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Return flattened action bounds and action dimension for LIBERO OffScreenRenderEnv.

    In huggingface/lerobot-libero, OffScreenRenderEnv is a wrapper with the underlying
    robosuite env stored at `env.env`. robosuite exposes action bounds via `action_spec`.
    """
    if not hasattr(env, "env"):
        raise AttributeError("Expected OffScreenRenderEnv wrapper to have `.env` attribute.")

    if not hasattr(env.env, "action_spec"):
        raise AttributeError("Expected underlying robosuite env to expose `.action_spec`.")

    low, high = env.env.action_spec
    low = np.asarray(low, dtype=np.float32).reshape(-1)
    high = np.asarray(high, dtype=np.float32).reshape(-1)
    if low.shape != high.shape:
        raise ValueError(
            f"Inconsistent action bounds shapes from env.env.action_spec: low={low.shape}, high={high.shape}"
        )
    return low, high, int(low.size)


def to_env_action(action: torch.Tensor, env) -> np.ndarray:
    a = action.detach().to("cpu").float().numpy().reshape(-1)
    low, high, env_dim = get_env_action_bounds_and_dim(env)
    if a.shape[0] > env_dim:
        a = a[:env_dim]
    elif a.shape[0] < env_dim:
        a = np.concatenate([a, np.zeros(env_dim - a.shape[0], dtype=np.float32)], axis=0)
    return np.clip(a, low, high).astype(np.float32)


def load_pi0_tokenizer(
    tokenizer_path: str,
    cache_dir: str | None,
    local_files_only: bool,
):
    """
    openpi / lerobot pi0 stack uses the PaliGemma tokenizer (slow path).
    """
    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            use_fast=False,
        )
    except Exception as e:
        msg = str(e).lower()
        if "sentencepiece" in msg:
            raise RuntimeError(
                "Failed to load PI0 tokenizer: sentencepiece is required for the slow PaliGemma tokenizer. "
                "Install `sentencepiece` in your Linux env and retry."
            ) from e
        raise


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)

    # 1) Build environment first to infer action/state dimensions.
    env, task_suite, task, init_states = make_libero_env(args)
    try:
        _, _, env_action_dim = get_env_action_bounds_and_dim(env)
        task_instruction = args.instruction or getattr(task, "language", "")

        # 2) Resolve checkpoint directory and tokenizer.
        ckpt_dir = resolve_hf_checkpoint_dir(
            args.policy_path,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
        tokenizer = load_pi0_tokenizer(
            tokenizer_path=args.tokenizer_path,
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )

        # 3) Infer state dim from the first observation.
        env.reset()
        init_state = init_states[args.init_state_id % len(init_states)]
        obs0 = env.set_init_state(init_state)
        if obs0 is None:
            obs0 = env.reset()
        state_obs_key = find_state_key(obs0)
        if state_obs_key is None:
            raise ValueError("Cannot infer state dimension from LIBERO observation.")
        inferred_state_dim = int(np.asarray(obs0[state_obs_key]).reshape(-1).shape[0])
        image_keys = find_image_keys(obs0)
        n_cams = min(2, max(1, len(image_keys)))

        # 4) Build our self-written model.
        cfg = Pi0CustomConfig(
            vocab_size=int(getattr(tokenizer, "vocab_size", len(tokenizer))),
            max_text_len=48,
            image_size=args.camera_height,
            n_cams=n_cams,
            state_dim=inferred_state_dim,
            action_dim=env_action_dim,
            action_horizon=args.action_horizon,
            hidden_dim=args.hidden_dim,
            cond_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            dropout=args.dropout,
            image_keys=[f"observation.images.cam{i}" for i in range(n_cams)],
        )
        model = Pi0CustomModel(cfg).to(device).eval()

        # 5) Load HF checkpoint tensors into our model (strictly through our own loader).
        report = load_hf_weights_into_custom_model(
            model=model,
            model_id_or_path=str(ckpt_dir),
            cache_dir=args.cache_dir,
            local_files_only=True,
            min_match_ratio_by_params=args.min_load_ratio,
        )
        print("[load-report]")
        print(
            f"  keys: {report.loaded_keys}/{report.total_model_keys} "
            f"({report.match_ratio_by_keys:.2%})"
        )
        print(
            f"  params: {report.loaded_params}/{report.total_model_params} "
            f"({report.match_ratio_by_params:.2%})"
        )
        print(f"  missing_keys(top10): {report.missing_keys[:10]}")
        print(f"  unexpected_keys(top10): {report.unexpected_keys[:10]}")

        # 6) Rollout.
        successes = 0
        for ep in range(args.episodes):
            env.reset()
            init_state = init_states[args.init_state_id % len(init_states)]
            obs = env.set_init_state(init_state)
            if obs is None:
                obs = env.reset()

            ep_return = 0.0
            done = False
            for step_idx in range(args.steps):
                images, state, input_ids, attention_mask = build_model_inputs(
                    obs=obs,
                    instruction=task_instruction,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    device=device,
                    debug=args.debug and ep == 0 and step_idx == 0,
                )
                with torch.inference_mode():
                    action = model.select_action(
                        images=images,
                        state=state,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        denoise_steps=args.denoise_steps,
                        trace=args.trace and ep == 0 and step_idx < 2,
                    )
                env_action = to_env_action(action[0], env)
                step_out = env.step(env_action)
                if len(step_out) == 5:
                    obs, reward, terminated, truncated, info = step_out
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, info = step_out
                ep_return += float(reward)
                if done:
                    break

            success = bool(ep_return > 0.0)
            successes += int(success)
            print(
                f"[episode {ep}] steps={step_idx+1} return={ep_return:.2f} "
                f"success={success} done={done}"
            )

        print(f"[summary] success_rate={successes}/{args.episodes}={(successes / max(1,args.episodes)):.2%}")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
