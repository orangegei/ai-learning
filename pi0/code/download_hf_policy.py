from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a HF policy snapshot to local directory."
    )
    parser.add_argument("--repo-id", type=str, default="lerobot/pi0_libero_finetuned")
    parser.add_argument("--local-dir", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    from huggingface_hub import snapshot_download

    args = parse_args()
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        revision=args.revision,
    )
    print(path)


if __name__ == "__main__":
    main()
