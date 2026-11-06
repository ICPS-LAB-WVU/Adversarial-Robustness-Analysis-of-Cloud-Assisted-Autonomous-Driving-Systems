#!/usr/bin/env python3
import json
import argparse
from pathlib import Path

CTRL_FILE = Path("/tmp/icc_attack_ctrl.json")

def write_ctrl(enable, mode="conf_suppress", eps=0.03, alpha=0.01, iters=2, target_cls=0):
    CTRL_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "enable": enable,
        "mode": mode,
        "eps": eps,
        "alpha": alpha,
        "iters": iters,
        "target_cls": target_cls
    }
    with open(CTRL_FILE, "w") as f:
        json.dump(data, f)
    print("[ICC CTRL]", data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--on", action="store_true")
    parser.add_argument("--off", action="store_true")
    parser.add_argument("--mode", default="conf_suppress")
    parser.add_argument("--eps", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--target_cls", type=int, default=0)
    parser.add_argument("--iters", type=int, default=2)

    args = parser.parse_args()

    if args.on:
        write_ctrl(
            True,
            args.mode,
            args.eps,
            args.alpha,
            args.iters,
            args.target_cls
        )
    elif args.off:
        write_ctrl(False)
    else:
        print("Usage: --on / --off")

