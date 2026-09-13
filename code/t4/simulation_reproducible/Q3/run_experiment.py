#!/usr/bin/env python3
"""问题三：300组全向干扰源随机案例实验。"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
from runner import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行问题三随机案例实验")
    parser.add_argument("--cases", type=int, default=300)
    parser.add_argument("--seed-start", type=int, default=2026091200)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    run_experiment(HERE, "Q3", args.cases, args.seed_start, args.workers)
