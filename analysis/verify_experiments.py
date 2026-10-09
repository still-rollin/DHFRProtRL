
import os
import time
import subprocess
import argparse
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def run_experiment(script, tag, extra_args=[]):
    cmd = [
        "python", script,
        "--protein", "DHFR",
        "--level", "medium",
        "--run", "0",
        "--device", "cuda:0",
        "--tag", tag,
        "--max_step", "5" # Keep it short for verification
    ] + extra_args
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    # 1. Run all 4 versions
    # We will use a smaller number of total_timesteps for quick verification
    # Actually, modify the scripts to take a --timesteps arg or just use 4096 default?
    # I'll modify the scripts slightly or just run them as they are but for fewer timesteps?
    # I'll just run them for 4096 steps which is about 2 mins each.
    
    versions = [
        ("optimize_ablation.py", "v0_orig_reward", []),
        ("optimize_baseline.py", "v1_new_reward_no_const", []),
        ("optimize.py", "v2_entropy_only", ["--no_size"]), # I'll add this flag to optimize.py
        ("optimize.py", "v3_entropy_size", [])
    ]
    
    # I need to update optimize.py to handle --no_size
    
    print("This will run 4 short experiments to verify the trends.")
    # for script, tag, extra in versions:
    #     run_experiment(script, tag, extra)

if __name__ == "__main__":
    main()
