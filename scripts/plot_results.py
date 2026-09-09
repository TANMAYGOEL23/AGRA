#!/usr/bin/env python3
"""
CLI Tool to generate comparison plots from saved simulation result JSON files.
"""
import os
import sys
import json
import argparse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.visualization.plot_generator import PaperPlotGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate plots from simulation results")
    parser.add_argument("--json-dir", type=str, default="results/json", help="Directory with saved benchmark JSONs")
    parser.add_argument("--output-dir", type=str, default="results/plots", help="Directory to save generated charts")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Reading result JSON files from: {args.json_dir}")
    print(f"Plots will be output to       : {args.output_dir}")

if __name__ == '__main__':
    main()
