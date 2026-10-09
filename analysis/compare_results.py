#!/usr/bin/env python3
"""
Compare entropy-constrained and baseline runs.
"""

import json
import os
import glob
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_dir):
    """Load final_research_metrics.json from a results directory"""
    json_path = os.path.join(results_dir, 'final_research_metrics.json')
    if not os.path.exists(json_path):
        return None

    with open(json_path, 'r') as f:
        return json.load(f)

def find_run_directories(base_dir='results'):
    """Find all result directories"""
    dirs = glob.glob(f"{base_dir}/*")
    return [d for d in dirs if os.path.isdir(d)]

def categorize_run(dirname):
    """Categorize run as baseline or constrained based on name"""
    dirname_lower = dirname.lower()
    if 'baseline' in dirname_lower or 'no_entropy' in dirname_lower:
        return 'baseline'
    elif 'entropy' in dirname_lower or 'constraint' in dirname_lower:
        return 'constrained'
    else:
        return 'unknown'

def compare_runs(baseline_results, constrained_results):
    """Compare two runs and generate report"""

    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS: Biological Constraints vs Baseline")
    print("="*80 + "\n")

    # Extract key metrics
    baseline_metrics = {
        'global_max_normalized': baseline_results.get('global_max_fitness_normalized', 0),
        'global_max_raw': baseline_results.get('global_max_fitness_raw', 0),
        'buffer_best': baseline_results.get('buffer_global_best', 0),
        'buffer_best_raw': baseline_results.get('buffer_global_best_raw', 0),
        'oracle_calls': baseline_results.get('total_oracle_calls', 0),
        'unique_sequences': baseline_results.get('total_unique_sequences', 0),
        'improvement': baseline_results.get('percentage_improvement', 0),
        'evicted': baseline_results.get('global_best_evicted', False)
    }

    constrained_metrics = {
        'global_max_normalized': constrained_results.get('global_max_fitness_normalized', 0),
        'global_max_raw': constrained_results.get('global_max_fitness_raw', 0),
        'buffer_best': constrained_results.get('buffer_global_best', 0),
        'buffer_best_raw': constrained_results.get('buffer_global_best_raw', 0),
        'oracle_calls': constrained_results.get('total_oracle_calls', 0),
        'unique_sequences': constrained_results.get('total_unique_sequences', 0),
        'improvement': constrained_results.get('percentage_improvement', 0),
        'evicted': constrained_results.get('global_best_evicted', False)
    }

    # Print comparison table
    print("┌" + "─"*78 + "┐")
    print("│ METRIC                              │ BASELINE      │ WITH CONSTRAINTS │ IMPROVEMENT │")
    print("├" + "─"*78 + "┤")

    # Global max fitness (normalized)
    baseline_norm = baseline_metrics['global_max_normalized']
    constrained_norm = constrained_metrics['global_max_normalized']
    improvement_norm = ((constrained_norm - baseline_norm) / baseline_norm * 100) if baseline_norm > 0 else 0
    print(f"│ Global Max Fitness (normalized)     │ {baseline_norm:13.6f} │ {constrained_norm:16.6f} │ {improvement_norm:10.2f}% │")

    # Global max fitness (raw)
    baseline_raw = baseline_metrics['global_max_raw'] or 0
    constrained_raw = constrained_metrics['global_max_raw'] or 0
    improvement_raw = ((constrained_raw - baseline_raw) / abs(baseline_raw) * 100) if baseline_raw != 0 else 0
    print(f"│ Global Max Fitness (raw)            │ {baseline_raw:13.6f} │ {constrained_raw:16.6f} │ {improvement_raw:10.2f}% │")

    # Buffer best
    baseline_buf = baseline_metrics['buffer_best']
    constrained_buf = constrained_metrics['buffer_best']
    improvement_buf = ((constrained_buf - baseline_buf) / baseline_buf * 100) if baseline_buf > 0 else 0
    print(f"│ Buffer Best (normalized)            │ {baseline_buf:13.6f} │ {constrained_buf:16.6f} │ {improvement_buf:10.2f}% │")

    # Oracle calls
    baseline_calls = baseline_metrics['oracle_calls']
    constrained_calls = constrained_metrics['oracle_calls']
    print(f"│ Total Oracle Calls                  │ {baseline_calls:13d} │ {constrained_calls:16d} │ {'':>11} │")

    # Unique sequences
    baseline_uniq = baseline_metrics['unique_sequences']
    constrained_uniq = constrained_metrics['unique_sequences']
    diversity_improvement = ((constrained_uniq - baseline_uniq) / baseline_uniq * 100) if baseline_uniq > 0 else 0
    print(f"│ Unique Sequences Discovered         │ {baseline_uniq:13d} │ {constrained_uniq:16d} │ {diversity_improvement:10.2f}% │")

    # Eviction status
    print(f"│ Global Best Evicted from Buffer?    │ {'Yes' if baseline_metrics['evicted'] else 'No':>13} │ {'Yes' if constrained_metrics['evicted'] else 'No':>16} │ {'':>11} │")

    print("└" + "─"*78 + "┘")

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    if improvement_norm > 0:
        print(f"Biological constraints improved normalized fitness by {improvement_norm:.2f}%")
    else:
        print(f"Biological constraints decreased normalized fitness by {abs(improvement_norm):.2f}%")

    if improvement_raw > 0:
        print(f"Biological constraints improved raw fitness by {improvement_raw:.2f}%")
    else:
        print(f"Biological constraints decreased raw fitness by {abs(improvement_raw):.2f}%")

    if diversity_improvement > 0:
        print(f"Biological constraints increased sequence diversity by {diversity_improvement:.2f}%")
    else:
        print(f"Biological constraints decreased sequence diversity by {abs(diversity_improvement):.2f}%")

    # Efficiency
    fitness_per_call_baseline = baseline_norm / baseline_calls if baseline_calls > 0 else 0
    fitness_per_call_constrained = constrained_norm / constrained_calls if constrained_calls > 0 else 0
    efficiency_improvement = ((fitness_per_call_constrained - fitness_per_call_baseline) / fitness_per_call_baseline * 100) if fitness_per_call_baseline > 0 else 0

    print(f"\nSample Efficiency:")
    print(f"  Baseline: {fitness_per_call_baseline:.8f} fitness per oracle call")
    print(f"  Constrained: {fitness_per_call_constrained:.8f} fitness per oracle call")
    print(f"  Improvement: {efficiency_improvement:.2f}%")

    # Best sequences
    print("\n" + "="*80)
    print("BEST SEQUENCES")
    print("="*80)
    print("\nBaseline Best Sequence:")
    print(f"  Normalized Fitness: {baseline_norm:.6f}")
    print(f"  Raw Fitness: {baseline_raw:.6f}")
    print(f"  Sequence: {baseline_results.get('global_best_sequence', 'N/A')[:80]}...")

    print("\nConstrained Best Sequence:")
    print(f"  Normalized Fitness: {constrained_norm:.6f}")
    print(f"  Raw Fitness: {constrained_raw:.6f}")
    print(f"  Sequence: {constrained_results.get('global_best_sequence', 'N/A')[:80]}...")

    # Top sequences comparison
    print("\n" + "="*80)
    print("TOP 10 SEQUENCES COMPARISON")
    print("="*80)

    baseline_top = baseline_results.get('top_20_sequences', [])[:10]
    constrained_top = constrained_results.get('top_20_sequences', [])[:10]

    print("\nBaseline Top 10:")
    for i, seq_info in enumerate(baseline_top, 1):
        print(f"  {i:2d}. Fitness: {seq_info['fitness']:.6f}")

    print("\nConstrained Top 10:")
    for i, seq_info in enumerate(constrained_top, 1):
        print(f"  {i:2d}. Fitness: {seq_info['fitness']:.6f}")

    print("\n" + "="*80)

    return {
        'fitness_improvement': improvement_norm,
        'raw_fitness_improvement': improvement_raw,
        'diversity_improvement': diversity_improvement,
        'efficiency_improvement': efficiency_improvement
    }

def main():
    parser = argparse.ArgumentParser(description='Compare protein optimization results')
    parser.add_argument('--baseline', type=str, help='Path to baseline results directory')
    parser.add_argument('--constrained', type=str, help='Path to constrained results directory')
    parser.add_argument('--auto', action='store_true', help='Auto-detect latest runs')
    args = parser.parse_args()

    if args.auto:
        # Auto-detect runs
        all_dirs = find_run_directories()
        baseline_dirs = [d for d in all_dirs if 'baseline' in d.lower() or 'no_entropy' in d.lower()]
        constrained_dirs = [d for d in all_dirs if ('entropy' in d.lower() or 'constraint' in d.lower()) and 'baseline' not in d.lower()]

        if not baseline_dirs or not constrained_dirs:
            print("Error: Could not auto-detect both baseline and constrained runs")
            print(f"Found {len(baseline_dirs)} baseline runs and {len(constrained_dirs)} constrained runs")
            return

        # Use most recent
        baseline_dir = sorted(baseline_dirs)[-1]
        constrained_dir = sorted(constrained_dirs)[-1]

        print(f"Auto-detected runs:")
        print(f"  Baseline: {baseline_dir}")
        print(f"  Constrained: {constrained_dir}")

    else:
        if not args.baseline or not args.constrained:
            print("Error: Must specify --baseline and --constrained paths, or use --auto")
            return

        baseline_dir = args.baseline
        constrained_dir = args.constrained

    # Load results
    baseline_results = load_results(baseline_dir)
    constrained_results = load_results(constrained_dir)

    if not baseline_results:
        print(f"Error: Could not load baseline results from {baseline_dir}")
        return

    if not constrained_results:
        print(f"Error: Could not load constrained results from {constrained_dir}")
        return

    # Compare
    summary = compare_runs(baseline_results, constrained_results)

    # Save summary
    summary_path = 'comparison_summary.json'
    with open(summary_path, 'w') as f:
        json.dump({
            'baseline_dir': baseline_dir,
            'constrained_dir': constrained_dir,
            'summary': summary,
            'baseline_metrics': baseline_results,
            'constrained_metrics': constrained_results
        }, f, indent=2)

    print(f"\nComparison summary saved to: {summary_path}")

if __name__ == '__main__':
    main()
