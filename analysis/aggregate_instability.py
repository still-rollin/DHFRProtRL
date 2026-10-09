
import os
import numpy as np
import pandas as pd
import yaml
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from utils.constants import REFSEQ

def calculate_instability(seq):
    try:
        return ProteinAnalysis(seq).instability_index()
    except:
        return np.nan

def calculate_violation_cost(seq, wt_seq, size_rules):
    if not size_rules:
        return 0.0
    cost = 0.0
    penalties = {'mild': 0.1, 'moderate': 0.3, 'severe': 0.5, 'no_cost': 0.0}
    for pos, (s1, s2) in enumerate(zip(wt_seq, seq)):
        if s1 != s2:
            rule = size_rules.get(s1, {})
            found = False
            for cat, aas in rule.items():
                if s2 in aas:
                    cost += penalties.get(cat, 0.0)
                    found = True
                    break
            if not found:
                cost += 0.2 # default
    return cost

def evaluate_folder(folder, version_name, wt_seq, size_rules):
    npy_path = os.path.join(folder, "4.npy")
    if not os.path.exists(npy_path):
        return None
    
    data = np.load(npy_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.ndim == 1:
        seqs = data.tolist()
    else:
        # data might be list of (seq, score)
        seqs = [x[0] if isinstance(x, (list, tuple)) else x for x in data]
    
    results = []
    # Take top 10 if possible
    # We don't have scores here easily unless we re-run oracle.
    # But usually 4.npy has the final filtered/top sequences.
    for i, seq in enumerate(seqs[:10]):
        instability = calculate_instability(seq)
        violation = calculate_violation_cost(seq, wt_seq, size_rules)
        mutations = sum(1 for a, b in zip(seq, wt_seq) if a != b)
        results.append({
            'Version': version_name,
            'Instability': instability,
            'Violation': violation,
            'Mutations': mutations
        })
    return results

def main():
    wt_seq = REFSEQ['DHFR']['medium']
    with open('size_rules.yaml', 'r') as f:
        size_rules = yaml.safe_load(f)
    
    # Existing runs we found earlier
    existing_runs = [
        ("v0_ablation", "results/DHFR_medium_0_dhfr_ablation_old_reward_01_58_27"),
        ("v1_new_reward_no_const", "results/DHFR_medium_2_dhfr_baseline_comparison_21_50_21"),
        ("v2_entropy_only", "results/DHFR_medium_0_with_entropy_constraints_05_26_04"),
        ("v3_entropy_size", "results/DHFR_medium_3_dhfr_size_constraint_test_23_24_55")
    ]
    
    all_results = []
    for name, folder in existing_runs:
        res = evaluate_folder(folder, name, wt_seq, size_rules)
        if res:
            all_results.extend(res)
    
    df = pd.DataFrame(all_results)
    if not df.empty:
        summary = df.groupby('Version').agg(['mean', 'std'])
        print("\n=== AGGREGATED RESULTS FROM PREVIOUS RUNS ===")
        print(summary)
        summary.to_csv("instability_aggregate_report.csv")
    else:
        print("No results found to aggregate.")

if __name__ == "__main__":
    main()
