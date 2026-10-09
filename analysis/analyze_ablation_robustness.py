
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from net.rew import DHFROracle
from utils.constants import seq_to_one_hot
from utils.constants import REFSEQ

def calculate_advanced_properties(seq):
    try:
        analyser = ProteinAnalysis(seq)
        instability = analyser.instability_index()
        gravy = analyser.gravy()
        return {'instability': instability, 'gravy': gravy}
    except Exception as e:
        return None

def hamming_distance(s1, s2):
    return sum(el1 != el2 for el1, el2 in zip(s1, s2))

def get_oracle_scores(seqs, device='cuda:0'):
    print("Loading Oracle for re-scoring...")
    oracle = DHFROracle(make_one_hot=True)
    ckpt_path = 'ckpt/DHFR/oracle.ckpt'
    try:
        oracle_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "state_dict" in oracle_ckpt.keys():
            oracle_ckpt = oracle_ckpt["state_dict"]
        elif "model_state_dict" in oracle_ckpt.keys():
            oracle_ckpt = oracle_ckpt["model_state_dict"]
        oracle.load_state_dict(oracle_ckpt, strict=True)
        oracle.eval()
        oracle.to(device)
    except Exception as e:
        print(f"Failed to load oracle: {e}")
        return []

    ORACLE_AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
    oracle_aa_to_idx = {aa: i for i, aa in enumerate(ORACLE_AA_ALPHABET)}
    
    scores = []
    with torch.no_grad():
        for seq in seqs:
            try:
                tokens = torch.tensor([[oracle_aa_to_idx[aa] for aa in seq]], dtype=torch.long).to(device)
                pred = oracle(tokens).cpu().item()
                scores.append((seq, pred))
            except Exception as e:
                pass
    return scores

def analyze_ablation():
    print("Loading Ablation Results...")
    
    # WT Reference
    wt_seq = REFSEQ['DHFR']['medium']
    
    # 1. Load Ablation Sequences
    try:
        data = np.load('results/DHFR_medium_0_dhfr_ablation_old_reward_01_58_27/4.npy', allow_pickle=True)
        # Handle list of strings vs array
        if isinstance(data, np.ndarray) and data.ndim == 1:
             seqs = data.tolist()
        else:
             seqs = data # Assume list
             
        print(f"Loaded {len(seqs)} sequences. Re-scoring to find Top-10...")
        
        scored_seqs = get_oracle_scores(seqs)
        scored_seqs.sort(key=lambda x: x[1], reverse=True)
        
        top_10_ablation = [x[0] for x in scored_seqs[:10]]
        print(f"Top 1 Ablation Score: {scored_seqs[0][1]:.4f}")
        
    except Exception as e:
        print(f"Error loading: {e}")
        return

    # 2. Hardcoded "Constrained" Top-10 (New Method)
    constrained_top10 = [
        "DIYAICACCKVESKNEGKKNEVFNNYTFRGLGNKGVLPWKCNSLDMKYFCAVTTYVNESKYEKLKYKRCKYLNKETVDNVNDMPNSKKLQNVVVMGRTSWESIPKKFKPLSNRINIILSRTLKKEDFDEDVYIINKVEDLIVLLGKLNYYKCFIIGGSVVYQEFLEKKLIKKIYFTRINSTYECDVFFPEINENEWQIISVSDVYTSNNTTLDFIIYKK",
        "DIYAICACCKVESKNEGKKNEVFNNYTFRGLGNKGVLPWKCNSLDMKYFCAVTTYVNESKYEKLKYKRCKYLNKETVDNVNDMPNSKKLQNVVVMGRTSWESIPKNFKPLSNRINIILSRTLKKEDFDEDVYIINKVEDLIVLLGKLNYYKCFIIGGSVVYQEFLEKKLIKKIYFTRINSTYECDVFFPEINENEYQIISVSDVYTSNNTTLDFIIYKK",
        "DIYAICACCKVESKNEGKKNEVFNNYTFRGLGNKGVLPWKCNSLDMKYFCAVTTYVNESKYEKLKYKRCKYLNKETVDNVNDMPNSKKLQNVVVMGRTSWESIPKKFKPLSNRINIILSRTLKKEDFDEDVYIINKVEDLIVLLGKLNYYKCFIIGGSVVYQEFLEKKLIKKIYFTRINSTYECDVFFPEINENEWQIISVSDVYTSNNTTLDFIIYKK",
        "DIYAICACCKVESKNEGKKNEVFNNYTFRGLGNKGVLPWKCNSLDMKYFCAVTTYVNESKYEKLKYKRCKYLNKETVDNVNDMPNSKKLQNVVVMGRTSWESIPKKFKPLSNRINIILSRTLKKEDFDEDVYIINKVEDLIVLLGKLNYYKCFIIGGSVVYQEFLEKKLIKKIYFTRINSTYECDVFFPEINENEFQIISVSDVYTSNNTTLDFIIYKK",
        "DIYAICACCKVESKNEGKKNEVFNNYTFRGLGNKGVLPWKCNSLDMKYFCAVTTYVNESKYEKLKYKRCKYLNKETVDNVNDMPNSKKLQNVVVMGRTSWESIPKRFKPLSNRINIILSRTLKKEDFDEDVYIINKVEDLIVLLGKLNYYKCFIIGGSVVYQEFLEKKLIKKIYFTRINSTYECDVFFPEINENEYQIISVSDVYTSNNTTLDFIIYKK",
        "DIYAICACCKVESKNEGKKNEVFNNYTFRGLGNKGVLPWKCNSLDMKYFCAVTTYVNESKYEKLKYKRCKYLNKETVDNVNDMPNSKKLQNVVVMGRTSWESIPKNFKPLSNRINVILSRTLKKEDFDEDVYIINKVEDLIVLLGKLNYYKCFIIGGSVVYQEFLEKKLIKKIYFTRINSTYECDVFFPEINENEYQIISVSDVYTSNNTTLDFIIYKK",
        "DIYAICACCKVESKNEGKKNEVFNNYTFRGLGNKGVLPWKCNSLDMKYFCAVTTYVNESKYEKLKYKRCKYLNKETVDNVNDMPNSKKLQNVVIMGRTSWESIPKQFKPLSNRINIILSRTLKKEDFDEDVYIINKVEDLIVLLGKLNYYKCFIIGGSVVYQEFLEKKLIKKIYFTRINSTYECDVFFPEINENEYQIISVSDVYTSNNTTLDFIIYKK",
        "DIYAICACCKVESKNEGKKNEVFNNYTFRGLGNKGVLPWKCNSLDMKYFCAVTTYVNESKYEKLKYKRCKYLNKETVDNVNDMPNSKKLQNVVVMGRTSWESIPKKFKPLSNRINIILSRTLKKEDFDEDVYIINKVEDLIVLLGKLNYYKCFIIGGSVVYQEFLEKKLIKKIYFTRINSTYECDVFFPEINENEYQIISVSDVYTSNNTTLDFIIYKK",
        "DIYAICACCKVESKNEGKKNEVFNNYTFRGLGNKGVLPWKCNSLDMKYFCAVTTYVNESKYEKLKYKRCKYLNKETVDNVNDMPNSKKLQNVVVMGRTSWESIPKKFKPLSNRINVILSRTLKKEDFDEDVYIINKVEDLIVLLGKLNYYKCFIIGGSVVYQEFLEKKLIKKIYFTRINSTYECDVFFPEINENEWQIISVSDVYTSNNTTLDFIIYKK",
        "DIYAICACCKVESKNEGKKNEVFNNYTFRGLGNKGVLPWKCNSLDMKYFCAVTTYVNESKYEKLKYKRCKYLNKETVDNVNDMPNSKKLQNVVVMGRTSWESIPKKFKPLSNRINVILSRTLKKEDFDEDVYIINKVEDLIVLLGKLNYYKCFIIGGSVVYQEFLEKKLIKKIYFTRINSTYECDVFFPEINENEFQIISVSDVYTSNNTTLDFIIYKK"
    ]

    # 3. Analyze
    ablation_res = []
    for s in top_10_ablation:
        props = calculate_advanced_properties(s)
        props['n_mut'] = hamming_distance(wt_seq, s)
        ablation_res.append(props)
        
    const_res = []
    for s in constrained_top10:
        props = calculate_advanced_properties(s)
        props['n_mut'] = hamming_distance(wt_seq, s)
        const_res.append(props)
    
    ablation_res = [r for r in ablation_res if r]
    const_res = [r for r in const_res if r]

    # 4. DataFrame
    df_ablation = pd.DataFrame(ablation_res)
    df_ablation['Method'] = 'Ablation (Old Reward)'
    
    df_const = pd.DataFrame(const_res)
    df_const['Method'] = 'LatProtRL (New + Constrained)'
    
    combined = pd.concat([df_ablation, df_const])
    
    # 5. Stats
    print("\n=== ROBUSTNESS & EXPLORATION COMPARISON ===")
    print(combined.groupby('Method')[['instability', 'gravy', 'n_mut']].agg(['mean', 'std']))
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.boxplot(x='Method', y='instability', data=combined, palette=['gray', 'blue'], ax=axes[0])
    axes[0].set_title('Instability (Lower is Better)')
    axes[0].axhline(y=40, color='k', linestyle='--', alpha=0.5)
    
    sns.boxplot(x='Method', y='n_mut', data=combined, palette=['gray', 'blue'], ax=axes[1])
    axes[1].set_title('Mutations from WT (Higher = More Exploration)')
    
    sns.boxplot(x='Method', y='gravy', data=combined, palette=['gray', 'blue'], ax=axes[2])
    axes[2].set_title('GRAVY (Solubility)')
    
    plt.tight_layout()
    plt.savefig('ablation_full_comparison.png')
    
    print("\nFull comparison plot saved: ablation_full_comparison.png")

if __name__ == "__main__":
    analyze_ablation()
