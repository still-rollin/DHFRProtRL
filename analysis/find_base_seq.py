import pandas as pd
import numpy as np
import argparse

# Sequences from the "Without Entropy Data" (Rank 1 and 2)
# and "With Entropy Data" (Rank 1 and 2)
# They all seem to share a very similar base sequence.
# Let's read the dataset and find the sequence with normalized fitness ~ 0.311339

df = pd.read_csv('data/DHFR/medium.csv')
print(df.head())

target_fitness = 0.311339
# Find the row with fitness closest to target_fitness
df['diff'] = abs(df['target'] - target_fitness)
best_match = df.loc[df['diff'].idxmin()]

print("Closest match in dataset:")
print(f"Sequence: {best_match['sequence']}")
print(f"Target: {best_match['target']}")
print(f"Difference: {best_match['diff']}")

# Let's define the top sequences from the logs
no_ent_seq = "DIYAICACCKVESKNEGKKNEVFNNYTFRGLGNKGVLPWKCNSLDMKYFCAVTTYVNESKYEKLKYKRCKYLNKETVDNVNDMPNSKKLQNVVVMGRTSWESIPKKFKPLSNRINIILSRTLKKEDFDEDVYIINKVEDLIVLLGKLNYYKCFIIGGSVIYQEFLEKKLIKKIYFTRINSTYECDVFFPEINENEWQIISVSDVYTSNNTTLDFIIYKK"
ent_seq =    "DIYAICACCKVESKNEGKKNEVFNNYTFRGLGNKGVLPWKCNSLDMKYFCAVTTYVNESKYEKLKYKRCKYLNKETVDNVNDMPNSKKLQNVVVMGRTSWESIPKKFKPLSNRINIILSRTLKKEDFDEDVYIINKVEDLIVLLGKLNYYKCFIIGGSVVYQEFLEKKLIKKIYFTRINSTYECDVFFPEINENEWQIISVSDVYTSNNTTLDFIIYKK"

base_seq = best_match['sequence']

def get_mutations(base, mutant):
    muts = []
    for i, (b, m) in enumerate(zip(base, mutant)):
        if b != m:
            muts.append(f"{b}{i+1}{m}")
    return muts

print("\nMutations Without Entropy (Rank 1):")
print(get_mutations(base_seq, no_ent_seq))

print("\nMutations With Entropy (Rank 1):")
print(get_mutations(base_seq, ent_seq))
