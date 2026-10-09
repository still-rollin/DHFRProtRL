import torch
import glob
import os
import pandas as pd

# Find the latest policy dir
base_dir = "policy/DHFR_medium_101_final_full_model_run_13_41_05"
buffer_files = glob.glob(os.path.join(base_dir, "buffer_*.pth"))
if not buffer_files:
    print("No buffer found!")
else:
    buffer_file = sorted(buffer_files)[-1]
    buf = torch.load(buffer_file)
    # The buffer holds (sequence, score, index) entries.
    # print top sequences
    if hasattr(buf, 'buffer'):
        items = buf.buffer
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        top = sorted_items[0]
        seq = top[0]
        score = top[1]
        print(f"Top Score: {score}")
        print(f"Sequence: {seq}")
        
        # Now find base sequence logic again
        df = pd.read_csv('data/DHFR/medium.csv')
        base_match = df.loc[abs(df['target'] - 0.311339).idxmin()]
        base_seq = base_match['sequence']
        
        muts = []
        for i, (b, m) in enumerate(zip(base_seq, seq)):
            if b != m:
                muts.append(f"{b}{i+1}{m}")
        print("Mutations from base:", muts)
    else:
        print("Buffer structure unknown:", type(buf))
