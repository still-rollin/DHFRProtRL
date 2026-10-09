import torch
import glob
import os

base_dir = "policy/DHFR_medium_101_final_full_model_run_13_41_05"
buffer_files = glob.glob(os.path.join(base_dir, "buffer_*.pth"))
if not buffer_files:
    print("No buffer found!")
else:
    buffer_file = sorted(buffer_files)[-1]
    buf = torch.load(buffer_file)
    
    if hasattr(buf, 'buffer'):
        items = buf.buffer
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        
        base_seq = "DIYAICACCKVESKNEGKKNEVFNNYTFRGLGNKGVLPWKCNSLDMKYFCAVTTYVIESKYEKLKYKRCKYLNKETVDNVNDMPNSKKLQNVVVMGRTSWESIPKKFKPLSNRINVILSRTLKKEDFDEDVYIINKVEDLIVLLGKLNYYKCFIIGGSVVYQEFLEKKLIKKIYFTRINSTYECDVFFPEINENEYQIISVSDVYTSNNTTLDFIIYKK"
        
        print("TOP 5 FINAL SEQUENCES:")
        for i in range(min(5, len(sorted_items))):
            seq = sorted_items[i][0]
            score = sorted_items[i][1]
            
            muts = []
            for idx, (b, m) in enumerate(zip(base_seq, seq)):
                if b != m:
                    muts.append(f"{b}{idx+1}{m}")
                    
            print(f"Rank {i+1}: Fitness = {score:.6f}")
            print(f"Mutations: {muts}")
    else:
        print("Buffer missing property 'buffer'")
