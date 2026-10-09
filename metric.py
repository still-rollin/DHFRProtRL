import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import itertools
import pandas as pd
from net.rew import BaseCNN, DHFRBaseCNN, DHFROracle
from utils.eval_utils import distance
from utils.constants import seq_to_one_hot


class OnehotDataset(Dataset):
    def __init__(self, seqs):
        self.seqs = seqs 

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        return seq_to_one_hot(self.seqs[index])


class Evaluator:
    def __init__(self, protein, max_target, min_target, device, batch_size=16):
        self.device = device
        self.batch_size = batch_size
        self.max_target, self.min_target = max_target, min_target

        # Load checkpoint first to determine architecture
        ckpt_path = f'ckpt/{protein}/oracle.ckpt'
        print(f"Loading oracle checkpoint from: {ckpt_path}")
        oracle_ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        # If wrapped in a dict
        if "state_dict" in oracle_ckpt.keys():
            state_dict = oracle_ckpt["state_dict"]
        elif "model_state_dict" in oracle_ckpt.keys():
            state_dict = oracle_ckpt["model_state_dict"]
        else:
            state_dict = oracle_ckpt

        # Choose model based on checkpoint keys
        if protein == 'DHFR':
            # Check if it's DHFROracle (has conv.0, conv.2, fc.0, fc.3 keys)
            # or DHFRBaseCNN (has encoder, embedding.*, decoder keys)
            if any(k.startswith('conv.0') for k in state_dict.keys()):
                # This is the DHFROracle architecture
                oracle = DHFROracle(make_one_hot=False)
                oracle.load_state_dict(state_dict, strict=True)
            else:
                # This is the DHFRBaseCNN architecture
                oracle = DHFRBaseCNN(make_one_hot=False)
                # Fix key mismatches if needed
                fixed_state_dict = {}
                for k, v in state_dict.items():
                    new_k = k
                    if "embedding.fc." in k:
                        new_k = k.replace("embedding.fc.", "embedding.layer.")
                    fixed_state_dict[new_k] = v
                oracle.load_state_dict(fixed_state_dict, strict=False)
        else:
            oracle = BaseCNN(make_one_hot=False)
            oracle.load_state_dict(state_dict, strict=False)

        print(f"Oracle model loaded successfully!")

        oracle.eval()
        self.oracle = oracle.to(device)

        # Load reference high-fitness sequences
        high = pd.read_csv(f'data/{protein}/all.csv')[['sequence', 'target']]
        high = high[high['target'] > high['target'].quantile(q=0.9).item()]
        self.high = high['sequence'].tolist()[:128]


    def evaluate(self, seqs, inits):
        dataset = OnehotDataset(seqs)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        targets = []
        with torch.no_grad():
            for batch in dataloader:
                _, target = self.oracle(batch.to(self.device), get_embed=True)
                target = (target - self.min_target) / (self.max_target - self.min_target)
                targets.extend(list(target.cpu().flatten()))
        fitness = np.median(targets)
        
        # Compute diversity
        distances = [distance(s1, s2) for s1, s2 in itertools.combinations(seqs, 2)]
        diversity = np.median(distances) if distances else 0.0
        
        # Compute novelty
        distances = []
        for j in seqs:
            dist_j = [distance(i, j) for i in inits]
            distances.append(min(dist_j))
        novelty = np.median(distances) if distances else 0.0
        
        # Compute proximity to high-fitness sequences
        distances = []
        for j in seqs:
            dist_j = [distance(i, j) for i in self.high]
            distances.append(min(dist_j))
        high = np.median(distances) if distances else 0.0

        return fitness, diversity, novelty, high
