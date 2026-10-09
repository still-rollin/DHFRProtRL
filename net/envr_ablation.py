
import torch 
import warnings
import numpy as np
import pandas as pd
import yaml
from gymnasium import Env, spaces 
from utils.constants import REFSEQ, ALPHABET, seq_to_one_hot
from utils.eval_utils import distance
from net.buffers import *
from net.rew import BaseCNN, DHFROracle
from net.seq_lm import VED
from config import * 

warnings.filterwarnings('ignore')

# COPY OF SingleOpt but with Reference Reward Function Logic
class BaselineRewardOpt(Env):
    def __init__(self, config, seed=422):
        super().__init__()
        seq_cfg = create_rep_from_opt(config)
        obs_shape = (seq_cfg.reduce_dim, )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=-config.action_size, high=config.action_size, shape=obs_shape, dtype=np.float32)

        self.device = seq_cfg.device
        self.protein = config.name
        self.length = config.length
        self.done_cond = config.done_cond
        self.oracle_calls = 0
        self.config = config

        # No Constraints loaded here

        # Load oracle model based on protein type
        is_dhfr_oracle = (config.name == 'DHFR' and 'oracle.ckpt' in config.rew_pretrained)

        if is_dhfr_oracle:
            oracle = DHFROracle(make_one_hot=True)
        else:
            oracle = BaseCNN(make_one_hot=False)

        oracle_ckpt = torch.load(config.rew_pretrained, map_location=self.device, weights_only=False)
        if "state_dict" in oracle_ckpt.keys():
            oracle_ckpt = oracle_ckpt["state_dict"]
        elif "model_state_dict" in oracle_ckpt.keys():
            oracle_ckpt = oracle_ckpt["model_state_dict"]

        if is_dhfr_oracle:
            oracle.load_state_dict(oracle_ckpt, strict=True)
        else:
            oracle.load_state_dict({k.replace('predictor.', ''): v for k, v in oracle_ckpt.items()}, strict=False)

        oracle.eval()
        self.oracle = oracle.to(self.device)

        # Load data
        data = pd.read_csv('data/{}/{}.csv'.format(config.name, config.level))[["sequence", "target"]]

        # Calibrate DHFROracle predictions to match target scale
        if is_dhfr_oracle:
            print("(Ablation) Calibrating DHFROracle predictions...")
            ORACLE_AA_ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
            self.oracle_aa_to_idx = {aa: i for i, aa in enumerate(ORACLE_AA_ALPHABET)}

            # Sample sequences for calibration
            sample_size = min(1000, len(data))
            sample_indices = np.random.choice(len(data), sample_size, replace=False)
            sample_seqs = data["sequence"].iloc[sample_indices].tolist()
            sample_targets = data["target"].iloc[sample_indices].values

            # Get oracle predictions
            with torch.no_grad():
                sample_preds = []
                batch_size = 128
                for i in range(0, len(sample_seqs), batch_size):
                    batch_seqs = sample_seqs[i:i+batch_size]
                    batch_tokens = torch.tensor([[self.oracle_aa_to_idx[aa] for aa in seq] for seq in batch_seqs], dtype=torch.long).to(self.device)
                    preds = self.oracle(batch_tokens).cpu().numpy()
                    sample_preds.extend(preds.flatten().tolist())
                sample_preds = np.array(sample_preds)

            # Fit linear calibration: target = slope * pred + intercept
            from scipy.stats import linregress
            slope, intercept, _, _, _ = linregress(sample_preds, sample_targets)

            self.oracle_scale = slope
            self.oracle_shift = intercept
        else:
            self.oracle_scale = 1.0
            self.oracle_shift = 0.0

        # Normalization
        fitness_range = self.config.max_fitness - self.config.min_fitness
        data["target"] = (data["target"] - self.config.min_fitness) / fitness_range

        # Buffer
        start_percentile = getattr(self.config, 'start_percentile', 95)
        self.buffer = Buffer(
            list(data.itertuples(index=False, name=None)),
            random=random.Random(seed),
            start_percentile=start_percentile
        )
        self.inits = data["sequence"].tolist()

        # Model
        model = VED(seq_cfg, pretrained=config.seq_pretrained)
        model.eval()
        self.model = model.to(self.device)
        self.model.set_wt_tokens(REFSEQ[self.protein][self.config.level])
        
        self.ep = 0
        self.steps = 0
        self.total_steps = 0
        self.state = None 
        self.state_seq = None 
        self.wt_seq = REFSEQ[self.protein][self.config.level]
        self.init_target = 0
        self.buffer_idx = None

        # logging
        self.reward = 0
        self.done = False
        self.target = 0
        self.best_discovered = 0
        self.init_seq = None
        self.n_mut = 0 
        self.aa = {a:0 for a in ALPHABET}
        self.pos = {a:0 for a in range(self.length)}

    def normalize_target(self, target):
        fitness_range = self.config.max_fitness - self.config.min_fitness
        normalized = (target - self.config.min_fitness) / fitness_range
        return max(0.0, min(1.0, normalized))

    def get_oracle_prediction(self, seq):
        with torch.no_grad():
            is_dhfr_oracle = hasattr(self, 'oracle_aa_to_idx')

            if is_dhfr_oracle:
                tokens = torch.tensor([[self.oracle_aa_to_idx[aa] for aa in seq]], dtype=torch.long).to(self.device)
                pred = self.oracle(tokens).cpu().item()
                pred = pred * self.oracle_scale + self.oracle_shift
            else:
                _, pred = self.oracle(seq_to_one_hot(seq).unsqueeze(0).to(self.device), get_embed=True)
                pred = pred.cpu().item()
            return pred

    def reset(self, seed=422):
        self.state_seq, self.buffer_idx = self.buffer.top()
        self.init_seq = self.state_seq
        with torch.no_grad():
            state = self.model.encode(self.state_seq)
            self.state = state.cpu().view(-1)
            target = self.get_oracle_prediction(self.state_seq)
            self.init_target = self.target = self.normalize_target(target)

        # Baseline logic doesn't use improvement, so we don't strictly need episode_best tracked in the same way,
        # but good for stats.
        self.episode_best = self.init_target
        
        self.ep += 1
        self.steps = 0
        self.total_steps += 1
        return self.state, {}
    
    def step(self, action):
        self.steps += 1
        self.total_steps += 1

        # --- Continuous Action (Latent) -> Sequence ---
        # The user provided discrete logic (aa, pos) but explicitly asked to test "current architecture" (latent)
        # on "old reward function" (absolute target). 
        # So we keep the Latent decoding step.
        action_tensor = torch.tensor(action)
        next_state = (action_tensor + self.state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            next_seq = self.model.decode(
                next_state, to_seq=True, template=self.state_seq, topk=self.config.topk
            )
            
            # Helper for stats
            step_mut = distance(self.state_seq, next_seq)
            n_mut = distance(self.wt_seq, next_seq)
            
            # --- Mutation Stats logic from user snippet ---
            # User snippet:
            # aa = int(action[0]), pos = int(action[1]) ... 
            # self.aa[IDXTOAA[aa]] += 1 ...
            # Per-residue statistics are not tracked in this ablation.

        # --- Done Logic ---
        # Latent done conditions
        done = (
            step_mut > self.done_cond.step_mut
            or self.steps > self.done_cond.max_steps
            or n_mut > self.done_cond.max_mutation
        )

        reward = 0.0
        called = False
        
        # --- REWARD LOGIC (ABLATION TARGET) ---
        # "if self.config.not_sparse or done:"
        # "self.reward = target"
        
        # We will assume not_sparse=True for this run to match the "dense" feedback implied by "LatProtRL" typically, 
        # or we check the config. The user snippet had `if self.config.not_sparse or done`.
        # We will follow that.
        
        should_calc_reward = getattr(self.config, 'not_sparse', False) or done
        
        # If the step is invalid (step_mut too high), normally we fail.
        # But let's follow the user logic: if done (even by fail), we calculate reward?
        # User snippet: done = step_mut==0 (discrete fail) or limits.
        # Then if done: calc reward.
        
        if should_calc_reward:
            self.oracle_calls += 1
            raw_target = self.get_oracle_prediction(next_seq)
            target = self.normalize_target(raw_target)
            
            # THE CORE CHANGE: Absolute Reward, no relative improvement, no constraints
            reward = target
            
            self.reward = reward 
            self.target = target
            self.buffer.push((next_seq, target, self.buffer_idx))
            
            if target > self.best_discovered:
                self.best_discovered = target
            
            called = True
        else:
            reward = 0.0
            self.reward = 0.0

        # Update state
        self.state = next_state.cpu().numpy()[0]
        self.state_seq = next_seq

        if done:
            info = {
                'candidates': self.state_seq, 
                'fitness': self.target,
                'init_seq': self.init_seq,
                'n_mut': n_mut,
                'aa': self.aa,
                'pos': self.pos,
                'called': called
            }
            return self.state, reward, True, False, info
        
        return self.state, reward, False, False, {}
