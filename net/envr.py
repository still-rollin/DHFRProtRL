import torch 
import argparse
torch.serialization.add_safe_globals([argparse.Namespace])
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

class SingleOpt(Env):
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
        self.blocked_positions = {32, 38, 39, 95, 104}

        self.position_entropy = position_entropy = {
    32: 0,
    38: 0,
    39: 0,
    95: 0,
    104: 0,
    30: 0.03188153,
    185: 0.033457278,
    67: 0.044819624,
    115: 0.056700133,
    156: 0.056700133,
    176: 0.056700133,
    91: 0.088529733,
    96: 0.088529733,
    109: 0.088529733,
    110: 0.088529733,
    71: 0.089580528,
    100: 0.099222085,
    68: 0.1334398,
    157: 0.135380162,
    161: 0.152110298,
    37: 0.168050782,
    102: 0.183860992,
    113: 0.220859264,
    53: 0.232367524,
    153: 0.252529806,
    119: 0.261035916,
    49: 0.263728167,
    144: 0.293979915,
    46: 0.298119658,
    188: 0.298584037,
    97: 0.298904265,
    103: 0.304658079,
    93: 0.310468849,
    165: 0.334089401,
    72: 0.360569712,
    184: 0.372844444,
    177: 0.379319024,
    35: 0.387005186,
    48: 0.460441134,
    45: 0.467834244,
    52: 0.467987383,
    155: 0.48222263,
    178: 0.505584212,
    189: 0.53366769,
    4: 0.562952514,
    8: 0.572242043,
    75: 0.573719253,
    2: 0.614021739,
    6: 0.62214994,
    11: 0.62396087,
    214: 0.626968261,
    5: 0.635356996,
    154: 0.645540994,
    116: 0.664395085,
    7: 0.679024865,
    1: 0.689340582,
    31: 0.696225338,
    108: 0.745850863,
    65: 0.761909059,
    66: 0.770694807,
    187: 0.780922616,
    40: 0.781574493,
    201: 0.793609991,
    64: 0.799638313,
    173: 0.806707304,
    70: 0.808570926,
    10: 0.816743288,
    9: 0.8206192,
    121: 0.829600467,
    92: 0.832708094,
    117: 0.833044278,
    3: 0.856803704,
    140: 0.861555734,
    118: 0.865909634,
    174: 0.882051927,
    56: 0.8939942,
    101: 0.901695903,
    219: 0.921518751,
    160: 0.930426542,
    77: 0.946343117,
    94: 0.950698416,
    151: 0.96909159,
    23: 0.969696973,
    182: 0.988938733,
    74: 0.994608535,
    47: 0.996984378,
    114: 1.000281367,
    158: 1.00857413,
    107: 1.024303271,
    186: 1.027410776,
    169: 1.030398405,
    69: 1.046595955,
    120: 1.074653755,
    175: 1.076727813,
    54: 1.08482117,
    42: 1.085997567,
    141: 1.093347409,
    133: 1.095942063,
    131: 1.103379481,
    163: 1.106406146,
    171: 1.117413659,
    217: 1.127229678,
    112: 1.145719789,
    200: 1.149320624,
    43: 1.15072687,
    106: 1.159847235,
    202: 1.167243673,
    139: 1.17798465,
    152: 1.179737968,
    136: 1.189585625,
    58: 1.192928443,
    33: 1.202293954,
    55: 1.207762533,
    79: 1.21631678,
    29: 1.216941684,
    150: 1.245575049,
    190: 1.255485927,
    125: 1.26208981,
    90: 1.291199142,
    99: 1.295780274,
    147: 1.296586278,
    76: 1.30804027,
    34: 1.310762741,
    179: 1.312853794,
    212: 1.313192515,
    137: 1.316366621,
    27: 1.318270923,
    63: 1.322177079,
    19: 1.330232195,
    61: 1.338704448,
    170: 1.34222776,
    213: 1.342481837,
    143: 1.349258851,
    88: 1.357713146,
    134: 1.358296291,
    80: 1.359670697,
    105: 1.361774759,
    210: 1.36562395,
    138: 1.382447373,
    196: 1.4011455,
    122: 1.408994528,
    28: 1.411004971,
    73: 1.41119779,
    41: 1.411666472,
    183: 1.414911132,
    16: 1.415527818,
    51: 1.43962843,
    129: 1.442838304,
    126: 1.442845737,
    204: 1.457150033,
    146: 1.460216909,
    78: 1.462044318,
    89: 1.468414184,
    85: 1.474798159,
    164: 1.492020521,
    208: 1.494897043,
    209: 1.502114229,
    21: 1.515876584,
    197: 1.521290271,
    207: 1.524750358,
    98: 1.529337254,
    193: 1.535448247,
    50: 1.565720233,
    130: 1.575944911,
    149: 1.577060227,
    60: 1.577518645,
    216: 1.578734011,
    59: 1.581365293,
    62: 1.583889929,
    199: 1.584132171,
    20: 1.597820334,
    172: 1.613834299,
    81: 1.617513269,
    198: 1.62174871,
    167: 1.624031867,
    191: 1.625850156,
    24: 1.631124217,
    159: 1.634220136,
    124: 1.645863303,
    57: 1.663083884,
    192: 1.674445414,
    13: 1.686060143,
    123: 1.689813279,
    17: 1.716573022,
    26: 1.726821335,
    44: 1.731849317,
    168: 1.753248832,
    203: 1.763579384,
    84: 1.768450194,
    180: 1.775168673,
    111: 1.786049002,
    36: 1.791816483,
    205: 1.792848151,
    128: 1.799209749,
    15: 1.801635987,
    12: 1.80413912,
    22: 1.82928392,
    132: 1.839314042,
    127: 1.852388867,
    162: 1.854721898,
    25: 1.892126435,
    86: 1.892868597,
    218: 1.895487887,
    145: 1.898401115,
    166: 1.920639034,
    142: 1.921003677,
    18: 1.932655666,
    14: 1.95012963,
    135: 1.964177725,
    181: 1.965666478,
    195: 1.969888714,
    215: 1.971637336,
    206: 2.011308007,
    148: 2.06428697,
    194: 2.096142263,
    83: 2.105096823,
    211: 2.143378144,
    87: 2.185419727,
    82: 2.201789496
}
        # assign the dict
        self.entropy_weight = getattr(config, 'entropy_weight', 1.0)
        self.use_entropy_constraint = getattr(config, 'use_entropy_constraint', True)  # Toggle for biological constraints 
        self.use_size_constraint = getattr(config, 'use_size_constraint', True) # Toggle for size rules independently

        self.size_penalties = {
            'mild': 0.1,
            'moderate': 0.3,
            'severe': 0.5,
            'no_cost': 0.0
        }

        try:
            with open('size_rules.yaml', 'r') as f:
                self.size_rules = yaml.safe_load(f)
            # print("Loaded size_rules.yaml for reward shaping", flush=True)
        except Exception as e:
            print(f"Could not load size_rules.yaml: {e}", flush=True) 

        # BLOSUM substitution constraints
        self.use_blosum_constraint = getattr(config, 'use_blosum_constraint', True)
        self.blosum_scores = None
        self.blosum_rules = None
        self.blosum_penalties = {
            'bonus_small': 0.05,
            'mild_penalty': -0.1,
            'moderate_penalty': -0.3,
            'severe_penalty': -0.5,
            'none': 0.0
        }
        try:
            # Load BLOSUM substitution scores (position-specific)
            df_blosum = pd.read_csv('blosum_substitution_scores.csv')
            # Clean up junk rows (e.g. key/notes at bottom)
            df_blosum = df_blosum[pd.to_numeric(df_blosum['Position'], errors='coerce').notnull()]
            
            self.blosum_scores = {}
            for _, row in df_blosum.iterrows():
                try:
                    pos = int(float(row['Position'])) - 1 # 0-indexed
                    # Convert all score columns to numeric, skip NaN
                    scores = {}
                    for aa in ALPHABET:
                        if aa in row:
                            val = pd.to_numeric(row[aa], errors='coerce')
                            scores[aa] = float(val) if not pd.isna(val) else 0.0
                    self.blosum_scores[pos] = scores
                except Exception as e:
                    # print(f"Error parsing BLOSUM row: {e}")
                    continue
            
            # Load BLOSUM logic rules
            with open('blosum_rules.yaml', 'r') as f:
                self.blosum_rules = yaml.safe_load(f)
            # print("Loaded BLOSUM constraints", flush=True)
        except Exception as e:
            print(f"Could not load BLOSUM data: {e}", flush=True)

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
            # DHFROracle checkpoint keys are already correct
            oracle.load_state_dict(oracle_ckpt, strict=True)
        else:
            # BaseCNN needs key fixing
            oracle.load_state_dict({k.replace('predictor.', ''): v for k, v in oracle_ckpt.items()}, strict=False)

        oracle.eval()
        self.oracle = oracle.to(self.device)

        # Load data for buffer and calibration
        data = pd.read_csv('data/{}/{}.csv'.format(config.name, config.level))[["sequence", "target"]]

        # Calibrate DHFROracle predictions to match target scale
        if is_dhfr_oracle:
            print("Calibrating DHFROracle predictions...")
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
            slope, intercept, r_value, _, _ = linregress(sample_preds, sample_targets)

            self.oracle_scale = slope
            self.oracle_shift = intercept
            print(f"  Calibration: target = {slope:.6f} * pred + {intercept:.4f} (R²={r_value**2:.4f})")
        else:
            self.oracle_scale = 1.0
            self.oracle_shift = 0.0

        # Normalize fitness to [0, 1] range
        fitness_range = self.config.max_fitness - self.config.min_fitness
        data["target"] = (data["target"] - self.config.min_fitness) / fitness_range

        # Initialize buffer with starting percentile
        start_percentile = getattr(self.config, 'start_percentile', 95)  # Default to elite sequences
        self.buffer = Buffer(
            list(data.itertuples(index=False, name=None)),
            random=random.Random(seed),
            start_percentile=start_percentile
        )
        self.inits = data["sequence"].tolist()

        # Initialize sequence model
        model = VED(seq_cfg, pretrained=config.seq_pretrained)
        model.eval()
        self.model = model.to(self.device)
        self.model.set_wt_tokens(REFSEQ[self.protein][self.config.level])
        
        self.ep = 0
        self.steps = 0
        self.total_steps = 0
        self.state = None 
        self.state_seq = None # String 
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
        """
        Normalize oracle predictions to [0, 1] using dataset fitness range.

        Oracle outputs are in the same scale as dataset targets (after calibration),
        so we use config.min_fitness and config.max_fitness for normalization.
        """
        # Use dataset range from config
        fitness_range = self.config.max_fitness - self.config.min_fitness
        normalized = (target - self.config.min_fitness) / fitness_range

        # Clip to [0, 1] to handle outliers
        return max(0.0, min(1.0, normalized))

    def get_oracle_prediction(self, seq):
        """Get calibrated oracle prediction for a sequence"""
        with torch.no_grad():
            is_dhfr_oracle = hasattr(self, 'oracle_aa_to_idx')

            if is_dhfr_oracle:
                # DHFROracle: use oracle alphabet and token indices
                tokens = torch.tensor([[self.oracle_aa_to_idx[aa] for aa in seq]], dtype=torch.long).to(self.device)
                pred = self.oracle(tokens).cpu().item()
                # Apply calibration
                pred = pred * self.oracle_scale + self.oracle_shift
            else:
                # BaseCNN: use one-hot encoding
                _, pred = self.oracle(seq_to_one_hot(seq).unsqueeze(0).to(self.device), get_embed=True)
                pred = pred.cpu().item()

            return pred

    def get_mutation_cost(self, original_aa, new_aa):
        """Calculate penalty for mutating original_aa to new_aa based on size/shape rules"""
        if not self.size_rules or original_aa == new_aa:
            return 0.0
            
        rule = self.size_rules.get(original_aa, {})
        
        # Check rule categories associated with the original amino acid
        # Rules structure: AA -> category -> list/str of target AAs
        if new_aa in rule.get('no_cost', []):
            return self.size_penalties['no_cost']
        if new_aa in rule.get('mild', []):
            return self.size_penalties['mild']
        if new_aa in rule.get('moderate', []):
            return self.size_penalties['moderate']
        if new_aa in rule.get('severe', []):
            return self.size_penalties['severe']
            
        # Default fallback if not found in lists (assume mild-moderate risk)
        return 0.2

    def get_blosum_cost(self, pos, old_aa, new_aa):
        """Calculate penalty/bonus based on BLOSUM substitution scores at specific position"""
        if not self.blosum_scores or pos not in self.blosum_scores or not self.blosum_rules:
            return 0.0
        
        # Get score for the new amino acid at this position
        score = self.blosum_scores[pos].get(new_aa, 0)
        
        # Interpret score using thresholds
        thresh = self.blosum_rules.get('thresholds', {})
        if score >= thresh.get('high', 1):
            return self.blosum_penalties['bonus_small']
        elif score >= thresh.get('neutral', 0):
            return self.blosum_penalties['none']
        elif score >= thresh.get('low', -1):
            return self.blosum_penalties['mild_penalty']
        elif score >= thresh.get('very_low', -2):
            return self.blosum_penalties['moderate_penalty']
        else:
            return self.blosum_penalties['severe_penalty']

    def record_mutation(self, s1, s2):
        for pos, (i,j) in enumerate(zip(list(s1), list(s2))):
            if i != j:
                self.aa[j] += 1
                self.pos[pos] += 1
    
    def reset(self, seed=422):
        self.state_seq, self.buffer_idx = self.buffer.top()
        self.init_seq = self.state_seq
        with torch.no_grad():
            state = self.model.encode(self.state_seq)
            self.state = state.cpu().view(-1)
            target = self.get_oracle_prediction(self.state_seq)
            self.init_target = self.target = self.normalize_target(target)

        # Track the episode best for improvement-based rewards
        self.episode_best = self.init_target
        self.episode_start_fitness = self.init_target

        self.ep += 1
        self.steps = 0
        self.total_steps += 1
        return self.state, {}
    
    
    def step(self, action):
    # --- Step counts ---
        # print(f"[DEBUG_STEP] Step={self.steps}, Total={self.total_steps}", flush=True)
        self.steps += 1
        self.total_steps += 1

        # --- Action details ---
        action_tensor = torch.tensor(action)
        # print(f"[DEBUG_ACTION] Action mean={action_tensor.mean():.4f}, std={action_tensor.std():.4f}", flush=True)

        next_state = (action_tensor + self.state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            next_seq = self.model.decode(
                next_state, to_seq=True, template=self.state_seq, topk=self.config.topk
            )

            # --- Sequence mutations ---
            step_mut = distance(self.state_seq, next_seq)
            n_mut = distance(self.wt_seq, next_seq)
            # print(f"[DEBUG_SEQ] step_mut={step_mut}, n_mut={n_mut}", flush=True)
            # print(f"[DEBUG_SEQ] prev_seq[:25]=%s... next_seq[:25]=%s..." %
            #     (self.state_seq[:25], next_seq[:25]), flush=True)

            diffs = [(i, a, b) for i, (a, b) in enumerate(zip(self.state_seq, next_seq)) if a != b]
            # print(f"[DEBUG_DIFF] Mutated positions: {diffs[:10]} (showing first 10)", flush=True)


            # --- Entropy factor ---
            entropy_vals = [self.position_entropy.get(pos, 0) for pos, _, _ in diffs]
            if entropy_vals:
                entropy_factor = 1.0 + self.entropy_weight * np.mean(entropy_vals)
            else:
                entropy_factor = 1.0
            # print(f"[DEBUG_ENTROPY] entropy_factor={entropy_factor:.4f}", flush=True)

            # --- Size/Shape Constraint Penalty ---
            size_penalty = 0.0
            if self.size_rules and self.use_size_constraint:
                for _, old_aa, new_aa in diffs:
                    size_penalty += self.get_mutation_cost(old_aa, new_aa)
            
            size_factor = max(0.1, 1.0 - size_penalty)

            # --- BLOSUM Constraint Intensity ---
            blosum_adjustment = 0.0
            if self.blosum_scores and self.use_blosum_constraint:
                for pos, old_aa, new_aa in diffs:
                    blosum_adjustment += self.get_blosum_cost(pos, old_aa, new_aa)
            
            blosum_factor = max(0.1, 1.0 + blosum_adjustment)

        # --- Done condition checks ---
        done = (
            step_mut > self.done_cond.step_mut
            or self.steps > self.done_cond.max_steps
            or n_mut > self.done_cond.max_mutation
        )
        # print(f"[DEBUG_DONE_CHECK] done={done}, step_mut_limit={self.done_cond.step_mut}, "
        #     "max_steps={self.done_cond.max_steps}, max_mut={self.done_cond.max_mutation}", flush=True)

        reward = 0.0
        target = 0.0
        called = False

        if not done and step_mut <= self.done_cond.step_mut:
            self.oracle_calls += 1
            raw_target = self.get_oracle_prediction(next_seq)
            target = self.normalize_target(raw_target)

            improvement = target - self.episode_best
            improvement_from_start = target - self.episode_start_fitness
            # print(f"[DEBUG_FITNESS] raw_target={raw_target:.4f}, normalized_target={target:.6f}, "
            #   f"improvement={improvement:.6f}, improvement_from_start={improvement_from_start:.6f}", flush=True)

            if improvement > 0:
                # New episode best. entropy_factor = 1 + weight * mean(entropy of the mutated positions),
                # so mutations at variable positions earn more than mutations at conserved ones.
                # size_factor and blosum_factor scale the reward down for disfavored substitutions.
                reward = (1.0 + (improvement * 100.0)) * entropy_factor * size_factor * blosum_factor
                self.episode_best = target
                print(f"[SUCCESS] Improvement! Δ={improvement:.6f}, new_best={target:.6f}, reward={reward:.2f}", flush=True)
            elif improvement > -0.001:
                # Small tolerance for noise, but scaled down reward
                reward = 0.05 * entropy_factor * size_factor * blosum_factor
            else:
                # Clear negative reward for worse mutations
                reward = -0.5 * (1.0 + abs(improvement) * 50.0)

            self.best_discovered = max(self.best_discovered, target)
            self.target = target
            self.reward = reward
            self.buffer.push((next_seq, target, self.buffer_idx))
            self.buffer.update_global_best(next_seq, target, raw_target)
            self.buffer.update() # Ensure pool is updated for off-policy algorithms
            called = True

            # print(f"[DEBUG_REWARD] reward={reward:.4f}, called={called}, reason=oracle called", flush=True)
        else:
            reward = -1
            # print(f"[DEBUG_REWARD] reward=-1, called=False, reason=too many mutations or done before reward", flush=True)

        # Update state and sequence
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
                'called': called,
            }
            # print(f"[DEBUG_DONE] Episode done. Info={info}", flush=True)
            return self.state, reward, True, False, info

        # print(f"[DEBUG_CONTINUE] Continuing episode, state updated", flush=True)
        return self.state, reward, False, False, {}