import random
import torch
import numpy as np
from utils.constants import seq_to_one_hot
from utils.eval_utils import diversity
from heapq import nsmallest

class Buffer:
    def __init__(self, data, random: random.Random, buffer_size=128, step_size=50, gamma=0.96, start_percentile=25):
        self.buffer_size = buffer_size
        self.random = random

        if isinstance(data, list):
            self.start_wt = False
            self.stats = torch.zeros((len(data[0][0]), 20))
            self.pool = self.initialize_pool(data, start_percentile=start_percentile)
        elif isinstance(data, str):
            self.start_wt = True
            self.wt = data
            self.pool = []
            self.stats = torch.zeros((len(data), 20))
        else:
            raise NotImplementedError("Buffer should be initialized to dataset or WT")

        self.epsilon = 1.0
        self.step = 0
        self.step_size = step_size
        self.gamma = gamma
        self.exploitation = 2.0
        self.trajs = []

        # Track global best sequence separately to prevent eviction
        self.global_best_seq = None
        self.global_best_fitness = -float('inf')
        self.global_best_raw_fitness = None  # Store raw (denormalized) fitness too

    def initialize_pool(self, data, start_percentile=25):
        """
        Initialize buffer with sequences from a specific percentile range.

        Args:
            data: List of (sequence, fitness) tuples
            start_percentile: Which percentile to start from (0-100)
                             5 = start from BOTTOM 5% (worst sequences, most room to improve)
                             25 = start from 25th percentile (low-quality)
                             50 = start from median quality
                             95 = start from TOP 5% (elite sequences, original behavior)
        """
        data = sorted(data, key=lambda x: -x[1])  # Sort by fitness descending

        # Calculate which sequences to use based on percentile
        # Data is sorted DESCENDING: [best=rank0, mid=rank2080, worst=rank4160]
        # start_percentile=5 means "5th percentile from bottom" = WORST 5%
        # start_percentile=95 means "95th percentile from bottom" = BEST 5%
        total = len(data)

        # Percentile to index conversion (for DESCENDING sorted data):
        # start_percentile=5  → (1-0.05)*4161 = index 3952 (worst 5%)
        # start_percentile=95 → (1-0.95)*4161 = index 208 (best 5%)
        percentile_idx = int(total * (1 - start_percentile/100))

        # Take buffer_size sequences centered around this percentile
        start_idx = max(0, percentile_idx - self.buffer_size // 2)
        end_idx = min(percentile_idx + self.buffer_size // 2, total)

        # If not enough sequences in this range, adjust
        if end_idx - start_idx < self.buffer_size:
            start_idx = max(0, end_idx - self.buffer_size)

        candidate_pool = data[start_idx:end_idx]

        print(f"[BUFFER] Initializing with sequences from {start_percentile}th percentile")
        print(f"[BUFFER] Fitness range: [{candidate_pool[-1][1]:.4f}, {candidate_pool[0][1]:.4f}]")
        print(f"[BUFFER] Using sequences at indices {start_idx}-{end_idx} out of {total}")

        pool = []
        for c in candidate_pool:
            self.stats += seq_to_one_hot(c[0])
            pool.append([c[0], c[1], c[1], 1])  # seq, fitness, score, visits
        return pool
    def find_min(self):
        idx = 0
        fit = self.pool[0][1]
        for i, p in enumerate(self.pool):
            if p[1] < fit:
                idx = i
                fit = p[1]
        return idx

    def push(self, traj):
        self.trajs.append(traj)

    def describe(self):
        return self.stats / len(self.pool)

    def top(self):
        self.step += 1
        if len(self.pool) == self.buffer_size and self.step % self.step_size == 0:
            self.epsilon = max(0.05, self.gamma * self.epsilon)

        if len(self.pool) < self.buffer_size:
            return self.wt, -1

        # ------------- exploration -------------
        if self.random.random() < self.epsilon:
            visit = np.array([s[3] for s in self.pool])
            prob = 1 / (np.sqrt(visit) + 1e-8)
            prob /= prob.sum()
            idx = self.random.choices(range(self.buffer_size), weights=prob, k=1)[0]
            return self.pool[idx][0], idx

        # ------------- exploitation -------------
        score = np.array([s[1] for s in self.pool])
        score = (score - score.min()) / (score.max() - score.min() + 1e-8)
        score *= self.exploitation
        weights = np.exp(score) / np.exp(score).sum()
        idx = self.random.choices(range(self.buffer_size), weights=weights, k=1)[0]
        return self.pool[idx][0], idx

    def update(self):
        for traj in self.trajs:
            seq, fitness, buffer_idx = traj
            if buffer_idx == -1 and len(self.pool) < self.buffer_size:
                self.stats += seq_to_one_hot(seq)
                self.pool.append([seq, fitness, fitness, 1])
                continue

            self.pool[buffer_idx][2] += fitness
            self.pool[buffer_idx][3] += 1

            # If sequence is new and has higher fitness — replace weakest one
            if seq not in [s[0] for s in self.pool]:
                weakest_idx = self.find_min()

                # Never evict the global best sequence
                # If weakest is global best, find next weakest
                if (self.global_best_seq is not None and
                    self.pool[weakest_idx][0] == self.global_best_seq):
                    # Find second weakest that is NOT the global best
                    sorted_pool = sorted(enumerate(self.pool), key=lambda x: x[1][1])
                    for idx, entry in sorted_pool:
                        if entry[0] != self.global_best_seq:
                            weakest_idx = idx
                            break

                if fitness > self.pool[weakest_idx][1]:
                    self.stats -= seq_to_one_hot(self.pool[weakest_idx][0])
                    self.stats += seq_to_one_hot(seq)
                    self.pool[weakest_idx] = [seq, fitness, fitness, 1]

        # Normalize scores but DO NOT reset visits each time
        for i, s in enumerate(self.pool):
            self.pool[i][2] = s[1]
            self.pool[i][3] = max(1, s[3])  # keep visit count growing

        self.trajs = []

    def update_global_best(self, seq, fitness, raw_fitness=None):
        """Update global best sequence if this is better"""
        if fitness > self.global_best_fitness:
            self.global_best_fitness = fitness
            self.global_best_seq = seq
            self.global_best_raw_fitness = raw_fitness
            print(f"[BUFFER] New global best! Fitness={fitness:.6f}, Raw={(f'{raw_fitness:.6f}' if raw_fitness is not None else 'N/A')}")
            return True
        return False

    def get_global_best(self):
        """Return global best sequence and its fitness"""
        return self.global_best_seq, self.global_best_fitness, self.global_best_raw_fitness

    def get_performance(self):
        buffer_diversity = diversity([s[0] for s in self.pool])
        buffer_fitness = sum([s[1] for s in self.pool]) / len(self.pool)
        return buffer_fitness, buffer_diversity

    def propose(self, k):
        def sort_key(x):
            return -x[1], x[0]
        top_k_results = nsmallest(k, self.pool, key=sort_key)
        return top_k_results
