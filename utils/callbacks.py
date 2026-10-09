import numpy as np
import wandb
import json
import warnings
from metric import Evaluator
from stable_baselines3.common.callbacks import BaseCallback

class BufferLoggingCallback(BaseCallback):
  def __init__(self, args, save_dir, pth_dir, verbose=0):
    super().__init__(verbose)
    self.save_dir = save_dir
    self.pth_dir = pth_dir
    self.evaluator = Evaluator(args.name, args.max_fitness, args.min_fitness, args.device)
    self.rounds = 0
    self.global_max_fitness = -float('inf')
    self.global_max_sequence = None
    self.global_max_raw_fitness = None  # Track raw (denormalized) fitness
    self.initial_max_fitness = None
    self.all_discovered_sequences = []
    self.args = args  # Store args for denormalization
    
  def _on_training_start(self) -> None:
    # Capture the snapshot of the pool at ABSOLUTE step 0
    buffer = self.model.env.get_attr('buffer')[0]
    if len(buffer.pool) > 0:
        self.initial_max_fitness = max([s[1] for s in buffer.pool])
        print(f"\n{'='*80}")
        print(f"STEP 0 SNAPSHOT")
        print(f"{'='*80}")
        print(f"Initial Maximum Fitness: {self.initial_max_fitness:.6f}")
        print(f"{'='*80}\n")

  def _on_step(self) -> bool:
    return True
  
  def _on_rollout_end(self) -> bool:

    buffer = self.model.env.get_attr('buffer')[0]
    infos = buffer.propose(k=128)
    seqs = [info[0] for info in infos]
    fitnesses = [info[1] for info in infos]
    inits = self.model.env.get_attr('inits')[0]
    fit, div, nov, high = self.evaluator.evaluate(seqs, inits)

    # (Initial fitness now captured at training start)

    # Update global max - check BOTH buffer pool AND buffer's global best tracker
    buffer_pool = buffer.pool
    buffer_max_fitness_in_pool = -float('inf')
    buffer_max_seq_in_pool = None

    if len(buffer_pool) > 0:
        # Buffer pool structure: [seq, fitness, score, visits]
        buffer_max_fitness_in_pool = max([s[1] for s in buffer_pool])
        buffer_max_idx = [s[1] for s in buffer_pool].index(buffer_max_fitness_in_pool)
        buffer_max_seq_in_pool = buffer_pool[buffer_max_idx][0]

    # Get the globally tracked best (which might not be in pool anymore)
    global_best_seq, global_best_fitness, global_best_raw = buffer.get_global_best()

    # Use the absolute best between pool and global tracker
    if global_best_fitness > buffer_max_fitness_in_pool:
        if global_best_fitness > self.global_max_fitness:
            self.global_max_fitness = global_best_fitness
            self.global_max_sequence = global_best_seq
            self.global_max_raw_fitness = global_best_raw
    else:
        if buffer_max_fitness_in_pool > self.global_max_fitness:
            self.global_max_fitness = buffer_max_fitness_in_pool
            self.global_max_sequence = buffer_max_seq_in_pool
            # Estimate raw fitness (we'll improve this later)
            self.global_max_raw_fitness = None

    # Store all discovered sequences with fitness
    self.all_discovered_sequences.extend([(seq, fit) for seq, fit in zip(seqs, fitnesses)])

    log = {
        'eval/round': self.rounds,
        'eval/fitness': np.mean(fitnesses),
        'eval/diversity': div,
        'eval/novelty': nov,
        'eval/high': high
    }
    self.rounds += 1

    # Print metrics every 10 rounds and at the end
    if self.rounds % 1 == 0 or self.rounds == 1:
        print(f"\n{'='*80}")
        print(f"EVALUATION METRICS - Round {self.rounds}")
        print(f"{'='*80}")
        print(f"Current Round Statistics:")
        print(f"  Average Fitness (top-128): {fit:.6f}")
        print(f"  Diversity: {div:.6f}")
        print(f"  Novelty: {nov:.6f}")
        print(f"  Highest in batch: {high:.6f}")
        print(f"\nGlobal Statistics:")
        print(f"  Global Maximum Fitness (normalized): {self.global_max_fitness:.6f}")
        if self.global_max_raw_fitness is not None:
            print(f"  Global Maximum Fitness (raw): {self.global_max_raw_fitness:.6f}")
        if self.initial_max_fitness is not None:
            improvement = self.global_max_fitness - self.initial_max_fitness
            pct_improvement = (improvement / abs(self.initial_max_fitness)) * 100
            print(f"  Initial Maximum Fitness: {self.initial_max_fitness:.6f}")
            print(f"  Absolute Improvement: {improvement:.6f}")
            print(f"  Percentage Improvement: {pct_improvement:.2f}%")
        print(f"  Total Unique Sequences Discovered: {len(set([s[0] for s in self.all_discovered_sequences]))}")
        print(f"  Total Oracle Calls: {self.model.env.get_attr('oracle_calls')[0]}")

        # Show buffer pool vs global best comparison
        buffer = self.model.env.get_attr('buffer')[0]
        buffer_best_seq, buffer_best_fit, buffer_best_raw = buffer.get_global_best()
        buffer_pool_max = max([s[1] for s in buffer.pool]) if buffer.pool else -float('inf')
        print(f"\n  Buffer Status:")
        print(f"    Best in current pool: {buffer_pool_max:.6f}")
        print(f"    Global best (tracked): {buffer_best_fit:.6f}")
        if buffer_best_seq and buffer_best_seq not in [s[0] for s in buffer.pool]:
            print(f"     WARNING: Global best was EVICTED from buffer pool!")

        # Top 10 sequences
        sorted_seqs = sorted(self.all_discovered_sequences, key=lambda x: x[1], reverse=True)
        unique_top = []
        seen = set()
        for seq, fitness in sorted_seqs:
            if seq not in seen:
                unique_top.append((seq, fitness))
                seen.add(seq)
            if len(unique_top) >= 10:
                break

        print(f"\nTop 10 Discovered Sequences:")
        for i, (seq, fitness) in enumerate(unique_top, 1):
            print(f"  {i}. Fitness: {fitness:.6f} | Seq: {seq}")

        print(f"{'='*80}\n")

    np.save('{}/{}.npy'.format(self.save_dir, self.rounds), seqs)
    self.model.save(self.pth_dir+'/{}.zip'.format(self.rounds))
    wandb.log(log)
    with open('{}/{}.txt'.format(self.save_dir, self.rounds), 'w') as file:
        json.dump({k: float(v) for k,v in log.items()}, file)

  def _on_training_end(self) -> None:
    """Print final summary"""
    print(f"\n{'#'*80}")
    print(f"FINAL RESEARCH PAPER METRICS - DHFR OPTIMIZATION")
    print(f"{'#'*80}")

    if self.initial_max_fitness is not None:
        improvement = self.global_max_fitness - self.initial_max_fitness
        pct_improvement = (improvement / abs(self.initial_max_fitness)) * 100

        print(f"\nKey Results:")
        print(f"  Initial Maximum Fitness: {self.initial_max_fitness:.6f}")
        print(f"  Global Maximum Fitness Achieved: {self.global_max_fitness:.6f}")
        print(f"  Absolute Improvement: {improvement:.6f}")
        print(f"  Percentage Improvement: {pct_improvement:.2f}%")

    print(f"\nExploration Statistics:")
    print(f"  Total Rounds Completed: {self.rounds}")
    print(f"  Total Oracle Calls: {self.model.env.get_attr('oracle_calls')[0]}")
    print(f"  Total Sequences Evaluated: {len(self.all_discovered_sequences)}")
    print(f"  Total Unique Sequences: {len(set([s[0] for s in self.all_discovered_sequences]))}")

    # Top sequences
    sorted_seqs = sorted(self.all_discovered_sequences, key=lambda x: x[1], reverse=True)
    unique_top = []
    seen = set()
    for seq, fitness in sorted_seqs:
        if seq not in seen:
            unique_top.append((seq, fitness))
            seen.add(seq)
        if len(unique_top) >= 20:
            break

    print(f"\nTop 20 Sequences Discovered:")
    for i, (seq, fitness) in enumerate(unique_top, 1):
        print(f"  Rank {i}: Fitness = {fitness:.6f}")
        print(f"         Sequence = {seq}")

    print(f"\nGlobal Best Sequence (for validation):")
    if self.global_max_sequence:
        print(f"  Fitness (normalized): {self.global_max_fitness:.6f}")
        if self.global_max_raw_fitness is not None:
            print(f"  Fitness (raw): {self.global_max_raw_fitness:.6f}")
        print(f"  Full Sequence: {self.global_max_sequence}")

    # Final buffer status check
    buffer = self.model.env.get_attr('buffer')[0]
    buffer_best_seq, buffer_best_fit, buffer_best_raw = buffer.get_global_best()
    buffer_pool_max = max([s[1] for s in buffer.pool]) if buffer.pool else -float('inf')
    print(f"\n  Final Buffer Status:")
    print(f"    Best in final pool: {buffer_pool_max:.6f}")
    raw_str = f"{buffer_best_raw:.6f}" if buffer_best_raw is not None else "N/A"
    print(f"    Global best (tracked): {buffer_best_fit:.6f} (raw: {raw_str})")
    if buffer_best_seq and buffer_best_seq not in [s[0] for s in buffer.pool]:
        print(f"     Global best sequence was evicted from buffer!")
        print(f"    This explains the discrepancy between 'Cumulative Best' and 'Global Best Sequence' fitness values.")

    # Save final results to JSON
    final_results = {
        'initial_max_fitness': float(self.initial_max_fitness) if self.initial_max_fitness else None,
        'global_max_fitness_normalized': float(self.global_max_fitness),
        'global_max_fitness_raw': float(self.global_max_raw_fitness) if self.global_max_raw_fitness else None,
        'absolute_improvement': float(improvement) if self.initial_max_fitness else None,
        'percentage_improvement': float(pct_improvement) if self.initial_max_fitness else None,
        'total_rounds': self.rounds,
        'total_oracle_calls': int(self.model.env.get_attr('oracle_calls')[0]),
        'total_sequences_evaluated': len(self.all_discovered_sequences),
        'total_unique_sequences': len(set([s[0] for s in self.all_discovered_sequences])),
        'global_best_sequence': self.global_max_sequence,
        'buffer_best_in_pool': float(buffer_pool_max),
        'buffer_global_best': float(buffer_best_fit),
        'buffer_global_best_raw': float(buffer_best_raw) if buffer_best_raw else None,
        'global_best_evicted': bool(buffer_best_seq and buffer_best_seq not in [s[0] for s in buffer.pool]),
        'top_20_sequences': [{'rank': i+1, 'fitness': float(fitness), 'sequence': seq}
                            for i, (seq, fitness) in enumerate(unique_top)]
    }

    with open(f'{self.save_dir}/final_research_metrics.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\nFinal metrics saved to: {self.save_dir}/final_research_metrics.json")
    print(f"{'#'*80}\n")
    

class RewardLoggingCallback(BaseCallback):
  def __init__(self, verbose=0):
    super().__init__(verbose)
    self.cumul = 0
  
  def _on_step(self) -> bool:
    # Log reward
    target = self.model.env.get_attr('target')[0]
    reward = self.model.env.get_attr('reward')[0]
    best = self.model.env.get_attr('best_discovered')[0]
  
    self.cumul += reward
    log = {
      'Fitness': target, 
      'Cumulative Reward': self.cumul, 
      'Cumulative Best': best
      }

    if self.model.env.get_attr('done')[0]:
      ep = self.model.env.get_attr('ep')[0]
      log['Mutation from WT'] = self.model.env.get_attr('n_mut')[0]
      log['Step Mutation'] = self.model.env.get_attr('step_mut')[0]
      log['Episode'] = ep   
      log['Oracle Calls'] = self.model.env.get_attr('oracle_calls')[0]
      if ep % 200 == 199:
        aa = self.model.env.get_attr('aa')[0]
        pos = self.model.env.get_attr('pos')[0]
        table_aa = wandb.Table(data=[[label, val] for (label, val) in aa.items()], columns = ["type", "count"])
        log['Mutated AA'] = wandb.plot.bar(table_aa, "type", "count", title="AA")
        table_pos = wandb.Table(data=[[label, val] for (label, val) in pos.items()], columns = ["position", "count"])
        log['Mutated Position'] = wandb.plot.bar(table_pos, "position", "count", title="Position")
        buffer = self.model.env.get_attr('buffer')[0]
        if len(buffer.pool) == buffer.buffer_size:
            fitness, diversity = buffer.get_performance()
            log['buffer/epsilon'] = buffer.epsilon
            log['buffer/fitness'] = fitness
            log['buffer/diversity'] = diversity
    try: 
      wandb.log(log)
    except FileNotFoundError: 
        # sometimes wandb fails to log table
        warnings.warn("tmp dir problem")
        del log['Mutated AA']
        del log['Mutated Position']
        wandb.log(log)
    return True