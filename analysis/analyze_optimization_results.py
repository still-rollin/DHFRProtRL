"""
Analysis script for DHFR optimization results
Analyzes whether the RL agent truly generates better fitness sequences
"""

# Sample data extracted from the debug output
episodes_data = [
    {"episode": 1, "init_fitness": 0.5383, "final_fitness": 0.5370, "n_mut": 3, "steps": 20},
    {"episode": 2, "init_fitness": 0.5380, "final_fitness": 0.5380, "n_mut": 2, "steps": 20},
    {"episode": 3, "init_fitness": 0.5383, "final_fitness": 0.5378, "n_mut": 3, "steps": 20},
    {"episode": 4, "init_fitness": 0.5383, "final_fitness": 0.5378, "n_mut": 3, "steps": 20},
    {"episode": 5, "init_fitness": 0.5383, "final_fitness": 0.5378, "n_mut": 2, "steps": 20},
    {"episode": 6, "init_fitness": 0.5374, "final_fitness": 0.5374, "n_mut": 2, "steps": 20},
    {"episode": 7, "init_fitness": 0.5383, "final_fitness": 0.5383, "n_mut": 2, "steps": 20},
    {"episode": 8, "init_fitness": 0.5383, "final_fitness": 0.5379, "n_mut": 1, "steps": 20},
    {"episode": 9, "init_fitness": 0.5380, "final_fitness": 0.5380, "n_mut": 2, "steps": 20},
    {"episode": 10, "init_fitness": 0.5383, "final_fitness": 0.5378, "n_mut": 8, "steps": 20},
]

# Detailed reward progression from first episode
episode_1_rewards = [
    {"step": 0, "raw": -0.0239, "norm": 0.5378, "improvement": 0.0000},
    {"step": 1, "raw": -0.0233, "norm": 0.5379, "improvement": 0.0001},
    {"step": 2, "raw": -0.0276, "norm": 0.5374, "improvement": -0.0004},
    {"step": 3, "raw": -0.0235, "norm": 0.5379, "improvement": -0.0000},
    {"step": 4, "raw": -0.0230, "norm": 0.5379, "improvement": 0.0000},
    {"step": 5, "raw": -0.0227, "norm": 0.5380, "improvement": 0.0000},
    {"step": 6, "raw": -0.0235, "norm": 0.5379, "improvement": -0.0001},
    {"step": 7, "raw": -0.0316, "norm": 0.5370, "improvement": -0.0009},
    {"step": 8, "raw": -0.0313, "norm": 0.5371, "improvement": -0.0009},
    {"step": 9, "raw": -0.0227, "norm": 0.5380, "improvement": -0.0000},
    {"step": 10, "raw": -0.0235, "norm": 0.5379, "improvement": -0.0001},
    {"step": 11, "raw": -0.0235, "norm": 0.5379, "improvement": -0.0001},
    {"step": 12, "raw": -0.0227, "norm": 0.5380, "improvement": -0.0000},
    {"step": 13, "raw": -0.0227, "norm": 0.5380, "improvement": -0.0000},
    {"step": 14, "raw": -0.0235, "norm": 0.5379, "improvement": -0.0001},
    {"step": 15, "raw": -0.0239, "norm": 0.5378, "improvement": -0.0001},
    {"step": 16, "raw": -0.0235, "norm": 0.5379, "improvement": -0.0001},
    {"step": 17, "raw": -0.0224, "norm": 0.5380, "improvement": 0.0000},
    {"step": 18, "raw": -0.0235, "norm": 0.5379, "improvement": -0.0001},
    {"step": 19, "raw": -0.0323, "norm": 0.5370, "improvement": -0.0010},
]

# Episode 7 - One with a significant improvement
episode_7_rewards = [
    {"step": 0, "raw": -0.0182, "norm": 0.5384, "improvement": 0.0001},
    {"step": 1, "raw": -0.0199, "norm": 0.5382, "improvement": -0.0002},
    {"step": 2, "raw": -0.0267, "norm": 0.5375, "improvement": -0.0009},
    {"step": 3, "raw": -0.0282, "norm": 0.5374, "improvement": -0.0010},
    {"step": 4, "raw": -0.0182, "norm": 0.5384, "improvement": 0.0000},
    {"step": 5, "raw": -0.0108, "norm": 0.5392, "improvement": 0.0008},  # Best improvement!
    {"step": 6, "raw": -0.0137, "norm": 0.5389, "improvement": -0.0003},
    {"step": 7, "raw": -0.0197, "norm": 0.5383, "improvement": -0.0009},
]

print("=" * 80)
print("ANALYSIS: IS THE RL AGENT TRULY GENERATING BETTER FITNESS SEQUENCES?")
print("=" * 80)
print()

# Analysis 1: Fitness changes across episodes
print("1. FITNESS PROGRESSION ACROSS EPISODES")
print("-" * 80)
improvements = 0
degradations = 0
neutral = 0

for ep in episodes_data:
    change = ep["final_fitness"] - ep["init_fitness"]
    if change > 0.0001:
        improvements += 1
        status = "IMPROVED"
    elif change < -0.0001:
        degradations += 1
        status = "DEGRADED"
    else:
        neutral += 1
        status = "→ NEUTRAL"

    print(f"Episode {ep['episode']:2d}: {ep['init_fitness']:.4f} → {ep['final_fitness']:.4f} "
          f"(Δ={change:+.4f}, {ep['n_mut']} muts) {status}")

print()
print(f"Summary: {improvements} improved, {neutral} neutral, {degradations} degraded")
print(f"Success Rate: {improvements}/{len(episodes_data)} = {improvements/len(episodes_data)*100:.1f}%")
print()

# Analysis 2: Within-episode dynamics
print("2. WITHIN-EPISODE BEHAVIOR (Episode 1 Example)")
print("-" * 80)
print("Step | Normalized Fitness | Improvement | Pattern")
print("-" * 80)

initial_fitness = episode_1_rewards[0]["norm"]
best_fitness = initial_fitness
worst_fitness = initial_fitness

for r in episode_1_rewards:
    if r["norm"] > best_fitness:
        best_fitness = r["norm"]
        marker = "  ← NEW BEST"
    elif r["norm"] < worst_fitness:
        worst_fitness = r["norm"]
        marker = "  ← WORST"
    else:
        marker = ""

    print(f"{r['step']:4d} | {r['norm']:.4f}         | {r['improvement']:+.4f}      | {marker}")

print()
print(f"Episode 1 Summary:")
print(f"  Initial fitness: {initial_fitness:.4f}")
print(f"  Final fitness:   {episode_1_rewards[-1]['norm']:.4f}")
print(f"  Best reached:    {best_fitness:.4f} (Δ={best_fitness-initial_fitness:+.4f})")
print(f"  Worst reached:   {worst_fitness:.4f} (Δ={worst_fitness-initial_fitness:+.4f})")
print(f"  Net change:      {episode_1_rewards[-1]['norm']-initial_fitness:+.4f}")
print()

# Analysis 3: Exploration vs Exploitation
print("3. EXPLORATION VS EXPLOITATION ANALYSIS")
print("-" * 80)

# Count improvements, degradations, and neutral within episode 1
within_ep_improvements = sum(1 for r in episode_1_rewards if r["improvement"] > 0.0001)
within_ep_degradations = sum(1 for r in episode_1_rewards if r["improvement"] < -0.0001)
within_ep_neutral = sum(1 for r in episode_1_rewards if abs(r["improvement"]) <= 0.0001)

print(f"Within-episode steps (Episode 1):")
print(f"  Steps with improvement:  {within_ep_improvements}/{len(episode_1_rewards)} ({within_ep_improvements/len(episode_1_rewards)*100:.1f}%)")
print(f"  Steps with degradation:  {within_ep_degradations}/{len(episode_1_rewards)} ({within_ep_degradations/len(episode_1_rewards)*100:.1f}%)")
print(f"  Steps neutral:           {within_ep_neutral}/{len(episode_1_rewards)} ({within_ep_neutral/len(episode_1_rewards)*100:.1f}%)")
print()

# Analysis 4: Key observations
print("4. KEY OBSERVATIONS")
print("-" * 80)

observations = [
    ("The agent starts from 25th percentile sequences (fitness ~0.537-0.538)",
     "This is a good initialization strategy - not too good, room for improvement"),

    ("Most episodes show DEGRADATION or NEUTRAL outcomes",
     "The agent is NOT consistently improving fitness"),

    ("Within episodes, most steps are NEUTRAL or show DEGRADATION",
     "The agent makes many mutations that don't improve fitness"),

    ("Fitness changes are VERY SMALL (±0.001 range)",
     "The fitness landscape is very flat - hard to find improvements"),

    ("The agent sometimes finds small improvements (e.g., +0.0008)",
     "There ARE beneficial mutations, but they're rare and small"),

    ("The agent explores extensively with many mutations",
     "With ent_coef=0.01, there's some exploration, but it's not highly directed"),

    ("Final sequences often have 1-8 mutations from initial",
     "The agent IS making changes, but they're not beneficial"),
]

for i, (obs, interpretation) in enumerate(observations, 1):
    print(f"{i}. {obs}")
    print(f"   → {interpretation}")
    print()

# Final verdict
print("=" * 80)
print("VERDICT: IS THE AGENT GENERATING BETTER FITNESS SEQUENCES?")
print("=" * 80)
print()
print("NO - The agent is NOT reliably generating better fitness sequences")
print()
print("REASONS:")
print("--------")
print("1. **Most episodes end WORSE or NEUTRAL**: 0-10% improvement rate")
print("2. **Very small fitness changes**: Improvements when they occur are ~0.0008 (0.08%)")
print("3. **Flat fitness landscape**: The oracle shows very similar scores for mutations")
print("4. **No learning visible**: Early episodes don't show better performance than later ones")
print("5. **Random walk behavior**: Agent explores but doesn't exploit good mutations")
print()
print("WHY IS THIS HAPPENING?")
print("----------------------")
print("1. **Oracle calibration issue**: The fitness landscape is TOO FLAT")
print("   - Initial fitness: ~0.538")
print("   - Best found: ~0.539")
print("   - Range: Only 0.001 (0.1%)!")
print()
print("2. **Reward signal too weak**: With such small changes, reward = 1 + 0.537 ≈ 1.537")
print("   - Difference between good and bad: ~0.001/1.537 ≈ 0.065%")
print("   - This is NOISE LEVEL for neural network learning")
print()
print("3. **Insufficient exploration**: ent_coef=0.01 × 5 = 0.05 is low")
print("   - Agent needs more aggressive exploration to find rare improvements")
print()
print("4. **Episode length constraint**: Max 20 steps limits search")
print("   - Agent terminates before finding good solutions")
print()
print("RECOMMENDATIONS:")
print("----------------")
print("1. Fix oracle calibration - fitness range should be wider (e.g., 0.3-0.8)")
print("2. Use sparse rewards based on IMPROVEMENT rather than absolute fitness")
print("3. Increase entropy coefficient (try 0.1-0.5)")
print("4. Increase episode length (try 50-100 steps)")
print("5. Implement curriculum learning - start from worse sequences")
print("6. Add shaped rewards for finding novel mutations")
print()
print("=" * 80)
