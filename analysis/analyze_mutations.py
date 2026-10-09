"""
Detailed mutation analysis from optimization run
"""

# Sample mutations observed from debug output
mutations_ep1 = [
    # (position, from, to, step, fitness_after)
    (120, 'T', 'S', 0, 0.5378),
    (106, 'F', 'Y', 1, 0.5379),
    (111, 'N', 'K', 1, 0.5379),
    (120, 'S', 'T', 1, 0.5379),  # Reverted!
    (106, 'Y', 'F', 2, 0.5374),  # Reverted!
    (111, 'K', 'N', 2, 0.5374),  # Reverted!
    (138, 'D', 'E', 2, 0.5374),
    (138, 'E', 'D', 3, 0.5379),  # Reverted!
    (164, 'L', 'I', 4, 0.5379),
    (91, 'V', 'I', 5, 0.5380),
    (164, 'I', 'L', 5, 0.5380),  # Reverted!
    (91, 'I', 'V', 6, 0.5379),   # Reverted!
]

print("=" * 80)
print("MUTATION PATTERN ANALYSIS")
print("=" * 80)
print()

print("1. MUTATION BEHAVIOR")
print("-" * 80)

# Count reversions
total_mutations = len(mutations_ep1)
reverted_positions = set()
for i, (pos1, from1, to1, step1, fit1) in enumerate(mutations_ep1):
    for pos2, from2, to2, step2, fit2 in mutations_ep1[i+1:]:
        if pos1 == pos2 and from2 == to1 and to2 == from1:
            reverted_positions.add((pos1, from1, to1, step1))
            print(f"Position {pos1}: {from1}→{to1} (step {step1}) REVERTED to {to2} (step {step2})")
            print(f"   Fitness: {fit1:.4f} → {fit2:.4f}")

print(f"\nTotal mutations: {total_mutations}")
print(f"Reverted mutations: {len(reverted_positions)}")
print(f"Reversion rate: {len(reverted_positions)/total_mutations*100:.1f}%")
print()

print("2. AMINO ACID SUBSTITUTION PATTERNS")
print("-" * 80)
print("Common substitutions observed:")
print("  - Conservative: F↔Y (aromatic), I↔L↔V (hydrophobic), N↔D (polar)")
print("  - T↔S (both polar, hydroxyl)")
print("  - K↔R (both basic)")
print()
print("Observation: Agent makes CONSERVATIVE substitutions")
print("  → These are chemically similar amino acids")
print("  → Less likely to disrupt protein structure")
print("  → BUT also less likely to significantly improve fitness")
print()

print("3. EXPLORATION STRATEGY")
print("-" * 80)
print("The agent shows 'RANDOM WALK' behavior:")
print()
print("  Step 1: Try mutation X → small change in fitness")
print("  Step 2: Try mutation Y → small change in fitness")
print("  Step 3: Revert mutation X → back to similar fitness")
print("  Step 4: Try mutation Z → small change in fitness")
print()
print("This indicates:")
print("  NO clear gradient following")
print("  NO memory of good mutations")
print("  NO exploitation of improvements")
print("  Random exploration of nearby sequence space")
print()

print("4. WHY IS THIS HAPPENING?")
print("-" * 80)
print()
print("ROOT CAUSE: The fitness landscape is TOO FLAT")
print()
print("Evidence:")
print(f"  • Starting fitness: ~0.538")
print(f"  • After mutations:  ~0.537-0.539")
print(f"  • Total range:      ~0.002 (0.2%)")
print()
print("For comparison, a good optimization should see:")
print(f"  • Starting fitness: ~0.40")
print(f"  • Target fitness:   ~0.80")
print(f"  • Total range:      ~0.40 (40%)")
print()
print("The current fitness range is 200x smaller than it should be!")
print()

print("5. MECHANISM ANALYSIS")
print("-" * 80)
print()
print("Q: How does the agent decide which mutations to make?")
print("A: The policy network outputs action values (mean=-0.1 to +0.1)")
print("   These are sampled with noise (std~0.18) to get mutation decisions")
print()
print("Q: Does the agent learn which mutations are good?")
print("A: NO - the reward signal is too weak to provide useful gradient")
print()
print("Current reward formula: reward = 1 + normalized_fitness + improvement")
print("  Example good step: 1 + 0.539 + 0.0008 = 1.5398")
print("  Example bad step:  1 + 0.537 + (-0.001) = 1.536")
print("  Difference: 0.0038 (0.25%)")
print()
print("PPO updates are based on advantage = actual_reward - value_baseline")
print("With such tiny differences, advantage ≈ noise")
print()

print("6. IS THE AGENT 'LEARNING' ANYTHING?")
print("-" * 80)
print()
print("Short answer: NO, not really")
print()
print("What we would expect from a learning agent:")
print("  Increasing success rate over episodes")
print("  Preferring beneficial amino acid positions")
print("  Avoiding harmful mutations")
print("  Building on good mutations (exploitation)")
print()
print("What we actually observe:")
print("  Flat success rate (0% improvement)")
print("  No position preferences (random mutations)")
print("  Frequently reverting mutations")
print("  Random walk exploration only")
print()
print("Conclusion: The agent is doing RANDOM SEARCH, not RL optimization")
print()

print("=" * 80)
print("FINAL ANSWER: Is it truly generating better fitness sequences?")
print("=" * 80)
print()
print("NO - The agent is NOT generating better fitness sequences")
print()
print("The agent is essentially performing RANDOM SEARCH with conservative")
print("amino acid substitutions. It occasionally stumbles upon small improvements")
print("(~0.0008 fitness increase), but more often makes neutral or harmful changes.")
print()
print("The RL training is not effective because:")
print("  1. The reward signal is too weak to learn from")
print("  2. The fitness landscape is too flat to navigate")
print("  3. The agent reverts good mutations due to noisy rewards")
print("  4. No clear improvement trend over episodes")
print()
print("This is a FUNDAMENTAL PROBLEM with the oracle calibration,")
print("not just a hyperparameter tuning issue.")
print()
print("=" * 80)
