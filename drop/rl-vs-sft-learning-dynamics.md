# Empirical comparison of learning dynamics: RL vs. SFT on the same data

Reinforcement learning (RL) and supervised fine-tuning (SFT) are both used for post-training, but we lack clear empirical understanding of when one is preferable to the other. RLHF is theoretically appealing because it directly optimizes what we care about (human preferences), but it's also more complex and potentially less stable than SFT.

A fundamental question: Given the same preference data, how do RL and SFT differ in what they learn, how fast they learn it, and what failure modes they exhibit?

## Core experimental setup

Start with identical preference data and compare two training approaches:

**Approach 1: Direct RL (RLHF)**
1. Train reward model on preference data
2. Use RL (PPO) to optimize policy against reward model
3. Track: KL divergence from base model, reward model scores, validation metrics

**Approach 2: Convert to SFT**
1. Convert preferences to SFT data: keep only preferred responses from each pair
2. Train policy with supervised learning on preferred responses
3. Track: same metrics as RL for comparison

Both start from the same base model and use the same preference data (just accessed differently).

## Research questions

1. **Sample efficiency**: Which approach reaches target performance with fewer data?
   - RL can learn from counterfactual reasoning (why B is better than A)
   - SFT learns from positive examples only
   - Does RL's additional signal translate to better data efficiency?

2. **Learning dynamics**: How do they progress differently during training?
   - Does RL show more instability or non-monotonic progress?
   - Does SFT plateau faster but at a lower level?
   - How does KL divergence from base model evolve differently?

3. **Generalization**: Which approach generalizes better to out-of-distribution prompts?
   - Does RL's exploration lead to better coverage?
   - Does SFT's simplicity lead to more robust patterns?

4. **Failure modes**: What goes wrong with each approach?
   - RL: reward hacking, mode collapse, training instability
   - SFT: overfitting to specific phrasings, lack of diversity
   - Can we characterize when each failure mode appears?

5. **Scaling behavior**: How do differences change with scale?
   - More data, larger models, longer training
   - Do RL advantages compound or diminish at scale?

## Controlled factors to vary

Run the comparison under different conditions:

**Data quality**:
- High-quality preferences (consistent, clear)
- Noisy preferences (inconsistent labels)
- Ambiguous preferences (both responses similar)

**Data quantity**:
- Low data regime (1K examples)
- Medium (10K examples)
- High (100K+ examples)

**Model scale**:
- Small models (1B parameters)
- Medium (7B parameters)
- Large (70B+ parameters)

**Task domains**:
- Instruction following
- Creative writing
- Math reasoning
- Code generation

## Detailed measurements

**Training metrics**:
- Loss curves and convergence speed
- Gradient norms and training stability
- Compute time to reach target performance
- Number of examples needed for target performance

**Evaluation metrics**:
- Performance on held-out test set (both in-distribution and OOD)
- Human preference win rate (RL vs. SFT outputs)
- Automated evals (instruction following benchmarks)
- Diversity metrics (unique n-grams, response variety)

**Behavioral analysis**:
- Qualitative analysis of where RL/SFT outputs differ
- Failure case categorization
- Response length and formatting differences
- Mode coverage (do they explore same regions of output space?)

## Expected insights

This study could reveal:
- Clear guidelines for when to use RL vs. SFT
- Whether RL's theoretical advantages materialize in practice
- How to get benefits of both (hybrid approaches)
- What improvements to RL would make it competitive with SFT's simplicity

## Extensions

**Hybrid approaches**:
- SFT initialization followed by RL fine-tuning
- Alternating SFT and RL phases
- Using RL for exploration, SFT for exploitation

**Advanced RL algorithms**:
- Compare PPO vs. DPO vs. other RL algorithms
- Does the choice of RL algorithm change conclusions?

**Preference vs. demonstration data**:
- What if we have demonstrations (single examples) instead of preferences?
- Does this change the RL vs. SFT trade-off?

## Connection to practice

Many practitioners default to simpler SFT even when they have preference data, possibly leaving performance on the table. Others invest heavily in RL infrastructure. This empirical study would ground these decisions in data rather than intuition.

The results could lead to:
- Decision trees for choosing training approaches
- Hybrid methods that combine strengths of both
- Improved RL algorithms that close efficiency gaps with SFT
- Better understanding of what reward modeling and RL actually provide
