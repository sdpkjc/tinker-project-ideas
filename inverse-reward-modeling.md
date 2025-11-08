# Inverse reward modeling: Finding rewards that maximally distinguish policies

Standard reward modeling takes a dataset of human preferences and learns a reward function that explains those preferences. The resulting reward is then used to train a policy via RL. But this pipeline has a subtle issue: the reward model is trained *before* seeing the policy it will be used to train, leading to potential misalignment when the policy explores out-of-distribution regions.

This project proposes **inverse reward modeling**: given a set of policies (good and bad), find the reward function that maximally distinguishes between them. This flips the usual paradigm—instead of learning rewards from human data and hoping they generalize to a policy, we directly optimize rewards to create the largest behavioral gap between desired and undesired policies.

## Core idea

Standard RLHF pipeline:
1. Collect human preferences on completions
2. Train reward model R on preferences
3. Train policy π to maximize R(π)

Inverse reward modeling:
1. Collect or train a set of reference policies: {π_good, π_bad}
2. Find reward R that maximizes: E[R(π_good)] - E[R(π_bad)]
3. Train new policy π to maximize R(π)

The key insight: if we can define what "good" and "bad" policies look like behaviorally (even without perfect preference data), we can reverse-engineer a reward that captures the difference.

## Where do reference policies come from?

**Option 1: Existing checkpoints**
- π_good: Models after instruction tuning, or high-quality assistant models
- π_bad: Base models, early training checkpoints, or models after adversarial fine-tuning

**Option 2: Synthetic construction**
- Generate two sets of completions from the same model with different prompts
- π_good: Model with helpful system prompt
- π_bad: Model with toxic system prompt or no system prompt

**Option 3: Behavioral filters**
- Sample many completions from a single model
- Use a coarse filter (simple heuristics, existing reward model, or human annotation of extremes) to divide into good/bad
- Learn refined reward that maximizes the gap

## Training the inverse reward model

Formulate as a max-margin problem:

```
max_R  E_{x,y~π_good}[R(x,y)] - E_{x,y~π_bad}[R(x,y)] - λ||R||²
```

Where:
- x is a prompt, y is a completion
- R(x,y) is the reward function being learned
- λ is a regularization term to prevent degeneracy

Alternatively, use a contrastive learning framework where R is trained to separate embeddings of good vs. bad policy behaviors.

## Why this might work better than standard reward modeling

1. **Direct optimization for the actual use case**: The reward is explicitly trained to shape policy behavior, not just to match human preference labels

2. **Handles distribution shift**: By training on actual policy outputs (not just fixed preference data), the reward model sees the distribution it will be applied to

3. **Leverages behavioral data**: We can define good/bad policies through demonstrated behavior, which may be easier than collecting fine-grained preference labels

4. **Amplification**: If π_good is slightly better than π_bad, inverse reward modeling finds features that amplify the difference, potentially leading to policies better than π_good

## Research questions

- Does inverse reward modeling produce policies that outperform standard RLHF on instruction-following evals?
- How sensitive is the method to the choice of reference policies?
- Can this approach reduce reward hacking, since the reward is optimized on actual policy behavior?
- What happens if we iteratively update the reward: train π with R, use π as new π_good, retrain R, repeat?
- Can we combine inverse reward modeling with standard preference data (hybrid approach)?

## Potential failure modes

- **Reward degeneracy**: R might learn spurious correlations that happen to differ between π_good and π_bad but don't capture the true objective
- **Overfitting to references**: R might only work for distinguishing the specific reference policies, not for training new policies
- **Computational cost**: May require many samples from reference policies to train R

These risks can be mitigated through:
- Regularization and diversity of reference policies
- Validation on held-out policy checkpoints
- Hybrid approaches that combine inverse RM with preference data

## Connection to existing work

This relates to:
- **Inverse reinforcement learning** ([Abbeel & Ng, 2004](https://ai.stanford.edu/~ang/papers/icml04-apprentice.pdf)): Similar in spirit but traditionally applied to learn rewards from expert demonstrations in single-agent RL
- **Adversarial imitation learning**: GAIL and related methods learn to imitate experts, but don't explicitly construct interpretable rewards
- **Preference learning from policy comparisons**: Some recent work ranks entire policies rather than individual completions, but doesn't frame it as reward optimization

The novelty here is applying inverse thinking specifically to LLM reward modeling, using policy-level behavioral differences rather than trajectory-level preferences.
