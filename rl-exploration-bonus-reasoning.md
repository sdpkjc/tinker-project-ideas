# Exploration bonuses for reasoning model RL

In RLVR (RL with verifiable rewards) settings like math and coding, policies can sometimes get stuck in local optima, repeatedly generating similar reasoning traces even when they're incorrect. Standard RL algorithms focus on exploitation once a rewarding strategy is found, but this may prevent discovering better reasoning approaches.

This project explores adding exploration bonuses to RL training for reasoning models, encouraging the policy to explore diverse solution strategies beyond what's immediately rewarding.

Proposed exploration mechanisms:

**1. Trajectory diversity bonus**: Reward the policy for generating reasoning traces that are dissimilar to previous attempts on the same problem. Measure diversity using:
- Embedding distance in reasoning step space
- N-gram diversity in the reasoning text
- Structural diversity (e.g., different proof techniques in math)

**2. Curiosity-driven exploration**: Adapt ideas from [curiosity-driven RL](https://arxiv.org/abs/1705.05363) where the agent is rewarded for encountering novel states. In the reasoning context, reward the model for:
- Reaching intermediate states (reasoning steps) it hasn't seen frequently
- Using uncommon but valid reasoning patterns
- Discovering new ways to decompose problems

**3. Population-based diversity**: Train multiple policies in parallel and add a bonus for generating solutions different from the population average, inspired by [Population-Based Training](https://arxiv.org/abs/1711.09846).

Implementation approach:
1. Start with standard RLVR setup (e.g., on MATH dataset or coding problems)
2. Add an exploration bonus term to the reward function, weighted by a coefficient β
3. Track exploration metrics: number of unique reasoning patterns, solution diversity, etc.
4. Compare final policy performance and sample diversity against vanilla RL

Key questions:
- Does exploration help escape local optima and find better solution strategies?
- How should β be annealed over training (high early, low later)?
- Does diversity during training lead to more robust policies at test time?
- Can exploration help the model discover novel problem-solving approaches not in the training data?

This could lead to more creative and robust reasoning models that don't rely on memorized solution templates.
