# Automated curriculum learning for RL from AI feedback

Traditional RLHF and RLAIF apply reinforcement learning uniformly across all prompts in a training dataset. However, prompts vary greatly in difficulty—some require simple instruction following, others demand complex reasoning or creative problem-solving. Training on all prompts simultaneously may be inefficient, as the policy might struggle with hard problems before mastering easier ones, or waste time on already-mastered easy problems.

Curriculum learning, as explored in [Bengio et al. (2009)](https://qmro.qmul.ac.uk/xmlui/bitstream/handle/123456789/15972/Bengio%2C%202009%20Curriculum%20Learning.pdf) for supervised learning and [Narvekar et al. (2020)](https://arxiv.org/abs/2003.04960) for RL, shows that gradually increasing task difficulty can improve learning efficiency and final performance. This project adapts curriculum learning to the RLAIF setting with automated difficulty assessment.

## Core approach

Instead of sampling uniformly from all training prompts, dynamically adjust the training distribution based on the policy's current capabilities:

**Phase 1: Difficulty scoring**
1. For each prompt in the dataset, estimate its difficulty using:
   - Reward model uncertainty (high uncertainty = hard)
   - Policy success rate (low success = hard)
   - Response entropy/diversity (high diversity = ambiguous)
   - Length of required reasoning chains

**Phase 2: Adaptive sampling**
2. Start training with easier prompts (high success rate, clear rewards)
3. Gradually introduce harder prompts as the policy improves
4. Implement "difficulty pacing" strategies:
   - **Threshold-based**: Unlock next difficulty tier when success rate exceeds threshold
   - **Gradient-based**: Prioritize prompts at the edge of current capability (maximal learning)
   - **Mixed curriculum**: Maintain a blend of difficulties, but shift the distribution over time

**Phase 3: Automatic curriculum**
3. Use meta-learning or bandit algorithms to learn the optimal pacing strategy:
   - Which difficulty progression schedule leads to best final performance?
   - Should some hard examples be introduced early to prevent overfitting to easy cases?

## Implementation with Tinker Cookbook

This could be implemented in the [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) by modifying the prompt sampling strategy during RL training:

- Add difficulty metrics to each prompt in the dataset
- Implement curriculum schedulers that adjust sampling probabilities
- Track per-prompt statistics (success rate, reward statistics) during training
- Compare curriculum RL against uniform sampling baseline

## Key research questions

- Does curriculum learning improve sample efficiency in RLAIF compared to uniform sampling?
- What difficulty metrics best predict which prompts are learnable at each stage?
- Should curriculum be "smooth" (gradual difficulty increase) or "stepped" (discrete phases)?
- Does curriculum help prevent reward hacking by ensuring foundational capabilities before complex tasks?
- How does the optimal curriculum change with model scale?

## Extensions

- **Personalized curriculum**: Different curriculum paths for different capability dimensions (reasoning, creativity, instruction-following)
- **Bidirectional curriculum**: Also include "reverse curriculum" that revisits easier prompts to prevent catastrophic forgetting
- **Multi-objective curriculum**: Balance multiple objectives (helpfulness, harmlessness, honesty) with different difficulty progressions
- **Transfer learning**: Does a curriculum learned on one domain transfer to other domains?

This could significantly improve training efficiency for RLAIF and lead to better final policies, especially for domains with highly variable task difficulty.

