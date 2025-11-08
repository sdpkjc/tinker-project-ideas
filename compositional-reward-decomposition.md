# Compositional reward decomposition for multi-objective RLHF

In RLHF, we typically train a single reward model that tries to capture all aspects of response quality: helpfulness, harmlessness, honesty, conciseness, formatting, etc. This "monolithic" reward model must balance multiple objectives implicitly, which can lead to:
- Conflation of distinct quality dimensions
- Difficulty debugging when the policy optimizes the wrong thing
- Inability to adjust trade-offs between objectives without full retraining
- Reward hacking that exploits correlations between different objectives

This project explores compositional reward decomposition: training separate reward models for distinct objectives and combining them with explicit weights, providing interpretability and control over multi-objective optimization.

## Motivation

Monolithic reward model approach:
```
R(prompt, response) = single_score
```
Problem: The score conflates multiple factors, and we can't adjust their relative importance.

Compositional approach:
```
R(prompt, response) = w₁·R_helpful + w₂·R_safe + w₃·R_honest + w₄·R_format
```
Benefit: Explicit control over objective trade-offs, interpretable failure modes, flexible reconfiguration.

This is inspired by multi-objective reinforcement learning ([Roijers et al., 2013](https://www.jmlr.org/papers/volume14/roijers13a/roijers13a.pdf)) and modular neural networks ([Andreas et al., 2016](https://arxiv.org/abs/1511.02799)).

## Proposed decomposition dimensions

Decompose response quality into orthogonal dimensions:

1. **Helpfulness**: Does the response address the user's query effectively?
2. **Safety**: Does it avoid harmful, toxic, or dangerous content?
3. **Honesty**: Is it factually accurate and does it acknowledge uncertainty?
4. **Conciseness**: Is it appropriately detailed without unnecessary verbosity?
5. **Format**: Does it follow formatting instructions (bullet points, code blocks, etc.)?
6. **Coherence**: Is it logically consistent and well-structured?

Each dimension gets its own reward model trained on targeted preference data or demonstrations.

## Training compositional reward models

Challenge: How do we obtain training data for individual dimensions when human preferences are holistic?

**Approach 1: Targeted annotation**
- Ask annotators to rate responses on specific dimensions separately
- Train each reward model on its corresponding dimension annotations
- More expensive but produces cleanest decomposition

**Approach 2: Synthetic decomposition**
- Train a monolithic reward model first
- Generate explanations for why responses are preferred: "Response A is better because it's more concise and factually accurate"
- Use LLM-based parsing to assign credit to specific dimensions
- Train dimension-specific reward models using decomposed signals

**Approach 3: Contrastive examples**
- Create synthetic response pairs that differ primarily on one dimension
  - Same content, different length → train conciseness reward
  - Same answer, different formatting → train format reward
  - Correct vs. incorrect facts → train honesty reward
- Use these contrastive pairs to train dimension-specific reward models

## Combining rewards for policy training

Once we have compositional reward models, how do we use them for RL?

**Static weighting**:
```
R_total(x,y) = Σ wᵢ · Rᵢ(x,y)
```
Choose weights {wᵢ} based on:
- Validation set performance on desired trade-offs
- Human preference studies on weighted outputs
- Task-specific requirements (safety-critical vs. creative tasks)

**Dynamic weighting**:
- Learn weights that adapt per-prompt: w = f(prompt)
- Some prompts need high safety weight, others need high helpfulness
- Train a meta-model to predict optimal weights for each prompt type

**Constraint-based**:
- Hard constraints on certain dimensions (safety must be > threshold)
- Soft optimization of other dimensions subject to constraints
- Inspired by constrained RL ([Achiam et al., 2017](https://arxiv.org/abs/1705.10528))

## Key research questions

- Can we reliably decompose holistic preferences into orthogonal dimensions?
- Do compositional reward models reduce reward hacking compared to monolithic models?
- How sensitive is policy performance to the choice of weights {wᵢ}?
- Can we learn optimal weights automatically from user feedback?
- Do dimension-specific reward models provide better interpretability for debugging?
- Does decomposition help with distribution shift (easier to update one dimension)?

## Practical benefits

This approach enables:

1. **Flexible deployment**: Adjust weights post-training for different use cases
   - High-safety mode for sensitive applications
   - High-creativity mode for brainstorming
   - Balanced mode for general use

2. **Incremental improvement**: Update individual reward components without full retraining
   - New safety data → retrain only R_safe
   - Improved fact-checking → update only R_honest

3. **Debugging and interpretability**: When policy fails, identify which reward component is responsible
   - "The model is being too verbose" → check R_conciseness weight
   - "The model is giving unsafe advice" → check R_safe scores

4. **User control**: Let users adjust trade-offs according to their preferences
   - API parameter: `weights={'helpful': 1.0, 'concise': 0.5}`

## Implementation with Tinker Cookbook

Using the [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook):

1. Collect or synthesize dimension-specific preference data
2. Train multiple reward models in parallel (one per dimension)
3. Implement weighted reward aggregation in RL training loop
4. Experiment with different weighting schemes and evaluate trade-offs
5. Ablation studies: compositional vs. monolithic reward models

## Extensions

- **Hierarchical decomposition**: Break dimensions into sub-dimensions (safety → toxicity, bias, dangerous advice)
- **Learned decomposition**: Instead of hand-specifying dimensions, learn latent factors that explain preference variance
- **Cross-dimension interactions**: Model dependencies between objectives (e.g., very concise responses may sacrifice helpfulness)
- **Per-user decomposition**: Combined with meta-learning for personalization, allow users to have custom dimension weights

This could make RLHF more modular, interpretable, and controllable, addressing some of the brittleness and opacity of current monolithic reward models.
