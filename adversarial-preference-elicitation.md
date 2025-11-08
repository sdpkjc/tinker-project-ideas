# Adversarial preference elicitation for robust reward modeling

Standard reward modeling assumes that human preference data is clean and consistent. However, real preference data contains noise from multiple sources: annotator mistakes, ambiguous comparisons, subjective disagreements, and even adversarial labeling. Training on noisy preferences can lead to reward models that overfit to spurious patterns or exploit annotation artifacts.

Recent work on robust machine learning ([Madry et al., 2017](https://arxiv.org/abs/1706.06083)) and learning from noisy labels ([Zhang & Sabuncu, 2018](https://arxiv.org/abs/1805.07836)) shows that adversarial training and noise-robust techniques can improve model robustness. This project applies these ideas to reward modeling, making RLHF more robust to imperfect human feedback.

## Problem: Noise in preference data

Types of noise in preference labels:

1. **Random noise**: Annotators make mistakes or are inconsistent
2. **Systematic bias**: Annotators prefer certain response styles regardless of quality (e.g., longer responses, responses with bullet points)
3. **Ambiguous comparisons**: Responses are truly similar in quality, but annotators must choose
4. **Adversarial corruption**: Malicious annotators or data poisoning attacks

Standard reward model training treats all labels as equally reliable, which can lead to:
- Overfitting to annotation artifacts
- Reward hacking that exploits annotator biases
- Brittleness to distribution shift in annotation quality

## Proposed approach: Adversarial preference elicitation

Instead of passively accepting noisy labels, actively identify and mitigate label noise during training:

### Method 1: Confidence-weighted training

1. Train an initial reward model on all preference data
2. Identify potentially noisy labels:
   - High model uncertainty (ensemble disagreement)
   - Inconsistent with nearby data points
   - Low annotator confidence scores (if available)
3. Re-weight training loss by label confidence:
   - L = Σ wᵢ · loss(yᵢ, ŷᵢ) where wᵢ = f(confidence_i)
4. High-confidence labels get more weight, low-confidence get less

### Method 2: Adversarial label perturbation

Inspired by adversarial training, make the reward model robust to label flips:

1. During training, randomly flip a fraction of labels (response A preferred → response B preferred)
2. Train the reward model to be robust to these perturbations
3. This forces the model to learn from robust features rather than memorizing noisy labels
4. Gradually reduce perturbation rate as training progresses

### Method 3: Multi-annotator disagreement modeling

When multiple annotators label the same comparison:

1. Model annotator-specific biases explicitly
   - Each annotator has a learned "bias vector"
   - Reward = base_reward + annotator_bias
2. Train reward model to predict consensus preference while accounting for individual biases
3. At deployment, use the base_reward without annotator-specific components

This is inspired by [Learning from Disagreement](https://arxiv.org/abs/2202.01034) and crowdsourcing literature.

### Method 4: Preference consistency regularization

Add regularization terms that enforce consistency:

1. **Transitivity**: If A > B and B > C, then A > C
   - Add soft constraint: R(A) > R(C) when transitive chain exists
2. **Symmetry**: Flipping order should flip preference
   - R(A|B) = -R(B|A)
3. **Calibration**: Reward magnitude should correlate with annotator confidence
   - High confidence comparisons should have larger reward gaps

## Active learning for preference elicitation

Proactively improve preference data quality:

1. **Uncertainty sampling**: Request human annotations on examples where the reward model is most uncertain
2. **Adversarial sampling**: Generate response pairs specifically designed to be confusing or ambiguous
3. **Disagreement sampling**: When multi-annotator disagreement is high, collect more labels
4. **Strategic re-annotation**: Re-label examples that appear noisy or inconsistent

This creates a feedback loop: identify uncertain/noisy regions → collect better data → improve reward model.

## Key research questions

- How much label noise exists in real preference datasets, and what are the primary sources?
- Do noise-robust training methods improve reward model accuracy on clean test sets?
- Does robustness to annotation noise reduce reward hacking during RL?
- Can we automatically detect adversarial or malicious annotations?
- How does annotator disagreement correlate with genuine preference ambiguity?
- Does active learning for preference elicitation improve data efficiency?

## Evaluation

1. **Synthetic noise**: Artificially corrupt a clean preference dataset with known noise, measure robustness
2. **Multi-annotator analysis**: Use datasets with multiple annotations per example to study disagreement
3. **Out-of-distribution**: Test reward models on held-out annotation sources (new annotators, different demographics)
4. **Downstream policy quality**: Do robust reward models lead to better policies after RL?

## Practical implementation

Using preference datasets with annotator metadata:
- Train baseline reward model on all data
- Train robust variants using methods above
- Compare reward model accuracy, calibration, and robustness
- Evaluate downstream RL policies trained with each reward model

## Connection to broader AI safety

Robust reward learning is critical for AI alignment:
- As we scale to superhuman AI, human feedback becomes noisier (humans evaluating capabilities beyond their own)
- Adversarial data poisoning is a real threat for deployed systems
- Learning robust values from imperfect feedback is a fundamental challenge

This project addresses the practical challenge of learning from realistic, noisy human preferences while maintaining robustness.

## Extensions

- **Uncertainty quantification**: Provide confidence intervals on reward predictions
- **Anomaly detection**: Automatically flag suspicious preference patterns
- **Robust active learning**: Combine noise robustness with strategic data collection
- **Debiasing**: Identify and correct for systematic annotator biases (length bias, recency bias, etc.)

By making reward learning robust to imperfect human feedback, we can build more reliable RLHF systems that don't exploit noise or overfit to annotation artifacts.
