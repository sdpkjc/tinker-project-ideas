# Using evaluator agreement patterns as a quality signal for RLHF

When collecting human preferences for RLHF, we typically aggregate multiple annotators' judgments and train on the consensus. However, the pattern of agreement and disagreement across annotators contains valuable information that is usually discarded. High agreement may indicate clear quality differences, while disagreement may indicate ambiguity, subjectivity, or genuinely close quality.

This project explores using annotator agreement patterns as an additional signal for reward modeling and RL training, inspired by work on learning from disagreement ([Gordon et al., 2021](https://arxiv.org/abs/2110.05719)) and uncertainty-aware learning.

## Core insight

**Current practice**:
- Collect preferences from multiple annotators
- Aggregate to single label (majority vote or average)
- Train reward model on aggregated labels
- Discard information about agreement patterns

**Proposed**:
- Treat agreement level as informative signal about quality landscape
- High agreement → Clear quality difference, high confidence
- Low agreement → Ambiguous quality, low confidence or subjective preference
- Use this to improve reward modeling and training

## Types of disagreement and their meanings

**Type 1: Random noise**
- Annotators make occasional mistakes
- No systematic pattern to disagreement
- Should reduce confidence but not change preference

**Type 2: Subjective preferences**
- Responses are genuinely similar in quality
- Different annotators have different tastes (verbosity, style, tone)
- Should recognize as multi-modal preference distribution

**Type 3: Difficulty/expertise gap**
- Some annotators can distinguish quality differences, others can't
- Disagreement indicates difficulty of comparison
- Should identify and upweight expert annotators

**Type 4: Ambiguous cases**
- Both responses have trade-offs (one more accurate, other more helpful)
- Disagreement reflects genuine tension between objectives
- Should recognize as multi-objective optimization challenge

## Proposed approaches

### Approach 1: Uncertainty-calibrated reward models

Train reward models to output both preference and uncertainty:

**Model output**:
- Reward score: R(x, y)
- Uncertainty: σ(x, y)

**Training objective**:
- Standard preference loss on mean reward
- Uncertainty loss: σ should match annotator disagreement level
  - High agreement → Low σ
  - High disagreement → High σ

**Use in RL**:
- Prioritize optimizing high-confidence regions
- Avoid overoptimizing uncertain regions (likely to reward hack)
- Bonus: Provides natural exploration signal

### Approach 2: Multi-modal reward modeling

Instead of single reward score, model multiple reward modes:

**Idea**: Different annotators may represent different reasonable preferences

**Implementation**:
- Cluster annotators by their preference patterns
- Train separate reward models for each cluster
- Use mixture of experts or ensemble at inference

**Benefits**:
- Captures diversity in human preferences
- Allows personalization (weight towards specific annotator cluster)
- More robust than single monolithic reward

### Approach 3: Agreement-weighted training

Weight training examples by annotator agreement:

**High agreement examples**:
- Strong, confident training signal
- Higher weight in loss function
- Model should be confident here

**Low agreement examples**:
- Weak, uncertain training signal
- Lower weight in loss function
- Or, train model to predict uncertainty explicitly

**Implementation**:
```
loss = Σ w_i · preference_loss(example_i)
where w_i = f(agreement_level_i)
```

### Approach 4: Disagreement-aware RL training

Use agreement patterns during RL:

**Strategy 1: Conservative optimization**
- In high-agreement regions: Optimize aggressively (clear signal)
- In low-agreement regions: Optimize conservatively (uncertain signal)

**Strategy 2: KL penalty modulation**
- High agreement: Allow larger KL divergence (confident we're improving)
- Low agreement: Tighter KL penalty (less confident)

**Strategy 3: Exploration bonus**
- Low agreement regions may contain underexplored good solutions
- Add exploration bonus to encourage diversity

## Experimental framework

### Data collection

Use preference datasets with multiple annotators per comparison:
- Anthropic HH dataset (has multiple annotators)
- OpenAI summarization dataset (has multiple annotators)
- Or collect new multi-annotator preference data

**Key requirement**: Need multiple independent judgments per comparison to measure agreement

### Analysis 1: What does disagreement predict?

**Questions**:
- Does low agreement correlate with:
  - Response similarity?
  - Task difficulty?
  - Domain ambiguity?
  - Response length?
- Can we predict agreement level from response features?

**Method**: Train classifier to predict agreement from (prompt, response_A, response_B)

### Analysis 2: Does uncertainty calibration improve reward models?

**Experiment**:
- Train baseline reward model (no uncertainty)
- Train uncertainty-calibrated reward model (outputs mean and variance)
- Compare on:
  - Accuracy on held-out preferences
  - Calibration (predicted uncertainty vs. actual disagreement)
  - Robustness (performance on ambiguous cases)

**Hypothesis**: Uncertainty-aware models are better calibrated and more robust

### Analysis 3: Does agreement-weighting improve RL?

**Experiment**:
- Train policies with standard RLHF (no agreement weighting)
- Train policies with agreement-weighted RLHF
- Compare:
  - Final performance
  - Training stability
  - Reward hacking incidence
  - Robustness to distribution shift

**Hypothesis**: Agreement-weighted RL is more stable and less prone to overfitting

### Analysis 4: Multi-modal preferences

**Experiment**:
- Cluster annotators by preference patterns
- Train separate reward models per cluster
- Evaluate: Can we identify coherent preference subgroups?
- Test: Does ensemble of cluster-specific RMs outperform single RM?

## Key research questions

1. **What information does disagreement contain?**
   - Is it mostly noise or meaningful signal?
   - Can we distinguish noise from genuine ambiguity?

2. **Can we improve reward models using disagreement?**
   - Does uncertainty calibration help?
   - Does agreement weighting improve accuracy?

3. **Should we treat disagreement as noise or signal?**
   - Downweight (noise perspective)
   - Model explicitly (signal perspective)
   - Both are valid—which works better?

4. **Does this reduce reward hacking?**
   - Hypothesis: Reward hacking often occurs in low-agreement regions
   - Does uncertainty-awareness prevent this?

5. **Can we personalize using annotator clusters?**
   - Do different annotators represent different user populations?
   - Can we serve different users with cluster-specific policies?

## Practical implementation

**Phase 1: Data analysis**
- Take existing multi-annotator preference data
- Analyze agreement patterns and correlates
- Build intuition about what disagreement means

**Phase 2: Reward modeling**
- Implement uncertainty-calibrated reward model
- Train on multi-annotator data
- Evaluate calibration and accuracy

**Phase 3: RL training**
- Integrate uncertainty into RL training loop (e.g., via PPO modifications)
- Compare against standard RLHF baseline
- Measure improvements in stability and robustness

## Expected outcomes

**If successful**:
- Better calibrated reward models (know when they're uncertain)
- More stable RL training (avoid overoptimizing uncertain regions)
- Reduced reward hacking (harder to exploit uncertain rewards)
- Path to personalization (model preference diversity)

**If unsuccessful**:
- Still learn valuable insights about human preference structure
- Understand limitations of disagreement as a signal
- Inform future data collection strategies

## Connection to related work

- **Uncertainty quantification**: Methods from probabilistic ML
- **Learning from disagreement**: NLP work on annotator variation
- **Multi-task learning**: Treating annotators as related tasks
- **Robust optimization**: RL under reward uncertainty

This project bridges these areas in the context of RLHF.

## Extensions

**Temporal dynamics**:
- How does individual annotator reliability change over time?
- Can we track and adapt to annotator drift?

**Active learning**:
- Request additional annotations for high-disagreement cases
- Efficiently reduce uncertainty where it matters

**Interpretability**:
- What features cause disagreement?
- Can we explain why annotators disagree?

**Synthetic disagreement**:
- Generate diverse preferences from single annotator
- Use disagreement simulation for training

By utilizing the full richness of multi-annotator preference data—not just the aggregated labels—we can build more robust, calibrated, and effective RLHF systems.
