# Confidence-calibrated generation: Models that know when they're uncertain

LLMs generate confidently even when wrong, misleading users. Calibrated confidence—where model confidence matches actual correctness—is critical for trust and safety. This project trains models to accurately estimate and communicate their uncertainty, inspired by uncertainty quantification in ML ([Guo et al., 2017](https://arxiv.org/abs/1706.04599)).

## Problem: Overconfidence

**Current behavior**:
```
User: "What is the capital of Atlantis?"
Model: "The capital of Atlantis is Poseidon City" (confident but wrong)
```

**Desired**:
```
Model: "I'm not sure—Atlantis is a mythical place without a real capital"
```

**Challenges**:
- Models don't know what they don't know
- Training incentivizes confident generation
- No explicit uncertainty modeling

## Approaches

### Approach 1: Uncertainty token generation

**Method**:
Train model to generate uncertainty markers:
- "I'm confident that..."
- "I think, but I'm not certain..."
- "I don't know enough to answer this"

**Training**:
- Label responses by correctness
- Reward model for calibrated confidence expressions
- Penalize overconfidence on wrong answers

### Approach 2: Ensemble-based uncertainty

**Method**:
1. Generate multiple responses (vary temperature, sample)
2. Measure diversity/disagreement
3. High disagreement → High uncertainty
4. Communicate: "My answers vary, suggesting uncertainty"

**Benefits**: Automatic uncertainty from sampling

### Approach 3: Explicit confidence scores

**Method**:
Model outputs (response, confidence_score)
```
Response: "Paris is the capital of France"
Confidence: 0.95
```

**Training**:
- Calibration loss: Confidence should match accuracy
- Expected Calibration Error (ECE) minimization

### Approach 4: Conformal prediction

**Method**:
Statistical framework for uncertainty quantification
- Provide prediction sets instead of single answers
- Guarantee coverage: True answer in set with probability p

**Application to LLMs**: Generate multiple plausible answers with coverage guarantees

### Approach 5: RL with calibration rewards

**Method**:
- Reward accurate + calibrated responses
- Penalize overconfident errors more than uncertain errors
- Bonus for correctly expressing uncertainty

**Objective**:
```
R = Accuracy - λ·|Confidence - Correctness|
```

## Evaluation

### Metric 1: Calibration metrics

**Expected Calibration Error (ECE)**:
- Bin predictions by confidence
- Measure: Confidence vs. actual accuracy in each bin
- Lower ECE = better calibration

### Metric 2: Selective prediction

**Test**: Model can refuse to answer when uncertain
- Allow model to say "I don't know"
- Measure: Accuracy on answered questions vs. coverage

**Goal**: High accuracy on answered, appropriate refusal rate

### Metric 3: Reliability diagrams

**Visualize**: Plot confidence vs. accuracy
- Perfect calibration: y=x line
- Measure deviation from perfect calibration

### Metric 4: Uncertainty-based ranking

**Test**: Rank responses by model confidence
- Higher confidence → Should be more accurate
- Measure: Rank correlation

## Research questions

1. **Can we train calibrated LLMs?**
   - Is calibration achievable without sacrificing capability?

2. **What calibration method works best?**
   - Uncertainty tokens, ensembles, explicit scores?

3. **Does calibration transfer across domains?**
   - Train on QA, test on reasoning?

4. **Do users trust calibrated models more?**
   - User studies on trust

5. **Trade-off between capability and calibration?**
   - Does expressing uncertainty hurt performance?

## Applications

- **High-stakes domains**: Medical, legal (critical to know uncertainty)
- **Information retrieval**: Calibrated relevance scores
- **Education**: Teach students to reason about uncertainty
- **Debugging**: Identify where model is uncertain (target improvement)

## Extensions

- **Fine-grained uncertainty**: Per-sentence or per-token confidence
- **Explanation of uncertainty**: "I'm uncertain because [reason]"
- **Active learning**: Request human help on uncertain queries
- **Adversarial calibration**: Robust to confidence-attacking prompts
- **Multicalibration**: Calibrated across user subgroups
