# Weak-to-strong generalization in preference learning

As language models become more capable, human evaluators increasingly struggle to accurately judge their outputs—particularly in domains requiring expertise (advanced math, complex code, specialized knowledge). This creates a fundamental challenge: how can we train superhuman models using feedback from less capable evaluators?

OpenAI's recent work on [weak-to-strong generalization](https://arxiv.org/abs/2312.09390) shows that strong models can be trained using weak model supervision in some settings. This project explores weak-to-strong generalization specifically for preference learning and RLHF.

## Core problem

**Standard RLHF assumption**: Human evaluators can reliably judge output quality

**Reality**: As models approach or exceed human capability:
- Humans can't evaluate complex reasoning chains
- Humans can't verify advanced code or math
- Humans fall back on surface features (length, formatting, confidence)

**Result**: Reward models trained on human preferences may not capture true quality, leading to misalignment or capability plateaus.

## Research question

Can we train capable reward models using preferences from weaker evaluators (human or AI), such that the resulting reward model generalizes beyond the evaluator's capability?

## Proposed approaches

### Approach 1: Confidence-aware preference learning

Weak evaluators are more reliable on easy examples than hard ones.

**Method**:
1. Collect preferences from weak evaluator (human or small model)
2. Estimate evaluator confidence or reliability for each judgment
   - Easy comparisons (clearly better/worse): High confidence
   - Hard comparisons (subtle differences): Low confidence
3. Train reward model to extrapolate from high-confidence to low-confidence regions
   - Weight training loss by confidence
   - Use semi-supervised learning on low-confidence examples

**Hypothesis**: Reward model learns general quality principles from easy examples and applies them to hard examples where evaluator is uncertain.

### Approach 2: Consistency-based bootstrapping

Multiple weak evaluators may disagree on hard examples, but their disagreements contain signal.

**Method**:
1. Collect preferences from multiple weak evaluators
2. For examples where evaluators agree: High-confidence training signal
3. For examples where evaluators disagree:
   - Train reward model to predict consensus or majority vote
   - Use ensemble methods to extract signal from disagreement
4. Bootstrap: Use reward model predictions on hard examples to improve itself

**Hypothesis**: Even if individual evaluators are weak, aggregate signal contains information about true quality.

### Approach 3: Auxiliary verifiable tasks

Use verifiable subtasks as scaffolding for non-verifiable quality.

**Method**:
1. Decompose responses into verifiable and non-verifiable aspects
   - Verifiable: Logical consistency, factual accuracy (when checkable), formatting correctness
   - Non-verifiable: Helpfulness, creativity, appropriateness
2. Train reward model on verifiable aspects with strong supervision
3. Learn correlation between verifiable quality and overall quality
4. Generalize to non-verifiable aspects based on learned correlations

**Example**:
- Verify that math reasoning is logically valid (verifiable)
- Assume valid reasoning correlates with correct final answer
- Generalize to problems where we can't verify the answer

### Approach 4: Conservative extrapolation

Train reward model to be cautious about overconfident predictions beyond evaluator capability.

**Method**:
1. Identify boundary of weak evaluator's reliable judgment
   - Test evaluator on problems of varying difficulty
   - Find threshold where reliability drops
2. Train reward model with uncertainty calibration
   - High confidence within reliable region
   - Increasing uncertainty beyond reliable region
3. Use conservative estimates for RL training
   - Prefer robustly-judged responses
   - Avoid overoptimizing on uncertain regions

**Hypothesis**: Better to be uncertain and robust than overconfident and wrong.

## Experimental framework

### Setup: Weak and strong models

**Weak evaluator**:
- Small language model (7B) providing preferences
- Or humans with limited expertise/time

**Strong model**:
- Large language model (70B+) being trained

**Ground truth**:
- Automated verification (math, code)
- Or expert human evaluation (expensive, used only for testing)

### Experiment 1: Capability gap testing

Vary the gap between weak evaluator and strong model:
- Small gap: 7B evaluator → 13B model (should work)
- Medium gap: 7B evaluator → 34B model (challenging)
- Large gap: 7B evaluator → 70B model (very challenging)

**Measure**: How well does reward model trained on weak preferences perform on strong model outputs?

**Metric**: Correlation between learned reward and ground truth quality

### Experiment 2: Domain difficulty scaling

Test on domains with varying verifiability:
- Easy: Math with automated grading (ground truth available)
- Medium: Code with fuzzy test cases (partial ground truth)
- Hard: Creative writing or open-ended reasoning (no ground truth)

**Question**: Does weak-to-strong generalization work better when ground truth is partially available?

### Experiment 3: Evaluator reliability simulation

Artificially degrade evaluator reliability:
- High reliability: 90% accuracy on preference judgments
- Medium reliability: 70% accuracy
- Low reliability: 55% accuracy (barely better than random)

**Question**: How much evaluator reliability is needed for weak-to-strong generalization to succeed?

### Experiment 4: Bootstrapping iterations

Iteratively improve reward model:
1. Train reward model on weak preferences
2. Use reward model to generate high-quality examples
3. Have weak evaluator label these examples
4. Retrain reward model on expanded dataset
5. Repeat

**Question**: Does bootstrapping amplify or compound errors?

## Key research questions

1. **Is weak-to-strong generalization possible in preference learning?**
   - Can reward models learn to judge quality beyond their training signal?

2. **What conditions enable it?**
   - Amount of data, capability gap, domain properties, evaluator reliability

3. **What are the failure modes?**
   - Does reward model just learn evaluator biases and scale them up?
   - Does it plateau at evaluator capability ceiling?
   - Does it extrapolate incorrectly and reward bad outputs?

4. **Can we detect when generalization fails?**
   - Warning signals that reward model is unreliable
   - Uncertainty quantification for out-of-distribution quality

5. **How does this compare to scalable oversight approaches?**
   - Debate, recursive reward modeling, market-based approaches
   - Is weak-to-strong generalization complementary or alternative?

## Evaluation methodology

**Quantitative**:
- Reward model accuracy on strong model outputs (vs. ground truth)
- Policy performance after RL training with weak-supervision reward model
- Correlation between weak preferences and strong performance

**Qualitative**:
- Error analysis: Where does weak-to-strong generalization fail?
- Case studies: Specific examples where it works or doesn't work
- Comparison: Weak supervision vs. strong supervision (when available)

## Connection to AI safety and alignment

This is a crucial problem for long-term AI safety:
- As AI becomes superhuman, human feedback becomes unreliable
- Need methods to align superhuman AI without superhuman evaluators
- Weak-to-strong generalization is one potential solution

This project provides empirical grounding for this theoretical concern and tests whether proposed solutions actually work.

## Practical applications

**Near-term**:
- Train advanced models using cheaper, weaker evaluators
- Reduce reliance on expensive expert evaluation
- Enable RLHF in domains where expert evaluation is scarce

**Long-term**:
- Path to aligning superhuman AI systems
- Scalable oversight without recursive complexity
- Robust to human evaluation errors and biases

## Extensions

**Hybrid approaches**:
- Combine weak-to-strong with other scalable oversight techniques
- Use debate or decomposition to help weak evaluators judge strong outputs

**Active learning**:
- Identify examples where weak evaluators are most informative
- Request expert evaluation only where weak signal is unreliable

**Theoretical analysis**:
- When is weak-to-strong generalization information-theoretically possible?
- Under what assumptions can we prove guarantees?

This project addresses one of the fundamental challenges in AI alignment: supervising systems that may be more capable than their supervisors, with direct practical relevance to current RLHF practice.
