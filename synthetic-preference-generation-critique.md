# Synthetic preference generation via model-based critique and comparison

RLHF requires large amounts of preference data, which is expensive to collect from humans. One promising approach is using AI-generated preferences (RLAIF, as in [Constitutional AI](https://arxiv.org/abs/2212.08073)), but current methods have limitations: they often require strong prompting, may inherit biases from the generating model, and lack fine-grained quality control.

This project explores advanced methods for synthetic preference generation using structured critique, comparative analysis, and multi-model consensus, aiming to produce higher-quality and more reliable synthetic preferences than naive prompting.

## Limitations of current synthetic preference methods

**Simple RLAIF approach**:
```
Prompt: "Which response is better? A or B?"
Model: "A is better because..."
```

**Problems**:
- Model may have biases or blind spots
- Judgments can be inconsistent
- Hard to control for specific quality dimensions
- No verification or confidence estimation

**Desired improvements**:
- More structured and explainable judgments
- Multi-dimensional quality assessment
- Consistency checking and uncertainty quantification
- Controllable focus (helpfulness vs. safety vs. accuracy)

## Proposed approaches

### Approach 1: Multi-step structured critique

Instead of direct comparison, decompose judgment into steps:

**Step 1: Individual critique**
For each response independently:
- What are the strengths?
- What are the weaknesses?
- What specific errors or issues exist?
- Rate on multiple dimensions (accuracy, helpfulness, safety, clarity)

**Step 2: Comparative analysis**
Compare the critiques:
- Which response has more strengths?
- Which has fewer or less severe weaknesses?
- Are there trade-offs (one more accurate, other more helpful)?

**Step 3: Final judgment**
- Overall preference with confidence level
- Explanation grounded in the critiques

**Benefits**:
- More thorough and consistent
- Provides explanations for preferences
- Identifies trade-offs explicitly
- Can weight dimensions differently for different tasks

### Approach 2: Multi-model consensus

Use multiple models to generate preferences and aggregate:

**Method**:
1. Generate preferences from multiple models (different sizes, families, or prompting strategies)
2. For each comparison, collect judgments from all models
3. Aggregate:
   - **High consensus** (models agree): High-confidence preference
   - **Partial consensus** (majority agrees): Medium-confidence preference
   - **Low consensus** (models split): Low-confidence or tie
4. Train reward model with confidence weighting

**Benefits**:
- More robust than single-model judgments
- Uncertainty quantification from disagreement
- Reduces impact of individual model biases

**Variants**:
- Different model sizes (small, medium, large)
- Different model families (Llama, GPT, Claude, etc.)
- Same model with different prompts or temperatures

### Approach 3: Adversarial critique and defense

Use debate-style interaction to generate more robust judgments:

**Method**:
1. **Advocate A**: Argues why response A is better
2. **Advocate B**: Argues why response B is better
3. **Judge**: Evaluates both arguments and decides
4. Multiple rounds of argument and counter-argument
5. Final judgment based on strongest case

**Benefits**:
- Adversarial process surfaces issues that single-pass critique might miss
- Similar to [debate for alignment](https://arxiv.org/abs/1805.00899)
- Can catch subtle errors through disagreement

**Implementation**:
- Use same model with different prompts (advocate vs. judge)
- Or use different models in different roles
- Record full debate as rationale for preference

### Approach 4: Decomposed aspect-based preferences

Instead of single holistic preference, generate preferences for specific aspects:

**Dimensions**:
- Factual accuracy
- Helpfulness for user's goal
- Safety (no harmful content)
- Clarity and coherence
- Appropriate level of detail
- Proper formatting

**Method**:
1. For each dimension, generate dimension-specific preference
2. For each dimension, provide explanation and confidence
3. Aggregate to overall preference (with weighted combination)

**Benefits**:
- More fine-grained than holistic judgment
- Can identify trade-offs (A more accurate but B more helpful)
- Enables multi-objective reward modeling
- Better handles complex judgments

### Approach 5: Self-consistency verification

Check consistency of synthetic preferences:

**Method**:
1. Generate preference: A > B
2. Generate reverse comparison: "Is B > A?"
   - If model now says B > A, flag inconsistency
3. Generate related comparisons: "If A > B and B > C, is A > C?"
   - Check transitivity
4. Regenerate with different prompts
   - If judgments change significantly, flag as uncertain

**Filter**:
- Keep only consistent, high-confidence preferences
- Discard or lower weight for inconsistent judgments

**Benefits**:
- Quality control for synthetic preferences
- Identifies ambiguous cases where model is uncertain
- Reduces noise in training data

## Training strategies with synthetic preferences

### Strategy 1: Curriculum from human to synthetic

Start with human preferences, gradually introduce synthetic:

**Phase 1**: Train reward model on 100% human preferences
**Phase 2**: Mix 80% human + 20% high-confidence synthetic
**Phase 3**: Mix 50% human + 50% synthetic
**Phase 4**: Majority synthetic with human preference validation

**Hypothesis**: Gradual transition maintains quality while reducing cost

### Strategy 2: Bootstrapping loop

Iteratively improve synthetic preference quality:

1. Train reward model on initial synthetic preferences
2. Use reward model to filter/rank new synthetic preferences
3. Keep only synthetic preferences that align with learned reward
4. Retrain reward model on filtered data
5. Repeat

**Risk**: Amplifying initial errors
**Mitigation**: Maintain human preference validation set, don't drift

### Strategy 3: Hybrid human-synthetic

Use humans for ambiguous cases, synthetic for clear cases:

1. Generate synthetic preferences for all comparisons
2. Estimate uncertainty/difficulty for each comparison
3. High-uncertainty cases → Human annotation
4. Low-uncertainty cases → Use synthetic

**Benefits**:
- Allocates expensive human feedback efficiently
- Synthetic preferences handle "easy" comparisons
- Humans focus on important edge cases

## Experimental framework

### Experiment 1: Synthetic vs. human preference quality

**Setup**:
1. Collect both human and synthetic preferences on same response pairs
2. Train reward models on:
   - 100% human preferences (gold standard)
   - 100% synthetic preferences (current methods)
   - 100% synthetic with proposed improvements
   - Mixture of human and synthetic

**Evaluate**:
- Reward model accuracy on held-out human preferences
- Policy quality after RLHF with each reward model
- Human evaluation of final policies

**Question**: How much quality gap between human and synthetic? Can our methods close it?

### Experiment 2: Ablation of synthetic preference methods

Test each proposed method individually:
- Baseline: Simple prompted comparison
- +Structured critique
- +Multi-model consensus
- +Adversarial debate
- +Aspect-based decomposition
- +Self-consistency filtering

**Measure**: How much does each component improve quality?

### Experiment 3: Scaling synthetic preferences

**Question**: Can synthetic preferences fully replace human preferences at sufficient scale?

**Test**:
- Train RMs on 10K human preferences
- Train RMs on 10K, 50K, 100K, 500K synthetic preferences
- Compare final policy quality

**Hypothesis**: With enough high-quality synthetic data, can match or exceed human preference training

### Experiment 4: Domain-specific synthetic preferences

Test on diverse domains:
- Instruction following (general)
- Math reasoning (verifiable)
- Creative writing (subjective)
- Coding (partially verifiable)

**Question**: Where do synthetic preferences work best/worst?

## Key research questions

1. **Can synthetic preferences match human preference quality?**
   - Under what conditions?
   - How much more data is needed?

2. **What methods most improve synthetic preference quality?**
   - Structured critique? Multi-model consensus? Debate?
   - Are they complementary or redundant?

3. **How do synthetic preferences scale?**
   - Does quality improve with more diverse generating models?
   - Diminishing returns or compound improvements?

4. **What are failure modes of synthetic preferences?**
   - Systematic biases?
   - Overconfidence on certain error types?
   - How to detect and mitigate?

5. **Can we automate quality control?**
   - Automatically filter low-quality synthetic preferences?
   - Active learning: identify which comparisons need human input?

## Cost-benefit analysis

**Costs of synthetic preferences**:
- Inference compute for generating judgments
- Potentially lower quality than human preferences
- Risk of amplifying model biases

**Benefits**:
- Much cheaper than human annotation (especially at scale)
- Can generate preferences for any domain/language
- Faster iteration cycles
- Can target specific quality dimensions

**Break-even point**: At what quality level are synthetic preferences worthwhile?

## Extensions

**Personalized synthetic preferences**:
- Generate preferences matching different user populations
- Use demographic or preference data to condition generation

**Multilingual synthetic preferences**:
- Generate preferences for low-resource languages
- Translate or directly generate in target language

**Continual improvement**:
- As models improve, regenerate synthetic preferences
- Older synthetic data may become outdated

**Human-in-the-loop synthesis**:
- Use human feedback to guide synthetic preference generation
- Learn what makes human and synthetic preferences agree/disagree

This project could dramatically reduce the cost of RLHF while maintaining quality, making advanced post-training accessible for more domains, languages, and organizations.
