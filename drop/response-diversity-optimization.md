# Optimizing for response diversity in RLHF: Preventing mode collapse

Standard RLHF optimization often leads to mode collapse: the policy generates repetitive, formulaic responses that maximize reward but lack diversity and creativity. While these responses may score well on reward models, they provide poor user experience—users want varied, interesting outputs, not templated answers.

This project explores methods to maintain response diversity during RLHF, balancing reward optimization with diversity objectives to produce capable and interesting models.

## Problem: Mode collapse in RLHF

**Observation**: After RLHF, models often generate stereotyped outputs:
- Always start with "Certainly! Here's..."
- Overuse certain phrases reward model associates with quality
- Generate similar structure across diverse prompts
- Lose creativity and variety from base model

**Why this happens**:
- RL objective: Maximize reward → Policy exploits RM quirks
- No explicit diversity incentive → Converges to high-reward templates
- Safe, generic responses score well → Specificity punished

**User impact**:
- Boring, repetitive interactions
- Loss of model personality
- Reduced usefulness for creative tasks

**Goal**: Maintain diversity while improving quality through RLHF

## Proposed approaches

### Approach 1: Diversity-regularized RL objective

Add explicit diversity term to RL objective:

**Modified objective**:
```
J = E[R(x, y)] + α·Diversity(y | x) - β·KL(π || π_ref)
```

**Diversity metrics**:
- **Self-BLEU**: Lower self-similarity across responses
- **N-gram diversity**: More unique n-grams
- **Embedding variance**: Diverse semantic content
- **Structural diversity**: Varied formatting, length, style

**Implementation**:
During RL training, sample K responses per prompt, measure diversity, add diversity reward

**Benefits**:
- Direct optimization for diversity
- Tunable (adjust α to balance reward vs. diversity)
- Compatible with existing RLHF pipeline

**Challenges**:
- Defining "good" diversity (not just noise)
- Computational cost (need multiple samples)
- Balancing with quality

### Approach 2: Ensemble policy training

Train ensemble of diverse policies and sample from mixture:

**Method**:
1. Train N policies with different:
   - Random seeds
   - Training data subsets
   - Hyperparameters
   - Architectures (if feasible)

2. At inference, randomly select policy from ensemble or mix their outputs

**Benefits**:
- Natural diversity from different policies
- Each policy can specialize (different response styles)
- Modular (add/remove policies)

**Challenges**:
- N× training cost
- N× inference cost (or need routing mechanism)
- Ensuring all policies maintain quality

**Optimization**: Distill ensemble into single diverse policy

### Approach 3: Conditional diversity through style tokens

Condition generation on style/diversity tokens:

**Architecture**:
```
[style_token] + prompt → policy → response
```

**Style tokens**:
- `<formal>`, `<casual>`, `<creative>`, `<concise>`, etc.
- Randomly sampled during training
- Model learns to generate diverse styles

**Training**:
1. Augment data with style labels (automated or manual)
2. Train policy conditioned on style
3. At inference, sample different styles for diversity

**Benefits**:
- Single model produces diverse outputs
- Controllable diversity (users can specify style)
- Efficient (no ensemble needed)

### Approach 4: Best-of-N with diversity filtering

During inference, generate diverse candidate set:

**Method**:
1. Generate N candidates with high temperature/diverse sampling
2. Filter for diversity: Remove very similar candidates
3. Score remaining candidates with reward model
4. Return highest-scoring diverse candidate

**Benefits**:
- No training changes (works with any RLHF model)
- Preserves quality (still use reward model)
- Guarantees diversity at inference

**Challenges**:
- Higher inference cost (generate N candidates)
- Diversity filtering logic needs tuning

### Approach 5: Adversarial diversity training

Use adversarial approach to encourage diversity:

**Setup**:
- **Generator** (policy): Tries to generate diverse, high-reward responses
- **Discriminator**: Detects repetitive or template-like responses

**Training**:
- Generator maximizes: Reward - Discriminator(is_template)
- Discriminator learns to identify low-diversity outputs
- Arms race drives toward diverse, high-quality generations

**Benefits**:
- Automatic discovery of diversity failures
- No need to specify diversity metric manually
- Adapts to model's specific mode collapse patterns

### Approach 6: Entropy regularization

Maintain policy entropy to prevent over-optimization:

**Method**:
Add entropy bonus to RL objective:
```
J = E[R(x, y)] + α·H(π(y|x))
```

Where H is entropy of policy's output distribution

**Benefits**:
- Prevents policy from becoming too deterministic
- Standard technique in RL
- Maintains exploration

**Challenges**:
- Token-level entropy may not capture response-level diversity
- Can lead to incoherent outputs if α too large

## Evaluation methodology

### Metric 1: Diversity metrics

**Intra-prompt diversity**: For same prompt, generate K responses, measure:
- Self-BLEU (lower is more diverse)
- Distinct-N (fraction of unique n-grams)
- Embedding variance (semantic diversity)
- Length variance
- Structural diversity (formatting, organization)

**Inter-prompt diversity**: Across different prompts:
- Do responses avoid templates?
- Measure: Template detection rate

### Metric 2: Quality-diversity trade-off

**Critical**: Diversity shouldn't sacrifice quality

**Measure**:
- Plot: Diversity score vs. Reward model score
- Also: Diversity vs. Human preference win rate
- Find: Pareto frontier (good quality + good diversity)

### Metric 3: User satisfaction

**A/B test**:
- Model A: Standard RLHF (potentially less diverse)
- Model B: Diversity-optimized RLHF
- Measure: User ratings, engagement, task success

**Question**: Do users prefer diverse models?

### Metric 4: Creative task performance

**Specific evaluation**: Tasks requiring diversity
- Brainstorming (generate 10 ideas)
- Story writing (multiple plot variations)
- Code solutions (different algorithmic approaches)

**Measure**: Quality and diversity of outputs

### Metric 5: Template detection

**Method**:
- Train classifier to detect templated/generic responses
- Test models: What fraction of responses are templated?
- Lower is better (more genuine diversity)

## Experimental framework

### Experiment 1: Diversity method comparison

**Compare**:
- Baseline: Standard RLHF
- Diversity regularization
- Ensemble
- Style conditioning
- Best-of-N with diversity filtering
- Adversarial training
- Entropy regularization

**Measure**: Quality-diversity Pareto curve for each

### Experiment 2: Diversity hyperparameter tuning

**Question**: What diversity weight (α) is optimal?

**Method**:
- Train with varying α values
- Plot: α vs. {diversity, quality, user satisfaction}
- Find: Sweet spot

### Experiment 3: Task-specific diversity needs

**Hypothesis**: Different tasks need different diversity levels

**Test**:
- Factual QA (lower diversity OK, correctness matters)
- Creative writing (high diversity essential)
- Code generation (moderate diversity, correctness critical)

**Find**: Optimal diversity level per task type

### Experiment 4: Long-term diversity tracking

**Question**: Does diversity degrade over training?

**Method**:
- Track diversity metrics at each training checkpoint
- Identify: When does mode collapse occur?
- Test interventions at different training stages

## Key research questions

1. **Is there a quality-diversity trade-off?**
   - Can we have both, or must we sacrifice one for the other?

2. **What causes mode collapse in RLHF?**
   - RM design? RL algorithm? Training dynamics?

3. **Which diversity method works best?**
   - Simple regularization? Complex adversarial training?

4. **Do users value diversity?**
   - Or prefer consistent, predictable responses?
   - Context-dependent?

5. **How much diversity is optimal?**
   - Too little: Boring
   - Too much: Incoherent
   - Finding the right level

## Practical implementation

**Phase 1**: Measure diversity in existing RLHF models
- Establish baseline diversity metrics
- Identify mode collapse patterns

**Phase 2**: Implement diversity-regularized RLHF
- Add diversity term to reward
- Tune hyperparameters

**Phase 3**: User studies
- Deploy diverse vs. standard models
- Collect preference data
- Iterate based on feedback

**Phase 4**: Task-adaptive diversity
- Different diversity levels for different domains
- Learned or user-specified

## Extensions

**Personalized diversity**:
- Some users prefer consistency, others variety
- Learn user-specific diversity preferences

**Contextual diversity**:
- Adjust diversity based on conversation history
- If user asked same question before, give different answer

**Controllable diversity slider**:
- API parameter: diversity={low, medium, high}
- Users choose their preferred level

**Diversity-aware reward models**:
- Train RMs that value diversity
- Incorporate novelty and variety into preferences

**Multi-turn diversity**:
- Avoid repeating across conversation turns
- Memory of past responses influences current generation

This research addresses a key quality-of-life issue in deployed RLHF systems: making models that are not only capable but also interesting and varied, improving user experience and expanding use cases to creative domains.
