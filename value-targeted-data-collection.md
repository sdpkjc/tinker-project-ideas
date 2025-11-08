# Value-targeted active learning for preference data collection

Collecting human preferences for RLHF is expensive—each comparison requires human time and attention. Standard practice collects preferences uniformly across prompts, but not all comparisons are equally valuable for learning. Active learning principles suggest we should strategically select which comparisons to label based on their expected information gain.

This project develops value-targeted active learning strategies for preference collection that maximize reward model quality per annotation dollar, inspired by [uncertainty sampling](https://arxiv.org/abs/1807.04801) and [Bayesian active learning](https://arxiv.org/abs/1112.5745).

## Problem: Inefficient preference collection

**Current practice**: Collect preferences uniformly
- Random or round-robin sampling of prompts
- Generate response pairs, ask humans to compare
- All comparisons weighted equally

**Inefficiency**:
- Easy comparisons (clear winner) provide little information
- Redundant comparisons (similar to existing data) waste resources
- Informative edge cases may be undersampled

**Cost**: Each comparison costs $0.10-$1.00 in annotator time

**Goal**: Collect preferences strategically to maximize RM quality per dollar spent

## Key insight: Not all comparisons are equally valuable

**High-value comparisons**:
- **Uncertain**: Reward model unsure which is better (high learning signal)
- **Diverse**: Cover underrepresented regions of task/response space
- **Discriminative**: Reveal fine-grained quality differences (not obvious cases)
- **High-stakes**: Important for deployment safety (edge cases, failures)

**Low-value comparisons**:
- **Obvious**: Clear winner, RM already confident (redundant signal)
- **Redundant**: Similar to many existing comparisons (no new information)
- **Out-of-distribution**: Irrelevant to deployment (wasted effort)

## Proposed active learning strategies

### Strategy 1: Uncertainty-based sampling

**Method**:
Sample comparisons where current RM is most uncertain:

**Algorithm**:
1. Train initial RM on small seed dataset
2. Generate many candidate response pairs
3. Score pairs with RM, compute uncertainty:
   - Option A: Prediction entropy
   - Option B: Ensemble disagreement
   - Option C: Prediction variance (Bayesian RM)
4. Select high-uncertainty pairs for human annotation
5. Retrain RM on augmented dataset
6. Repeat

**Benefit**: Focus annotations on informative cases

**Risk**: May focus on intrinsically ambiguous cases (where even humans uncertain)

### Strategy 2: Diversity-based sampling

**Method**:
Ensure coverage of diverse regions of task space:

**Algorithm**:
1. Embed prompts and responses in latent space
2. Cluster into regions
3. Sample comparisons to cover all clusters
4. Within each cluster, apply uncertainty sampling
5. Ensures both diversity AND uncertainty

**Benefit**: Balanced coverage + informativeness

**Metrics for diversity**:
- Prompt embedding diversity (cover different task types)
- Response embedding diversity (cover different response styles)
- Domain coverage (math, code, creative, etc.)

### Strategy 3: Disagreement-based sampling

**Method**:
Sample cases where multiple RMs (or humans) disagree:

**Algorithm**:
1. Train ensemble of reward models
2. Generate response pairs
3. Score with all ensemble members
4. Select pairs with high ensemble disagreement
5. Collect human preferences on these pairs
6. Retrain ensemble on augmented data

**Benefit**: Targets modeling blind spots and ambiguous cases

**Variant**: Use synthetic preferences from prompted LLMs as cheap proxy for ensemble

### Strategy 4: Policy-aware sampling

**Method**:
Prioritize comparisons relevant to current policy:

**Algorithm**:
1. Sample responses from current policy
2. Create comparisons from policy outputs
3. Focus annotations on: "What would help policy improve most?"
4. As policy evolves, re-prioritize annotations

**Benefit**: Directly targets policy improvement, efficient

**Inspired by**: Reinforcement learning with human feedback in robotics

### Strategy 5: Error-driven sampling

**Method**:
Sample comparisons that reveal reward model errors:

**Algorithm**:
1. Generate comparisons, score with RM
2. Collect human labels on subset
3. Identify cases where RM and humans disagree
4. Generate similar comparisons to those error cases
5. Collect more labels on error-prone regions

**Benefit**: Targets systematic RM failures

### Strategy 6: Cost-aware sampling

**Method**:
Account for varying annotation difficulty:

**Observation**: Some comparisons are harder to judge (take more time, lower agreement)

**Algorithm**:
1. Estimate annotation cost per comparison (time, difficulty)
2. Estimate information gain per comparison
3. Optimize: maximize information gain per unit cost
4. Select comparisons with best gain/cost ratio

**Benefit**: Explicit cost-benefit optimization

## Multi-objective active learning

Combine multiple criteria in selection:

**Scoring function**:
```
Value(comparison) = w1·Uncertainty(RM)
                   + w2·Diversity(from existing data)
                   + w3·PolicyRelevance(current policy)
                   + w4·Safety(high-stakes case)
                   - w5·Cost(annotation difficulty)
```

**Selection**:
- Score all candidate comparisons
- Select top-K by value score
- Send to human annotators

**Benefits**: Balances multiple desiderata

## Practical implementation

### Phase 1: Build candidate pool

**Method**:
Generate large pool of candidate response pairs:
- Sample diverse prompts from dataset
- Generate multiple responses per prompt (vary sampling params)
- Create pairs: (prompt, response_A, response_B)

**Size**: 10K-100K candidate pairs (much larger than annotation budget)

### Phase 2: Score candidates

**Criteria**:
For each candidate, compute:
- RM uncertainty score
- Diversity score (distance to nearest existing comparison)
- Policy relevance score (does current policy generate such responses?)
- Predicted annotation difficulty

### Phase 3: Select batch for annotation

**Budget**: Can afford N annotations (e.g., N=1000)

**Selection**:
- Rank candidates by value score
- Select top-N
- Send to annotators

### Phase 4: Update and iterate

**After collecting labels**:
1. Retrain reward model
2. Re-score remaining candidates (uncertainty changes)
3. Select next batch
4. Repeat until budget exhausted or performance saturates

## Evaluation methodology

### Experiment 1: Active vs. random sampling

**Setup**:
- Budget: 10K preference labels
- Random baseline: Sample 10K comparisons randomly
- Active treatment: Use active learning to select 10K comparisons

**Measure**:
- Final RM accuracy on held-out test set
- Policy quality after RLHF with each RM

**Expected**: Active learning achieves better RM with same budget

### Experiment 2: Data efficiency curves

**Question**: How many annotations does active learning save?

**Method**:
- Plot: Number of annotations vs. RM accuracy
- Compare random sampling vs. active learning
- Find: At what point do they reach same accuracy?

**Metric**: Data efficiency gain (e.g., "Active learning needs 50% fewer labels")

### Experiment 3: Selection strategy comparison

**Compare**:
- Uncertainty sampling
- Diversity sampling
- Disagreement sampling
- Policy-aware sampling
- Multi-objective combination

**Measure**: RM accuracy and policy quality for each strategy

### Experiment 4: Cold start problem

**Question**: Does active learning work with minimal seed data?

**Test**:
- Start with K seed labels (K = 100, 500, 1000)
- Apply active learning for remaining budget
- Measure: How does seed size affect active learning benefit?

## Key research questions

1. **How much can active learning improve data efficiency?**
   - 2×? 5×? 10× reduction in required labels?

2. **Which selection strategy works best?**
   - Uncertainty, diversity, disagreement, or combination?

3. **Does active learning improve policy quality?**
   - Not just RM accuracy, but final RLHF outcome?

4. **Are there failure modes?**
   - Can active learning lead to worse outcomes (e.g., oversampling ambiguous cases)?

5. **How does this scale?**
   - Does benefit persist at large annotation budgets?
   - Or diminishing returns?

## Practical considerations

**Infrastructure needs**:
- System to generate and score candidate comparisons
- Annotation platform that supports dynamic selection
- Real-time RM retraining for iterative selection

**Human factors**:
- Active learning may select harder comparisons
- Annotators might take longer or have lower agreement
- Need to balance difficulty with annotator experience

**Cost-benefit analysis**:
- Overhead of active learning system vs. annotation savings
- Break-even point: When is complexity worth it?

## Extensions

**Batch-mode active learning**:
- Select batches of comparisons (not one at a time)
- Optimize for diversity within batch
- More practical for parallel annotation

**Transfer active learning**:
- Use active learning insights from one domain to bootstrap another
- Meta-learn selection strategies

**Continual active learning**:
- In deployment, continuously select most valuable user interactions for labeling
- Online improvement of RM

**Multi-task active learning**:
- Collect preferences for multiple tasks simultaneously
- Optimize for collective learning across tasks

**Safety-critical active learning**:
- Prioritize comparisons relevant to safety
- Ensure dangerous edge cases are covered

**Hybrid human-AI annotation**:
- Use cheap AI labels for most comparisons
- Use expensive human labels for high-value comparisons selected by active learning
- Best of both worlds

By strategically selecting which preference comparisons to collect, active learning could dramatically reduce the cost of RLHF, making it accessible for resource-constrained applications and enabling more frequent model updates with limited annotation budgets.
