# Efficient reward model distillation for faster RLHF inference

Reward models in RLHF are often large language models (7B-70B+ parameters) that score each generated response. This creates computational bottlenecks:
- Best-of-N sampling requires N×RM_size forward passes
- RL training needs frequent reward evaluations
- Real-time scoring for deployment is expensive

This project explores distilling large, accurate reward models into smaller, faster models while preserving ranking quality, enabling efficient RLHF inference and training.

## Motivation

**Current situation**:
- High-quality reward models are large (often same size as policy or larger)
- Reward evaluation is a bottleneck in:
  - Best-of-N sampling (need to score N candidates)
  - RL training (frequent reward queries)
  - Online deployment (real-time feedback)

**Desired**:
- Small, fast reward model (100M-1B parameters)
- Preserves ranking quality of large RM
- Enables efficient inference and training

**Inspiration**: Knowledge distillation ([Hinton et al., 2015](https://arxiv.org/abs/1503.02531)) has successfully compressed large models in other domains.

## Challenges specific to reward model distillation

**Challenge 1: Regression is hard**
- Distilling exact reward scores may fail (large model gives precise scores)
- Small model may not have capacity to match exact scores

**Solution**: Focus on ranking preservation, not score matching
- Only need relative ordering: A > B > C
- More forgiving than exact score regression

**Challenge 2: Distribution coverage**
- Reward model sees diverse responses during training and deployment
- Small model must generalize across this diversity

**Solution**: Careful curation of distillation dataset
- Sample from diverse policies and prompts
- Include edge cases and challenging comparisons

**Challenge 3: Calibration**
- Large RM may be well-calibrated (scores correlate with true quality)
- Small RM might preserve ranking but not calibration

**Solution**: Multi-objective distillation
- Ranking loss + calibration loss
- Ensure small RM's scores are meaningful, not just ordered correctly

## Proposed distillation approaches

### Approach 1: Ranking-based distillation

Train small RM to preserve pairwise rankings from large RM:

**Method**:
1. Generate diverse (prompt, response) pairs
2. Score all pairs with large RM: R_large(x, y)
3. For pairs (x, y_1), (x, y_2):
   - If R_large(x, y_1) > R_large(x, y_2)
   - Train small RM: R_small(x, y_1) > R_small(x, y_2)

**Loss**:
```
L_rank = Σ max(0, margin - (R_small(x, y_1) - R_small(x, y_2)))
```

**Benefits**:
- Focuses on what matters (ranking, not exact scores)
- More robust to capacity limitations
- Directly optimizes for RLHF use case

### Approach 2: Distribution matching

Train small RM to match large RM's score distribution:

**Method**:
1. For each (prompt, response), get R_large(x, y)
2. Train small RM to minimize:
   ```
   L_distill = MSE(R_small(x, y), R_large(x, y))
   ```
3. Additionally match higher-order statistics:
   - Mean and variance of scores per prompt
   - Correlation between scores

**Benefits**:
- Preserves calibration (scores have similar scale)
- Useful if absolute scores matter (not just ranking)

### Approach 3: Contrastive distillation

Use contrastive learning to separate good and bad responses:

**Method**:
1. For each prompt, sample N responses
2. Large RM ranks them: y_1 > y_2 > ... > y_N
3. Train small RM with contrastive loss:
   ```
   L_contrast = -log( exp(R_small(x, y_1)) / Σ_i exp(R_small(x, y_i)) )
   ```

**Benefits**:
- Explicitly optimizes for top-1 selection (best-of-N)
- Naturally handles multiple candidates
- Similar to how we use RM in practice

### Approach 4: Policy-aware distillation

Distill RM on outputs from the policies it will actually evaluate:

**Method**:
1. Generate responses from target policy (or ensemble of policies)
2. Score with large RM
3. Distill small RM on this policy-specific distribution

**Benefits**:
- Small RM specializes to relevant distribution
- More efficient than covering all possible responses
- Can be combined with continual learning as policy evolves

**Risk**: Small RM may not generalize to new policy behaviors

**Mitigation**: Mix policy-specific and diverse synthetic data

### Approach 5: Hierarchical distillation

Distill progressively from large to small:

**Method**:
```
Large RM (70B) → Medium RM (7B) → Small RM (1B) → Tiny RM (100M)
```

Each stage:
- Use previous stage as teacher
- Distill to next smaller size
- Validate that ranking quality is preserved

**Benefits**:
- Gradual capacity reduction may preserve more quality
- Each intermediate model is useful (trade-off point on efficiency/quality curve)

## Training data for distillation

**Data generation strategies**:

1. **Policy sampling**:
   - Sample responses from various policies (base, instruct-tuned, RLHF-trained)
   - Ensures coverage of realistic outputs

2. **Diverse prompts**:
   - Sample from multiple datasets and domains
   - Include different difficulty levels and types

3. **Contrastive pairs**:
   - Generate responses with varied quality (temperature, top-k sampling)
   - Ensure clear positive and negative examples

4. **Edge cases**:
   - Adversarial examples, reward hacking attempts
   - Hard-to-judge ambiguous cases
   - Ensure small RM handles challenging scenarios

**Data efficiency**:
- How much data is needed for effective distillation?
- Can we use active learning to select most informative examples?

## Evaluation methodology

### Metric 1: Ranking correlation

Measure how well small RM preserves large RM's rankings:

**Metrics**:
- Kendall's τ (rank correlation)
- Spearman's ρ
- Top-k agreement (do they agree on best-k responses?)

**Test on**: Diverse held-out (prompt, responses) sets

### Metric 2: Policy quality after RL

**Setup**:
1. Train policy with RL using large RM (gold standard)
2. Train policy with RL using small distilled RM
3. Compare final policy quality (human eval, benchmarks)

**Question**: Does using distilled RM degrade final policy quality?

### Metric 3: Best-of-N performance

**Setup**:
1. Generate N responses per prompt
2. Select best using large RM vs. small RM
3. Compare quality of selected responses (human eval)

**Question**: Does small RM select comparably good responses?

### Metric 4: Inference efficiency

**Measure**:
- Latency per reward evaluation
- Throughput (evaluations per second)
- Memory footprint
- Cost (compute × time)

**Compare**: Large RM vs. small distilled RM

## Experimental framework

### Experiment 1: Distillation method comparison

**Compare**:
- Ranking-based
- Distribution matching
- Contrastive
- Policy-aware
- Hierarchical

**Measure**: Ranking correlation, policy quality, efficiency

**Question**: Which method best preserves quality while minimizing size?

### Experiment 2: Size-quality trade-off

**Setup**:
Distill large RM to various sizes: 100M, 300M, 1B, 3B, 7B

**Plot**: Model size vs. ranking correlation

**Find**: Optimal point on Pareto frontier (smallest model with acceptable quality)

### Experiment 3: Data requirements

**Question**: How much data is needed for effective distillation?

**Test**:
- Vary distillation data size: 1K, 10K, 100K, 1M examples
- Measure distilled RM quality

**Find**: Minimum data for good distillation

### Experiment 4: Generalization to new policies

**Setup**:
1. Distill RM on outputs from policy A
2. Test on outputs from policy B (different policy)

**Question**: Does distilled RM generalize to new policies?

## Practical implementation

**Implementation in Tinker Cookbook**:
- Add reward model distillation pipeline
- Integrate distilled RMs into RLHF training
- Benchmark distillation methods

**Production deployment**:
- Use large RM for training
- Deploy small distilled RM for inference
- Periodically re-distill as large RM improves

## Key research questions

1. **Can small RMs preserve large RM ranking quality?**
   - How much size reduction is possible?
   - What's the size-quality trade-off curve?

2. **Which distillation method works best?**
   - Ranking, distribution matching, or contrastive?
   - Do they complement each other?

3. **Does distilled RM hurt final policy quality?**
   - Can we use small RM for RL training without degradation?
   - Or only for inference (best-of-N)?

4. **How does distillation affect different RM failure modes?**
   - Does distilled RM inherit large RM's weaknesses?
   - Or introduce new failure modes?

## Extensions

**Ensemble of small RMs**:
- Multiple small RMs distilled differently
- Aggregate their scores for better reliability
- Trade-off: Multiple small models vs. one large

**Adaptive RM selection**:
- Use small RM for easy examples
- Fall back to large RM for hard/uncertain cases
- Best of both worlds: efficiency + accuracy

**Continual distillation**:
- As large RM improves, re-distill small RM
- Keep small RM up-to-date with minimal cost

**Multi-task RM distillation**:
- Single small RM distilled from multiple specialized large RMs
- Handles diverse tasks efficiently

This project could make RLHF more practical and scalable by removing computational bottlenecks, enabling real-time reward evaluation and efficient best-of-N sampling with minimal quality loss.
