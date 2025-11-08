# Inference-time reward model ensembles for best-of-N sampling

Best-of-N sampling is a simple but effective technique for improving language model outputs: generate N candidate responses and select the best one according to a reward model. However, reward models have limitations—they may be overfit to certain response patterns, fail to capture all aspects of quality, or be vulnerable to reward hacking.

Ensemble methods have proven effective across machine learning for improving robustness and reducing overfitting. Recent work on uncertainty estimation in reward models ([Gleave et al., 2022](https://arxiv.org/abs/2210.10760)) shows that reward model uncertainty is often high in critical regions. This project explores using ensembles of diverse reward models at inference time to improve best-of-N sampling.

## Core idea

Instead of training a single reward model and using it for best-of-N selection, train multiple diverse reward models and combine their scores:

**Ensemble composition strategies:**

1. **Multi-objective ensemble**: Train separate reward models for different aspects
   - Helpfulness reward model
   - Harmlessness/safety reward model
   - Honesty/accuracy reward model
   - Combine with weighted sum: R_total = α·R_helpful + β·R_safe + γ·R_honest

2. **Architecture diversity**: Same training data, different model architectures
   - Different backbone models (different sizes, families)
   - Different reward head architectures
   - Reduces overfitting to specific model biases

3. **Data diversity**: Same architecture, different training data
   - Train on different splits or subsets of preference data
   - Bootstrap sampling of preference dataset
   - Different synthetic data generation strategies
   - Captures different aspects of human preferences

4. **Uncertainty-aware ensembles**: Weight models by their confidence
   - Higher weight for models with low uncertainty
   - Helps avoid relying on unreliable scores

## Aggregation methods

How to combine multiple reward scores:

- **Simple averaging**: R_ensemble = (1/K) Σ R_k
- **Weighted voting**: R_ensemble = Σ w_k · R_k (learn weights on validation set)
- **Rank aggregation**: Each model ranks candidates, aggregate rankings (reduces score miscalibration)
- **Uncertainty weighting**: Weight by inverse uncertainty: w_k ∝ 1/σ_k
- **Conditional**: Use different ensembles for different prompt types

## Implementation approach

1. Train K diverse reward models using one of the ensemble strategies above
2. For best-of-N sampling:
   - Generate N candidates from policy
   - Score each candidate with all K reward models
   - Aggregate scores and select best candidate
3. Compare ensemble best-of-N vs. single-model best-of-N on:
   - Automated evaluations (diverse benchmarks)
   - Human preference studies
   - Adversarial robustness tests

## Computational cost considerations

Ensemble scoring increases inference cost by K×. Mitigation strategies:

- **Cascade approach**: Use cheap models to filter candidates, expensive models for final selection
- **Learned routing**: Train a meta-model to predict which ensemble member is most reliable for each prompt
- **Distillation**: Distill the ensemble into a single model that approximates ensemble behavior
- **Selective ensembling**: Only use ensemble for high-stakes or uncertain queries

## Key research questions

- How much does ensemble best-of-N improve over single-model best-of-N?
- What type of diversity (objective, architecture, data) is most valuable?
- How many ensemble members are needed for diminishing returns?
- Does ensembling reduce reward hacking compared to a single reward model?
- Can we predict when ensemble disagreement indicates problematic outputs?
- Does ensemble uncertainty correlate with human uncertainty on preferences?

## Extensions

- **Online learning**: Update ensemble weights based on user feedback during deployment
- **Ensemble for RL training**: Use ensemble reward models during policy training, not just inference
- **Consensus requirements**: Only select candidates where ensemble agrees (high confidence)
- **Disagreement analysis**: Study cases where ensemble members disagree to understand reward model failures

## Practical benefits

This approach is appealing because:
- Improves output quality without retraining the policy
- Can be deployed immediately to existing systems that use best-of-N
- Computational cost is bounded (only at inference, not training)
- Easy to update by adding new reward models to ensemble
- Provides uncertainty estimates for safety-critical applications

If effective, inference-time ensembles could be a practical way to make best-of-N sampling more robust and reliable without the complexity of policy retraining.
