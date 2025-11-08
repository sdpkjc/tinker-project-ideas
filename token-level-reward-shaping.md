# Token-level reward shaping for fine-grained RL control

Standard RLHF provides a single reward score for an entire response, treating it as an atomic unit. However, responses contain many tokens, and quality varies within a response—some parts may be excellent while others are problematic. This coarse reward signal may make learning inefficient and fail to provide fine-grained guidance.

This project explores token-level reward shaping: assigning rewards to individual tokens or small spans within a response, providing granular feedback about which parts of the generation are good or bad.

## Motivation

**Current RLHF**:
- Prompt → Response (100+ tokens) → Single scalar reward
- Problem: Can't distinguish which tokens contributed to quality

**Desired**:
- Prompt → Response → Per-token or per-span rewards
- Benefit: Credit assignment to specific generation choices

This is inspired by [process reward models](https://arxiv.org/abs/2305.20050) which reward reasoning steps, but generalizes beyond reasoning to all text generation.

## Approaches to token-level rewards

### Approach 1: Counterfactual token importance

For each token in a response:
1. Generate full response: "The capital of France is Paris, which is..."
2. Measure baseline reward R₀
3. For each token position i, regenerate response with that token masked or replaced
4. Measure new reward R_i
5. Token importance = R₀ - R_i (how much does quality drop without this token?)

This gives token-level attributions: which tokens are most critical for quality.

### Approach 2: Gradient-based attribution

Use gradient-based methods from interpretability:
- Compute ∇_token R (gradient of reward w.r.t. token embeddings)
- Larger gradients = token has more influence on reward
- Similar to attention-based attribution but more direct

### Approach 3: Learned token-level reward model

Train a reward model that outputs per-token scores:
- Input: (prompt, partial_response, next_token)
- Output: Quality score for generating that token in that context

Training signal from human preferences:
- Given preference A > B, not only train R(A) > R(B)
- Also align token-level rewards with contrastive editing:
  - If responses differ mainly in span X, that span's tokens should explain the preference

### Approach 4: Self-evaluation at each step

Train the policy to evaluate its own tokens as it generates:
- After generating token t_i, model outputs confidence score
- Train this self-evaluation to match outcome-based quality
- Similar to [step-wise uncertainty estimation](https://arxiv.org/abs/2305.14975)

## Using token-level rewards for RL training

Once we have per-token rewards, several training approaches:

### Option 1: Advantage reshaping

Standard PPO uses single reward for entire trajectory. With token-level rewards:
```
A_t = R_t + γR_{t+1} + γ²R_{t+2} + ... (token-level advantage)
```

This provides denser reward signal and better credit assignment.

### Option 2: Token-level policy gradients

Directly optimize:
```
max E[Σ_t r_t · log π(token_t | context)]
```

Where r_t is the reward for token t. This is more granular than episode-level gradients.

### Option 3: Selective token optimization

Focus RL updates on high-impact tokens:
- Identify tokens with large reward impact
- Concentrate gradient updates on those positions
- More sample-efficient than uniform updates

### Option 4: Real-time steering

Use token-level rewards for inference-time steering:
- At each generation step, evaluate candidate tokens
- Choose tokens with highest predicted future reward
- Similar to beam search but guided by learned reward

## Key research questions

1. **Can we reliably estimate token-level rewards?**
   - Do different methods (counterfactual, gradient, learned) agree?
   - How stable are token-level attributions?

2. **Does token-level RL improve sample efficiency?**
   - Compare against standard episode-level RLHF
   - How much faster does learning occur with finer-grained feedback?

3. **Does this improve credit assignment?**
   - Can models learn to fix specific problematic tokens?
   - Does this reduce reward hacking (by making it harder to exploit coarse rewards)?

4. **What granularity is optimal?**
   - Per-token too fine-grained (noisy)?
   - Per-sentence too coarse?
   - Per-phrase or per-clause optimal?

5. **Does this improve final policy quality?**
   - Better task performance?
   - Fewer failure modes?
   - More targeted improvements?

## Evaluation methodology

**Controlled experiments**:
1. Train policies with:
   - Baseline: Standard episode-level RLHF
   - Treatment: Token-level reward shaping
2. Control for total compute and data
3. Measure: Sample efficiency, final performance, training stability

**Analysis**:
- Do token-level models make more targeted edits during training?
- Can we visualize which tokens receive strongest learning signal?
- Does this prevent certain types of reward hacking?

**Qualitative**:
- Compare responses from both approaches
- Do token-level models produce more consistently high-quality text?
- Fewer "good overall but with bad parts" responses?

## Practical implementation

**Start simple**:
1. Use counterfactual method (computationally expensive but interpretable)
2. Apply to small models and datasets
3. Measure: Does credit assignment improve?

**Scale up**:
1. Train efficient token-level reward model
2. Integrate into PPO training loop
3. Test on instruction-following and reasoning tasks

**Implementation in Tinker Cookbook**:
- Extend existing RLHF pipeline with token-level rewards
- Compare standard vs. token-level RL side-by-side

## Challenges

**Computational cost**:
- Counterfactual evaluation requires many forward passes
- Mitigation: Use learned token-level reward model

**Reward sparsity**:
- Many tokens are neutral (e.g., "the", "is")
- Mitigation: Focus on content tokens, aggregate function words

**Credit assignment ambiguity**:
- Token quality depends on context
- "Paris" is good if the question is "capital of France", bad otherwise
- Mitigation: Contextual token-level rewards

**Training instability**:
- More complex reward signal might destabilize training
- Mitigation: Gradual introduction, careful hyperparameter tuning

## Extensions

**Hierarchical rewards**:
- Token-level (fine-grained)
- Sentence-level (medium)
- Response-level (coarse)
- Combine all three for multi-scale credit assignment

**Interactive refinement**:
- Show users token-level reward predictions
- Let them correct: "This token is actually good/bad"
- Fine-tune reward model on corrections

**Speculative decoding integration**:
- Use token-level rewards to guide speculative decoding
- Generate multiple continuations, select based on token-level quality

**Error localization**:
- Automatically identify problematic tokens in responses
- Useful for debugging and targeted improvement

This approach could make RLHF more efficient and effective by providing the fine-grained feedback that modern deep RL systems benefit from, while maintaining the end-to-end differentiability that makes language model training tractable.
