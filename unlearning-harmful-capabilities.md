# Selective unlearning of harmful capabilities while preserving benign performance

Language models trained on internet data inevitably learn harmful capabilities—generating malware, synthesizing dangerous information, or producing toxic content. Simple filtering during post-training is insufficient: the knowledge remains latent and can be elicited through jailbreaks or adversarial prompts.

This project explores machine unlearning for LLMs: selectively removing specific harmful capabilities while preserving general performance, inspired by [right-to-be-forgotten](https://arxiv.org/abs/1911.04933) work and recent advances in model editing.

## Motivation

**Problem**: LLMs learn unwanted capabilities during pre-training
- Code generation models learn to write exploits
- General models learn to produce hate speech or misinformation
- Models memorize private or sensitive data

**Current approaches and limitations**:
- **RLHF/safety training**: Teaches refusal but doesn't remove capability
  - Vulnerable to jailbreaks
  - Capability remains latent
- **Data filtering**: Prevents learning but can't fix already-trained models
- **Output filtering**: Reactive, not proactive

**Desired**: Remove harmful capabilities at the model level
- Model genuinely cannot perform harmful tasks (not just refusing)
- Preserve performance on benign tasks
- Robust to adversarial elicitation

## Challenges

**Challenge 1**: Capability entanglement
- Harmful and benign capabilities may share representations
- Example: Cybersecurity knowledge used for both defense and attack
- Removing harm may damage legitimate use cases

**Challenge 2**: Defining "harmful"
- Context-dependent (e.g., medical information helpful in one context, harmful in another)
- Evolving definitions (what's considered harmful changes)
- Trade-offs (safety vs. utility)

**Challenge 3**: Verification
- Hard to prove a capability is truly removed
- May appear removed on test sets but remain latent
- Adversarial elicitation can reveal hidden capabilities

**Challenge 4**: Efficiency
- Can't retrain from scratch for each unlearning request
- Need efficient post-hoc methods

## Proposed approaches

### Approach 1: Gradient-based unlearning

Reverse the training process for harmful examples:

**Method**:
1. Identify harmful capability to remove (e.g., bioweapon synthesis)
2. Collect dataset of harmful examples demonstrating this capability
3. Apply "negative gradient" updates:
   ```
   θ_new = θ_old + α · (-∇L(harmful_data))
   ```
4. Simultaneously maintain performance on benign data:
   ```
   Total loss = -L(harmful) + L(benign)
   ```

**Benefits**:
- Conceptually simple: undo harmful learning
- Can target specific capabilities
- Efficient (fine-tuning, not retraining)

**Challenges**:
- May not completely erase capability
- Risk of catastrophic forgetting on benign tasks
- Need to balance harmful removal vs. benign preservation

### Approach 2: Activation steering and ablation

Identify and remove internal representations of harmful capabilities:

**Method**:
1. Probe model internals to find neurons/directions encoding harmful knowledge
   - Techniques: Causal mediation analysis, activation patching
2. Ablate or zero-out these components
3. Test: Does harmful capability disappear? Does benign performance remain?

**Inspired by**:
- [Mechanistic interpretability](https://arxiv.org/abs/2211.00593)
- Activation steering ([Turner et al., 2023](https://arxiv.org/abs/2308.10248))

**Benefits**:
- Targeted surgical intervention
- Potentially more complete removal
- Interpretable (can visualize what's removed)

**Challenges**:
- Requires deep model understanding
- May be capability-entangled (shared representations)

### Approach 3: Contrastive unlearning

Train model to distinguish and avoid harmful patterns:

**Method**:
1. Create contrastive pairs:
   - Harmful example: "How to make a bomb"
   - Benign alternative: "History of explosives in mining"
2. Train model to:
   - Maximize distance between harmful and benign representations
   - Reduce likelihood of harmful completions
   - Maintain likelihood of benign completions

**Loss**:
```
L = -log P(benign) + log P(harmful) + contrastive_loss(repr_harmful, repr_benign)
```

**Benefits**:
- Explicitly models distinction between harmful and benign
- Less likely to damage related benign capabilities
- Can handle nuanced distinctions

### Approach 4: Scrubbing through knowledge distillation

Distill model into new model while filtering harmful knowledge:

**Method**:
1. Original model (teacher): Knows harmful + benign
2. Generate training data from teacher on benign prompts only
3. Train student model on filtered data
4. Student learns benign capabilities but not harmful ones

**Benefits**:
- Clean slate approach
- Naturally avoids harmful knowledge
- Can combine with other safety measures

**Challenges**:
- Computationally expensive (full distillation)
- May still learn harmful patterns from benign data
- Reduces overall capability (distillation gap)

### Approach 5: Task-specific unlearning via fine-tuning

Fine-tune model to "forget" specific tasks:

**Method**:
1. Identify harmful task (e.g., writing malware)
2. Fine-tune on:
   - Random/nonsense outputs for harmful prompts
   - Or refusal responses
   - With high loss penalty for correct harmful outputs
3. Use LoRA or adapter to avoid full model modification

**Benefits**:
- Simple and efficient
- Can be reversed if needed (remove adapter)
- Preserves most of original model

**Challenges**:
- May not fully remove capability (still accessible via jailbreaks)
- Essentially teaches refusal, not true unlearning

## Verification and evaluation

**Challenge**: How do we verify harmful capability is truly removed?

### Test 1: Direct elicitation

**Method**: Try to elicit harmful capability with direct prompts
- Example: "Write code to exploit SQL injection"
- Success = Model refuses or produces nonsense
- Failure = Model produces functional harmful output

**Limitation**: Model may refuse but still have capability

### Test 2: Adversarial elicitation

**Method**: Use jailbreaking techniques to bypass safety
- Role-playing scenarios
- Encoded prompts
- Multi-step reasoning
- Hypothetical scenarios

**Success**: Model resists all adversarial attempts

### Test 3: Capability probing

**Method**: Use interpretability tools to probe for harmful knowledge
- Does model's internal state contain harmful information?
- Can we recover harmful outputs via activation steering?

**Success**: No traces of harmful capability in model internals

### Test 4: Transfer tests

**Method**: Can model's harmful knowledge transfer to other contexts?
- Example: If we removed malware generation, can it still explain vulnerabilities?
- Success: Related but benign knowledge preserved, harmful application removed

### Test 5: Benign performance preservation

**Critical**: Unlearning shouldn't hurt general performance

**Metrics**:
- Benchmarks (MMLU, HumanEval, etc.) before and after
- User studies on benign tasks
- Success: <X% degradation on benign tasks

## Experimental framework

### Experiment 1: Capability removal effectiveness

**Setup**:
1. Select harmful capabilities to remove:
   - Malware generation
   - Hate speech production
   - Dangerous knowledge (bioweapons, explosives)
   - Personal information memorization

2. Apply unlearning methods
3. Test with direct, adversarial, and probing methods
4. Measure: Complete removal vs. partial removal vs. failure

### Experiment 2: Benign performance preservation

**Measure**:
- Before unlearning: Baseline performance
- After unlearning: Performance on same benchmarks
- Calculate: Performance degradation

**Acceptable**: <5% degradation on benign tasks

### Experiment 3: Capability entanglement

**Question**: How much do harmful and benign capabilities overlap?

**Method**:
1. Attempt to remove harmful capability X
2. Measure impact on related benign capability Y
3. Example: Remove exploit writing → Test on cybersecurity defense

**Find**: Which capabilities can be cleanly separated?

### Experiment 4: Unlearning efficiency

**Measure**:
- Compute cost of each unlearning method
- Time to complete unlearning
- Compare to retraining from scratch

**Goal**: Efficient post-hoc unlearning (<<1% of original training cost)

## Key research questions

1. **Is true unlearning possible?**
   - Can we completely remove a capability, or only suppress it?
   - What's the difference between unlearning and refusal?

2. **What's the capability entanglement level?**
   - How much do harmful and benign capabilities share?
   - Can we surgically remove one without damaging the other?

3. **How do we verify unlearning?**
   - What tests are sufficient to prove removal?
   - Can we ever be confident harmful capability is gone?

4. **What's the performance-safety trade-off?**
   - How much benign performance do we sacrifice?
   - Is it worth it?

5. **Can unlearning be subverted?**
   - Are unlearned capabilities recoverable through fine-tuning?
   - Can adversaries reverse engineer unlearning?

## Practical considerations

**Use cases**:
- Model providers removing dangerous capabilities from public models
- Compliance with regulations (right to be forgotten, content removal)
- Safety updates to deployed models
- Customization (different safety levels for different users)

**Challenges**:
- Defining what to unlearn (subjective, context-dependent)
- Verification costs (extensive testing needed)
- Arms race (users trying to recover unlearned capabilities)

## Extensions

**Continual unlearning**:
- Support ongoing removal of newly-identified harmful capabilities
- Without accumulating performance degradation

**Personalized unlearning**:
- Different users want different things unlearned
- Multi-tenant models with user-specific unlearning

**Certified unlearning**:
- Formal verification that capability is removed
- Provable guarantees (maybe impossible, but worth exploring)

**Federated unlearning**:
- Remove capabilities without centralizing data
- Privacy-preserving unlearning

This research is critical for safely deploying powerful LLMs: we need methods to remove dangerous capabilities post-hoc, without expensive retraining or sacrificing general utility.
