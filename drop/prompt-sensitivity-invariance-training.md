# Training for prompt invariance: Consistent behavior across rephrased queries

Language models often exhibit high sensitivity to minor prompt variations—rephrasing a question or changing word order can yield significantly different outputs, even when the semantic meaning is identical. This brittleness is problematic for deployment: users shouldn't need to craft perfect prompts to get reliable responses.

This project explores methods to train models that are invariant to semantically-preserving prompt variations, providing consistent, reliable behavior regardless of phrasing, inspired by work on adversarial robustness ([Madry et al., 2017](https://arxiv.org/abs/1706.06083)) and consistency regularization ([Xie et al., 2020](https://arxiv.org/abs/1911.04252)).

## Problem: Prompt sensitivity

**Example variations** (semantically equivalent):
1. "What is the capital of France?"
2. "Tell me the capital city of France."
3. "France's capital is located where?"
4. "In France, which city serves as the capital?"
5. "Capital of France?"

**Desired**: Consistent, high-quality answer to all variations

**Reality**: Models may give:
- Different answer confidence levels
- Different levels of detail
- Different formatting
- Sometimes even different answers

**Why this matters**:
- Users have diverse linguistic styles
- Accessibility: Some users struggle with "optimal" phrasing
- Reliability: Critical applications need consistent behavior
- Reduces need for prompt engineering

## Proposed training approaches

### Approach 1: Consistency regularization

Train model to produce similar outputs for semantically equivalent prompts:

**Method**:
1. For each training prompt, generate K semantic variations
   - Use paraphrase models
   - Use LLM-based rewriting
   - Use rule-based transformations (active→passive, etc.)

2. Add consistency loss:
   ```
   L_consistency = Σ distance(output(prompt_i), output(prompt_j)) for i≠j
   ```

   Where distance can be:
   - KL divergence between output distributions
   - Embedding similarity (encode outputs, measure distance)
   - Token-level differences

3. Joint training objective:
   ```
   L_total = L_task + λ · L_consistency
   ```

**Benefits**:
- Directly optimizes for invariance
- Works with any base training objective (SFT, RLHF)
- Controllable trade-off via λ

### Approach 2: Data augmentation with equivalence classes

Create training data with explicit equivalence classes:

**Data format**:
```
Equivalence class 1: {prompt_1, prompt_2, ..., prompt_K} → response
Equivalence class 2: {prompt_1', prompt_2', ..., prompt_K'} → response'
```

**Training**:
- Sample random prompt from equivalence class during training
- Model learns: All prompts in class should map to same response
- Implicitly learns invariance through exposure

**Benefits**:
- Simple to implement (just data augmentation)
- No architecture changes needed
- Can be combined with standard training

**Data generation**:
- Human annotation: Identify equivalent prompts
- Automated paraphrasing: Generate variations
- Mix both: Generate then human-validate

### Approach 3: Adversarial prompt perturbations

Use adversarial training to find worst-case prompt variations:

**Method**:
1. Generate initial response to prompt
2. Search for prompt variation that maximizes output change
   - Use gradient-based perturbations on prompt embeddings
   - Or use LLM to generate "adversarial paraphrases"
3. Train model to give consistent output on adversarial variation
4. Repeat iteratively

**Benefits**:
- Finds hard cases automatically
- More efficient than random augmentation
- Strongest robustness guarantees

**Inspired by**: Adversarial training in computer vision

### Approach 4: Contrastive learning on prompt embeddings

Learn prompt representations where semantic equivalence → similar embeddings:

**Method**:
1. Encode prompts into latent space
2. Contrastive objective:
   - Equivalent prompts → close embeddings
   - Different prompts → distant embeddings
3. Use these embeddings to condition model output
4. Model learns to "normalize" semantically equivalent prompts

**Architecture**:
```
prompt → encoder → normalized_embedding → transformer → output
```

**Benefits**:
- Learns robust prompt representation
- Can visualize equivalence classes
- Interpretable: See which prompts model considers equivalent

### Approach 5: Multi-prompt ensembling at inference

Generate variations at inference time and aggregate:

**Method**:
1. User provides prompt
2. Generate K semantic variations automatically
3. Run model on all variations
4. Aggregate outputs:
   - For classification: Majority vote
   - For generation: Ensemble decoding or consistency voting
   - For QA: Return most consistent answer

**Benefits**:
- No training changes needed
- Can be applied to existing models
- Improved reliability immediately

**Drawbacks**:
- Higher inference cost (K× more forward passes)
- Need method to generate variations quickly

## Evaluation methodology

### Metric 1: Prompt sensitivity score

**Setup**:
1. Collect or generate test prompts with semantic variations
2. Measure output consistency across variations

**Metrics**:
- **Output stability**: How often does model give same answer?
- **Confidence variance**: How much does model confidence change?
- **Semantic similarity**: Embedding distance between outputs
- **Task performance variance**: Accuracy difference across variations

**Lower variance = Better prompt invariance**

### Metric 2: Adversarial prompt robustness

**Setup**:
1. Generate adversarial paraphrases designed to change output
2. Measure how often model maintains consistent behavior

**Metrics**:
- **Adversarial success rate**: Fraction of adversarial prompts that change output
- **Certified robustness**: Worst-case consistency guarantee

### Metric 3: User study

**Setup**:
1. Ask users to submit queries naturally (not prompt engineering)
2. Measure satisfaction and task success
3. Compare models trained for invariance vs. baseline

**Question**: Do invariant models provide better user experience?

### Metric 4: Prompt paraphrase benchmark

Create benchmark of equivalent prompts across diverse domains:
- Factual QA (same question, different phrasing)
- Instruction following (same task, different wording)
- Math problems (same problem, different notation)
- Code generation (same spec, different description)

**Evaluate**: Consistency on this benchmark

## Experimental framework

### Experiment 1: Invariance training effectiveness

**Compare**:
- Baseline: Standard training
- +Consistency regularization
- +Data augmentation with equivalences
- +Adversarial robustness training
- +Contrastive prompt encoding

**Measure**:
- Prompt sensitivity score (lower is better)
- Task performance (shouldn't degrade)
- Training efficiency (additional cost)

### Experiment 2: Domain specificity

**Question**: Does invariance differ across domains?

**Test**:
- Train invariance on domain A (e.g., factual QA)
- Evaluate on domain B (e.g., creative writing)

**Hypothesis**: Some invariance generalizes, some is domain-specific

### Experiment 3: Invariance vs. capability trade-off

**Question**: Does training for invariance hurt performance?

**Test**:
- Train models with varying λ (invariance weight)
- Plot: Task performance vs. Invariance level

**Expected**: Some trade-off, but hopefully small

### Experiment 4: Types of variations

**Question**: Which types of variations are hardest to handle?

**Test**:
- Word order changes (passive ↔ active voice)
- Formality level (casual ↔ formal)
- Verbosity (brief ↔ detailed)
- Syntax (question ↔ statement)
- Language dialect or style

**Measure**: Which variations cause most inconsistency?

## Practical implementation

**Phase 1: Data collection**
- Collect or generate semantically equivalent prompt sets
- For each prompt, create 5-10 paraphrases
- Human validation of semantic equivalence

**Phase 2: Training**
- Start with instruction-tuned model
- Add consistency regularization or data augmentation
- Train with joint objective

**Phase 3: Evaluation**
- Test on prompt paraphrase benchmark
- User study with natural queries
- Compare to baseline on reliability metrics

## Key research questions

1. **Can we train models robust to prompt variations?**
   - How much invariance is achievable?
   - What are fundamental limits (some variations must change output)?

2. **What training method works best?**
   - Data augmentation vs. consistency regularization vs. adversarial training
   - Trade-offs in cost and effectiveness?

3. **Does invariance hurt capability?**
   - Is there a performance cost?
   - Can we have both robustness and performance?

4. **Does invariance generalize?**
   - Train on one domain, test on another
   - Learn general invariance principles vs. domain-specific?

5. **What user impact does this have?**
   - Quantifiable improvement in user satisfaction?
   - Reduces need for prompt engineering?

## Connection to related work

- **Adversarial robustness**: Similar techniques to adversarial training in vision
- **Data augmentation**: Standard technique, applied to prompts
- **Consistency regularization**: Used in semi-supervised learning
- **Certified robustness**: Formal guarantees on model behavior

This project adapts these techniques specifically for prompt invariance in LLMs.

## Extensions

**Selective invariance**:
- Some variations should change output (genuinely different questions)
- Learn to distinguish semantic equivalence from meaningful differences
- Meta-learning: Learn when to be invariant vs. sensitive

**Personalized invariance**:
- Different users have different linguistic patterns
- Adapt invariance to user's typical variations
- Improve accessibility for specific populations

**Multilingual invariance**:
- Same semantic content across languages
- Train for consistency across translations
- Cross-lingual reliability

**Compositional invariance**:
- Handle variations in complex multi-part prompts
- Robustness to instruction ordering, formatting, etc.

By training models to be invariant to semantically-preserving prompt variations, we can make LLMs more reliable, accessible, and user-friendly—reducing the "prompt engineering" burden and providing consistent behavior across diverse users and use cases.
