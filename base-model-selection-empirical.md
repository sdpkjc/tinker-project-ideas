# Empirical study: Does base model choice matter after RLHF?

A foundational question in post-training: How much does the choice of base (pre-trained) model affect final capabilities after RLHF or instruction tuning? Do different base models converge to similar capabilities after sufficient post-training, or do differences persist?

This has major practical implications. If base model choice is less important than post-training, practitioners could use cheaper or more accessible base models. If base model differences persist, investment in better pre-training remains crucial.

Anecdotally, practitioners observe that "better base models lead to better instruction-tuned models," but this hasn't been systematically quantified while controlling for post-training methodology.

## Core research questions

1. **Performance ceiling**: Do better base models lead to better post-trained models, or does post-training equalize them?

2. **Sample efficiency**: Do better base models require less post-training data to reach target performance?

3. **Domain transfer**: Do base models trained on certain domains (code, math, general text) maintain advantages after RLHF?

4. **Failure modes**: Do different base models have different failure modes that persist through post-training?

5. **Scaling interaction**: How does base model size interact with post-training data size?

## Experimental setup

### Base model selection

Choose diverse base models with controlled variation:

**Scale variation** (same architecture, different sizes):
- 1B, 3B, 7B, 13B, 34B parameters
- All from same model family (e.g., Llama 2)

**Training data variation** (similar size, different data):
- General web crawl (e.g., Llama base)
- Code-heavy (e.g., StarCoder base)
- Math-heavy (e.g., base model continued pre-training on math)
- Multilingual (e.g., BLOOM base)

**Architecture variation** (similar size, different architectures):
- Standard transformer
- With different attention mechanisms
- Different tokenizers

**Training quality variation**:
- Well-trained checkpoints (full training budget)
- Undertrained checkpoints (early stopping)
- Overtrained checkpoints (past optimal stopping)

### Post-training protocol

Apply **identical post-training** to all base models:
1. Same instruction tuning data (SFT)
2. Same preference data (RLHF)
3. Same hyperparameters, training steps, compute budget
4. Multiple random seeds for statistical significance

This isolates the effect of base model choice.

### Evaluation dimensions

**Capability benchmarks**:
- MMLU (general knowledge)
- HumanEval (coding)
- GSM8K, MATH (reasoning)
- TruthfulQA (factuality)
- BBH (challenging reasoning)

**Human evaluations**:
- Chatbot Arena style head-to-head comparisons
- Instruction following ratings
- Safety ratings

**Qualitative analysis**:
- Error categorization
- Behavioral patterns (verbosity, hedging, formatting)
- Failure modes

## Key experiments

### Experiment 1: Do better bases lead to better post-trained models?

**Hypothesis**: Yes, but with diminishing returns

**Test**:
- Post-train base models of increasing quality (measured by pre-training loss or zero-shot benchmarks)
- Measure final performance after identical post-training
- Plot: Base model quality → Post-trained model quality
- Expected: Positive correlation, but possibly sublinear

**Metric**: Correlation strength, slope of relationship

### Experiment 2: Sample efficiency of post-training

**Hypothesis**: Better base models need less post-training data

**Test**:
- For each base model, vary amount of post-training data (1K, 3K, 10K, 30K, 100K examples)
- Measure performance vs. data size curves
- Compare: How quickly does each base reach target performance?

**Metric**: Data required to reach 80% of maximum performance

### Experiment 3: Domain specialization persistence

**Hypothesis**: Domain advantages in base model persist after post-training

**Test**:
- Start with base models specialized in different domains
  - Model A: Code-pretrained
  - Model B: Math-pretrained
  - Model C: General-pretrained
- Apply identical general instruction tuning
- Test on both in-domain and out-of-domain tasks

**Expected**: Domain specialists retain advantage in their domain even after general post-training

**Metric**: Gap between specialist and generalist on domain-specific benchmarks

### Experiment 4: Scaling interactions

**Hypothesis**: Larger base models benefit more from post-training

**Test**:
- Matrix of (base model size) × (post-training data size)
- Measure performance for all combinations
- Look for interaction effects

**Question**: Is there a multiplicative effect where large base + large post-training data is better than expected from their individual effects?

### Experiment 5: Failure mode persistence

**Hypothesis**: Base model failure modes persist through post-training

**Test**:
- Identify specific failure modes in base models:
  - Hallucination patterns
  - Reasoning errors
  - Bias patterns
  - Refusal behaviors
- Post-train all models identically
- Test: Do the same failure modes persist in post-trained models?

**Analysis**: Failure mode correlation between base and post-trained

## Detailed measurements

For each base model, track:

**Pre-training characteristics**:
- Model size, architecture
- Training data composition
- Training compute, final loss
- Zero-shot and few-shot capabilities

**Post-training trajectory**:
- Performance at each post-training checkpoint
- Convergence speed
- Training stability
- Final capabilities

**Final comparisons**:
- Absolute performance gap between best and worst base
- Whether gaps narrow or widen during post-training
- Which gaps persist and which disappear

## Expected insights

This study could reveal:

1. **How much does base model matter?**
   - Quantify: "Using base model X vs Y leads to Z% performance difference after RLHF"

2. **When does base model matter most?**
   - Low-resource post-training: base model crucial
   - High-resource post-training: base model less important
   - Specific domains: base model matters more/less

3. **What aspects of base quality transfer?**
   - General knowledge → transfers
   - Reasoning ability → transfers
   - Specific facts → don't transfer
   - Code syntax → transfers

4. **Practical recommendations**:
   - If limited post-training data: invest in better base model
   - If abundant post-training data: acceptable base model sufficient
   - For domain-specific applications: use domain-pretrained base

## Challenges and controls

**Challenge**: Different base models may need different post-training hyperparameters

**Control**: Run hyperparameter sweep for each base, then compare at optimal settings for each

**Challenge**: Tokenizer differences complicate comparison

**Control**: Focus on models with same tokenizer, or re-tokenize data consistently

**Challenge**: Random seed variance

**Control**: Multiple runs per condition, statistical significance testing

## Connection to scaling laws

This relates to [Chinchilla scaling laws](https://arxiv.org/abs/2203.15556) but for post-training:
- How do compute-optimal base models compare after post-training?
- Should we train larger base models for less time if planning to post-train?
- New scaling laws: performance = f(pretrain_compute, posttrain_data, model_size)

## Extensions

**Continual pre-training**:
- How does domain-adaptive pre-training compare to starting from general base?

**Multimodal extension**:
- Do vision-language base models matter after instruction tuning?

**Multilingual extension**:
- Do multilingual bases maintain language coverage after English-dominant post-training?

This empirical study would provide crucial guidance for the expensive decision of which base model to use for post-training, grounded in systematic experimental data rather than anecdotes.
