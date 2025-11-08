# Empirical investigation of synthetic data scaling: when does quality trump quantity?

Synthetic data generation is increasingly used for training LLMs, from [Phi-1.5](https://arxiv.org/abs/2309.05463) to various distillation and self-improvement approaches. A fundamental question remains empirically underexplored: given fixed generation compute budget, how should we trade off data quality versus quantity?

While scaling-laws-synthetic-data.md proposes a framework for studying this, this empirical study focuses on systematically measuring the trade-off curves across different domains and model scales.

## Core experimental question

**Setup**: Fix total compute budget for data generation (e.g., 1000 GPU-hours)

**Trade-off spectrum**:
- **High quantity, low quality**: Generate 1M examples from a small, fast model with greedy decoding
- **Medium quantity, medium quality**: Generate 100K examples from a medium model with best-of-4 sampling
- **Low quantity, high quality**: Generate 10K examples from a large model with best-of-16 + filtering

**Measure**: Train identical student models on each dataset and compare downstream performance

Where does the optimal point lie? How does it change with domain, model scale, and task type?

## Experimental matrix

Test across multiple dimensions:

### Dimension 1: Task domains

- **Math reasoning** (GSM8K, MATH): Verifiable correctness, benefits from high-quality reasoning
- **Code generation** (HumanEval, MBPP): Can test correctness, clear quality signal
- **Instruction following**: Subjective quality, harder to measure
- **Creative writing**: Quality is ambiguous, quantity may matter more
- **Factual QA**: Requires accuracy, quality likely important

Hypothesis: Math and code favor quality, creative writing favors quantity.

### Dimension 2: Student model scale

- Small (1B parameters): May saturate on small high-quality data
- Medium (7B parameters): May benefit from larger data even if lower quality
- Large (70B parameters): May need high-quality data to improve

Hypothesis: Larger students need higher quality to learn anything new; smaller students benefit more from quantity.

### Dimension 3: Quality dimensions

What aspects of quality matter most?

- **Correctness**: Is the answer right?
- **Reasoning quality**: Are intermediate steps valid?
- **Diversity**: How varied are the examples?
- **Difficulty**: Are examples challenging enough?
- **Format quality**: Is the format clean and consistent?

Systematically vary each dimension independently and measure impact.

### Dimension 4: Generation strategies

Compare different quality-quantity trade-offs:

1. **Model size**: Small model (high quantity) vs. large model (low quantity)
2. **Sampling strategy**: Greedy (fast, low quality) vs. best-of-N (slow, high quality)
3. **Filtering**: Generate lots, filter by quality vs. generate less with high temperature
4. **Refinement**: Single-pass generation vs. iterative self-refinement
5. **Verification**: Unfiltered vs. verified (math/code only)

Measure: Examples per compute hour, downstream student performance

## Detailed measurements

For each configuration:

**Data metrics**:
- Number of examples generated
- Generation cost (compute time, FLOPs)
- Quality scores (automated metrics, human eval samples)
- Diversity scores (unique n-grams, embedding clustering)
- Error rates (for verifiable domains)

**Training metrics**:
- Student model performance at different training checkpoints
- Convergence speed (steps to reach target performance)
- Final performance on test sets

**Analysis**:
- Pareto frontier: best performance for each compute budget
- Scaling curves: how does optimal trade-off shift with more total compute?
- Domain differences: where does quality matter most?

## Key questions to answer

1. **Is there a universal optimal point** on the quality-quantity trade-off, or is it highly domain-dependent?

2. **How much quality degradation can be compensated by increased quantity?**
   - If we 10x the data but halve the quality, is that better or worse?

3. **Does the optimal trade-off change during training?**
   - Early training: prefer quantity to cover distribution?
   - Late training: prefer quality to refine capabilities?

4. **What quality metrics best predict downstream performance?**
   - Correctness, diversity, difficulty, fluency?
   - Can we develop cheap proxies for expensive quality?

5. **How sensitive are results to student model capacity?**
   - Do small models bottleneck on capacity before data quality matters?
   - Do large models bottleneck on data quality before quantity matters?

## Practical guidelines as output

The study should produce actionable guidance:

**Decision tree format**:
```
IF domain = math/code AND student_model > 7B:
  THEN: Prioritize quality (use best-of-N, filtering)
ELSE IF domain = creative AND student_model < 3B:
  THEN: Prioritize quantity (use greedy sampling)
ELSE IF compute_budget > X:
  THEN: Use medium quality, medium quantity
...
```

**Scaling laws format**:
- Performance = f(quality, quantity, compute, domain, model_size)
- Fit empirical formula from experimental data

## Connection to existing work

- [Gunasekar et al., 2023](https://arxiv.org/abs/2306.11644) on textbook-quality data (focuses on quality)
- [Zhou et al., 2023](https://arxiv.org/abs/2305.11206) on self-improvement through large-scale generation (focuses on quantity)
- This study provides the missing systematic comparison

## Extensions

**Curriculum approach**:
- Start training with high-quantity low-quality data
- Transition to low-quantity high-quality data
- Does this combined approach beat either extreme?

**Iterative improvement**:
- Use student model to generate next round of data
- Does optimal trade-off change in recursive setting?

**Hybrid datasets**:
- Mix data of different qualities
- What's the optimal mixing ratio?

This empirical study would provide concrete, data-driven guidance for one of the most common decisions in modern LLM training: how to allocate synthetic data generation compute.
