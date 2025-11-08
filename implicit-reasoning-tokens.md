# Implicit reasoning tokens: Learning to think without showing work

Chain-of-thought prompting improves reasoning by generating explicit intermediate steps, but this has costs: increased latency, verbosity, and exposure of internal reasoning. Recent work on [Quiet-STaR](https://arxiv.org/abs/2403.09629) explores internal "thought tokens" not shown to users.

This project investigates training models to perform reasoning using implicit tokens—special tokens that exist in the forward pass for computation but are not part of the final output, enabling fast, efficient reasoning without verbose demonstrations.

## Core concept

**Standard CoT**: Generate visible reasoning steps, then answer
```
User: "What is 15 × 23?"
Model: "Let me calculate: 15 × 20 = 300, 15 × 3 = 45, so 300 + 45 = 345"
```

**Implicit reasoning**: Insert hidden reasoning tokens
```
User: "What is 15 × 23?"
Model: <think><think><think> "345"
(Internal computation happens, but user only sees answer)
```

**Benefits**:
- Faster inference (fewer tokens generated)
- Cleaner outputs (no verbose reasoning)
- More efficient (reasoning happens in latent space)
- Private (internal thoughts not exposed)

## Approach: Learnable reasoning tokens

### Architecture modification

Add special reasoning tokens to vocabulary:
- `<think_1>`, `<think_2>`, ..., `<think_K>` (K reasoning tokens)
- Model can generate these during forward pass
- Not rendered in output (filtered before showing user)

**Modified generation**:
```
Input → [tokens + <think> tokens] → Output (filter <think>)
```

**Key**: Model learns when and how to use `<think>` tokens for internal computation

### Training methodology

**Challenge**: How to teach model to use implicit reasoning tokens?

**Approach 1: Distillation from CoT**
1. Train teacher model with explicit CoT
2. Student model has access to `<think>` tokens
3. Distill: Student matches teacher's final answers using `<think>` instead of visible tokens

**Approach 2: Reinforcement learning**
1. Model can generate `<think>` tokens (not penalized)
2. Reward only final visible output
3. RL discovers that `<think>` tokens help without penalty
4. Model learns to reason implicitly

**Approach 3: Auxillary prediction loss**
1. Insert `<think>` tokens in forward pass
2. Train auxiliary heads to predict intermediate reasoning states
3. Model learns to use `<think>` for computation even if not generating text

**Approach 4: Compress CoT into implicit tokens**
1. Start with full CoT training
2. Gradually replace visible reasoning with `<think>` tokens
3. Curriculum: Explicit → mixed → implicit
4. Model learns compressed reasoning

## Research questions

1. **Can models learn effective implicit reasoning?**
   - Do `<think>` tokens actually help performance?
   - Or are they ignored/unused?

2. **How many implicit tokens are needed?**
   - 5? 10? 50?
   - Trade-off: More tokens = more computation but diminishing returns?

3. **What computations happen in implicit tokens?**
   - Interpretability: Can we decode what `<think>` tokens compute?
   - Probe activations, attention patterns

4. **Does implicit reasoning transfer across tasks?**
   - Learn on math, apply to code?
   - General reasoning skill vs. task-specific?

5. **Efficiency gains?**
   - How much faster than explicit CoT?
   - Quality vs. speed trade-off?

## Evaluation

**Metric 1: Task performance**
- Compare accuracy: Implicit reasoning vs. explicit CoT vs. no reasoning
- Test on: Math, logic, code, commonsense reasoning

**Metric 2: Efficiency**
- Tokens generated (implicit should be fewer)
- Inference latency
- Computational cost (FLOPs)

**Metric 3: Ablation studies**
- Remove `<think>` tokens, measure performance drop
- Vary number of `<think>` tokens, find optimal

**Metric 4: Interpretability**
- Probe: What information is encoded in `<think>` token activations?
- Do they represent reasoning steps?

## Extensions

- **Adaptive reasoning**: Model learns how many `<think>` tokens needed per problem
- **Hybrid reasoning**: Mix explicit and implicit as needed
- **User control**: Toggle between showing reasoning (debug mode) and hiding it (production)
- **Multi-level thinking**: Different `<think>` token types for different reasoning levels
