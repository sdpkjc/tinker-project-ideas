# Compute-optimal RLHF: Scaling laws for post-training

Scaling laws for pre-training ([Chinchilla](https://arxiv.org/abs/2203.15556), [Kaplan et al.](https://arxiv.org/abs/2001.08361)) guide compute allocation between model size and training data. However, no equivalent exists for RLHF: How should we allocate post-training compute between RM training, policy training, data collection, and model scale?

This project develops scaling laws for RLHF to answer: Given X compute budget for post-training, what's the optimal allocation to maximize final policy quality?

## Key questions for RLHF scaling

**Resource allocation decisions**:

1. **RM vs. policy compute**:
   - Invest in better reward model (more data, larger model)?
   - Or invest in longer policy training?

2. **Data vs. training**:
   - Collect more preference data?
   - Or train longer on existing data?

3. **Model scale**:
   - Larger RM and policy?
   - Or smaller models with more data/training?

4. **RL vs. SFT budget**:
   - More supervised pre-training?
   - Or more RL fine-tuning?

**Goal**: Derive scaling laws: `Policy_Quality = f(RM_compute, Policy_compute, Data, Model_size)`

## Experimental framework

### Experiment 1: RM compute vs. policy compute

**Setup**:
- Fix total compute budget C
- Vary allocation: α·C for RM, (1-α)·C for policy
- α ∈ {0.1, 0.3, 0.5, 0.7, 0.9}

**Measure**: Final policy quality vs. α

**Find**: Optimal α* (best allocation)

### Experiment 2: Data quantity vs. training compute

**Setup**:
- Fix compute budget
- Vary: N preference labels, T training steps
- Constraint: Cost(N labels) + Cost(T steps) = C

**Measure**: Policy quality vs. (N, T) combinations

**Find**: Pareto frontier, optimal (N*, T*)

### Experiment 3: Model scale vs. data

**Setup**:
- Fix compute budget
- Vary: Model size S, Preference data N
- Constraint: Cost(S) + Cost(N) = C

**Find**: Optimal (S*, N*) for different C values

**Derive**: Scaling law: How does optimal S scale with C?

### Experiment 4: Multi-dimensional optimization

**Full experiment**:
- Vary all factors: RM size, policy size, RM data, RL steps
- Fit: Quality = f(RM_size, Policy_size, Data, Steps, ...)
- Find: Compute-optimal allocation for any budget C

## Scaling law formulation

**Hypothesized form** (inspired by pre-training scaling laws):

```
Quality ∝ (RM_compute)^α · (Policy_compute)^β · (Data)^γ
```

Where α, β, γ are exponents to be empirically determined

**Or interaction terms**:
```
Quality ∝ (RM_size · RM_data)^α · (Policy_size · RL_steps)^β
```

**Goal**: Fit parameters from experimental data

## Practical implications

**If we find** (hypothetical results):
- RM compute has diminishing returns after certain point
- Policy training scales well with compute
- Data quantity dominates at low budgets

**Recommendation**:
- Low budget: Maximize data collection
- Medium budget: Balanced RM and policy
- High budget: Invest in policy training

**Similar to Chinchilla findings for pre-training**

## Research questions

1. **Do RLHF scaling laws exist?**
   - Smooth, predictable relationships?
   - Or noisy and unpredictable?

2. **What factors matter most?**
   - Data quantity? RM quality? Policy compute?

3. **Do scaling laws transfer across domains?**
   - Same laws for code, math, general chat?

4. **How do scaling laws interact with RL algorithms?**
   - PPO vs. DPO: Different optimal allocations?

5. **Can we predict performance before training?**
   - Given compute budget, predict final quality?

## Evaluation

- **Fit quality**: How well do scaling laws fit empirical data?
- **Prediction accuracy**: Can we predict held-out configurations?
- **Cross-validation**: Do laws generalize to new settings?
- **Practical value**: Do recommendations improve real deployments?

## Extensions

- **Algorithm-specific scaling laws**: PPO, DPO, REINFORCE
- **Task-specific laws**: Math, code, chat
- **Multi-objective scaling**: Quality vs. safety vs. efficiency
- **Continual learning scaling**: How does continuous RLHF scale?
