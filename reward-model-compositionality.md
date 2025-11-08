# Compositional reward models: Combining atomic quality judgments

Current reward models output single scores, collapsing all quality dimensions into one number. This makes it hard to handle multi-objective optimization, user preferences, and context-specific quality criteria. This project develops compositional reward models that build complex judgments from atomic components, enabling flexible and interpretable evaluation.

## Core idea

**Atomic reward functions**:
- R_accurate: Is response factually correct?
- R_helpful: Does it address user's need?
- R_safe: No harmful content?
- R_concise: Appropriate length?
- R_clear: Easy to understand?

**Composition**:
```
R_total = f(R_accurate, R_helpful, R_safe, R_concise, R_clear)
```

Where f can be:
- Linear: `w1·R_accurate + w2·R_helpful + ...`
- Non-linear: Handle trade-offs and interactions
- Conditional: Depends on context

**Benefits**:
- **Flexible**: Change composition without retraining atomics
- **Interpretable**: See which aspects contribute
- **Transferable**: Reuse atomics across tasks
- **Personalizable**: Different users, different compositions

## Training atomic rewards

**Challenge**: Need labeled data for each dimension

**Approach 1: Multi-dimensional annotation**
- Collect ratings on each dimension separately
- Train specialized reward model per dimension
- Expensive but highest quality

**Approach 2: Weakly supervised**
- Use heuristics/automated metrics for some dimensions
- Human labels only for subjective dimensions
- Mix signals

**Approach 3: Synthetic decomposition**
- Train on overall preferences
- Use attention/gradients to attribute to dimensions
- Decompose holistic into atomics

## Composition functions

**Fixed composition**:
```
R = 0.4·R_accurate + 0.3·R_helpful + 0.2·R_safe + 0.1·R_concise
```
Simple, interpretable, weights set by designer

**Learned composition**:
Train meta-model: `f(atomics) → overall quality`
- Learns optimal combination from data
- Can capture non-linear interactions

**Context-dependent composition**:
```
weights = g(prompt)
R = weights · atomics
```
Different prompts value different dimensions

**Hierarchical composition**:
```
R_quality = R_accurate + R_helpful
R_style = R_concise + R_clear
R_total = R_quality + R_safe + α·R_style
```
Group related atomics

## Research questions

1. **Can we reliably decompose reward into atoms?**
   - Are dimensions truly independent?
   - How fine-grained should atomics be?

2. **Do atomics transfer across tasks?**
   - Train R_accurate on QA, use for summarization?

3. **What composition function works best?**
   - Linear, learned, contextual?

4. **Does compositionality improve flexibility?**
   - Easier to adapt to new requirements?

5. **Is interpretability preserved?**
   - Can users understand atomic contributions?

## Evaluation

- **Decomposition quality**: Do atomics match human dimension ratings?
- **Composition accuracy**: Does f(atomics) match overall preferences?
- **Flexibility**: How easily can we change objectives?
- **Transfer**: Do atomics generalize to new tasks?

## Applications

- **Multi-objective RL**: Optimize multiple dimensions with constraints
- **Personalization**: Users customize composition weights
- **Domain adaptation**: Reweight atomics per domain
- **Debugging**: Identify which dimension causes low scores
- **A/B testing**: Compare models on specific dimensions

## Extensions

- **Conditional atomics**: R_accurate(response | context)
- **Temporal composition**: Weights change during conversation
- **User-in-the-loop**: Users adjust weights in real-time
- **Automatic atomic discovery**: Learn decomposition from data
