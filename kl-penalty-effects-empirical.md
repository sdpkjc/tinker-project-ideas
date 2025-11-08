# Empirical study of KL penalty effects in RLHF: stability vs. capability trade-offs

The KL divergence penalty in RLHF constrains how far the trained policy can diverge from a reference model (usually the pre-trained base model). This penalty serves multiple purposes: preventing mode collapse, maintaining language quality, and ensuring training stability. However, the KL penalty also limits how much the policy can improve—too strong and the policy can't learn, too weak and training becomes unstable.

Despite its critical role, the effects of KL penalty strength are not well-characterized empirically. This study systematically investigates how KL penalty affects training dynamics, final performance, and failure modes.

## Research questions

1. **What is the relationship between KL penalty and final performance?**
   - Is there an optimal KL budget for maximizing capability?
   - Is the relationship monotonic or are there phase transitions?
   - How does this differ across task domains?

2. **How does KL penalty affect training dynamics?**
   - Convergence speed: Do higher penalties slow down learning?
   - Stability: Do lower penalties cause more instability or divergence?
   - Reward optimization: At what KL level does reward growth plateau?

3. **What behaviors emerge at different KL regimes?**
   - Very low KL (< 1 nat): Policy stays very close to base model, minimal improvement?
   - Low KL (1-5 nats): Stable training, moderate improvement?
   - Medium KL (5-20 nats): Higher capability but potential instability?
   - High KL (> 20 nats): Maximum improvement but risk of mode collapse?

4. **Does optimal KL penalty depend on context?**
   - Model scale (1B vs. 70B parameters)
   - Task type (instruction following vs. reasoning vs. creative writing)
   - Reward model quality (strong vs. weak reward model)
   - Training data diversity

5. **Can we identify failure modes associated with KL choices?**
   - Too high KL: Nonsensical text, mode collapse, reward hacking
   - Too low KL: Insufficient improvement, unable to correct base model errors
   - Early warning signals for each failure mode

## Experimental framework

**Setup**: Run RLHF training (PPO) with systematic variation of KL penalty coefficient β

**KL sweep experiments**:
- Train 20-30 policies with different β values spanning wide range
- β ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0}
- All other hyperparameters held constant
- Multiple random seeds for each β

**Measurement during training** (logged at each checkpoint):
- KL divergence from reference model
- Reward model scores (in-distribution and out-of-distribution)
- Perplexity on held-out text
- Human evaluation win rates (sampled checkpoints)
- Automated benchmarks (MMLU, HumanEval, etc.)
- Response length and format statistics
- Training stability metrics (gradient norms, loss variance)

**Final evaluation**:
- Task performance across multiple domains
- Qualitative analysis of outputs
- Robustness tests (adversarial prompts, distribution shift)
- Failure mode classification

## Detailed analysis

### Analysis 1: Performance curves

Plot performance vs. KL budget:
- Does performance increase monotonically with KL budget?
- Where is the optimal point (maximum performance)?
- Where do diminishing returns begin?
- How much does this vary across domains?

### Analysis 2: Pareto frontier

For each β, plot: (KL from base model, Task performance)
- What is the Pareto frontier (best performance for each KL level)?
- How does this frontier change with:
  - Model size
  - Task domain
  - Reward model quality

### Analysis 3: Training dynamics

Compare learning curves for different β:
- Time to convergence
- Stability (variance in metrics across training)
- Monotonicity (do metrics steadily improve or fluctuate?)
- Phase transitions (sudden changes in behavior)

### Analysis 4: Failure mode taxonomy

Categorize and count failure modes by KL regime:

**Low KL failures**:
- Insufficient improvement (base model errors persist)
- Overly conservative responses
- Unable to learn new behaviors

**High KL failures**:
- Repetitive text or mode collapse
- Nonsensical or low-fluency outputs
- Extreme reward hacking (exploiting RM without semantic quality)
- Catastrophic forgetting (loss of pre-training capabilities)

### Analysis 5: Reward hacking vs. KL

Test hypothesis: Reward hacking becomes more severe at higher KL divergence
- For each β, measure: RM score vs. human rating divergence
- Identify: At what KL level does reward hacking become problematic?

## Domain-specific investigations

Run the full KL sweep on diverse domains:

1. **Math reasoning (GSM8K, MATH)**:
   - Hypothesis: May need higher KL to learn reasoning patterns different from base model
   - Measure: Accuracy vs. KL budget

2. **Code generation (HumanEval)**:
   - Hypothesis: Medium KL optimal—need to move from base model but stay readable
   - Measure: Pass@1, pass@10 vs. KL

3. **Instruction following**:
   - Hypothesis: Lower KL sufficient—base model already fluent, just needs steering
   - Measure: Instruction-following eval scores vs. KL

4. **Creative writing**:
   - Hypothesis: Higher KL beneficial for diversity and creativity
   - Measure: Diversity metrics, human preference vs. KL

5. **Factual QA**:
   - Hypothesis: Lower KL preferred to avoid hallucination drift
   - Measure: Factual accuracy vs. KL

## Practical guidelines as output

Provide concrete recommendations:

**Default values**:
- For instruction following with 7B model: β = X
- For reasoning tasks with 70B model: β = Y
- For creative tasks with 1B model: β = Z

**Adaptive schedules**:
- Start with β = X (exploration phase)
- Anneal to β = Y (consolidation phase)
- Validated through experiments

**Warning signals**:
- If perplexity increases by >X%, KL too high
- If reward plateau at <X improvement, KL too low
- Concrete thresholds from empirical data

## Connection to existing work

- [Gao et al., 2022](https://arxiv.org/abs/2210.10760) studies overoptimization but not KL effects specifically
- [Ouyang et al., 2022](https://arxiv.org/abs/2203.02155) (InstructGPT) uses KL penalty but doesn't study its effects systematically
- This study provides missing empirical characterization

## Extensions

**Adaptive KL penalties**:
- Based on empirical findings, design adaptive schemes
- Increase KL budget for hard prompts, decrease for easy ones
- Per-domain KL coefficients

**Alternative divergence measures**:
- Compare KL vs. other divergences (JS, Wasserstein, etc.)
- Does choice of divergence measure matter?

**KL targeting**:
- Instead of fixed β, directly target specific KL level
- Does this provide better control?

This empirical study would demystify one of the most important but least understood hyperparameters in RLHF, providing practical guidance for training more capable and stable policies.
