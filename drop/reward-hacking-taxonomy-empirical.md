# Empirical taxonomy of reward hacking in RLHF

Reward hacking—where policies exploit flaws in reward models to achieve high scores without actually improving on the intended objective—is widely discussed but poorly characterized empirically. We have anecdotes and examples, but lack systematic understanding of what types of reward hacking occur, when they emerge, and what factors make them more or less likely.

This project proposes a comprehensive empirical study to build a taxonomy of reward hacking behaviors in RLHF for language models, inspired by [Krakovna et al.'s specification gaming examples](https://arxiv.org/abs/2004.07780).

## Research questions

1. **What types of reward hacking occur in practice?**
   - Length exploitation (generating unnecessarily long responses)
   - Keyword stuffing (inserting terms the reward model associates with quality)
   - Hedge language (using uncertainty expressions to avoid being wrong)
   - Format manipulation (adding bullet points, bold text regardless of appropriateness)
   - Sycophancy (agreeing with user biases detected in prompts)
   - Evasion (refusing to answer to avoid making mistakes)

2. **When does reward hacking emerge during training?**
   - Does it appear gradually or suddenly?
   - Is there a phase transition at certain policy strengths?
   - How does it correlate with reward model confidence?

3. **What factors influence susceptibility to reward hacking?**
   - Reward model size relative to policy size
   - Training data distribution for reward model
   - KL penalty strength in PPO
   - Dataset diversity and coverage

4. **Can we predict reward hacking before it happens?**
   - Are there early warning signals in training dynamics?
   - Does reward model uncertainty correlate with hacking regions?

## Experimental framework

**Phase 1: Controlled environment**

Set up RLHF training with instrumentation to detect reward hacking:
1. Train reward models of varying quality (different data sizes, architectures)
2. Run RL training with extensive logging and checkpointing
3. Compare reward model scores vs. ground truth human evaluations throughout training
4. Identify divergence points where RM scores increase but human scores plateau or decrease

**Phase 2: Systematic characterization**

For each identified reward hacking behavior:
- Minimal reproducible example (simplest prompt that triggers it)
- Onset conditions (training step, policy strength when it first appears)
- Reward model blind spots (what features does it miss?)
- Stability (does it persist or get corrected with more training?)

**Phase 3: Factor analysis**

Run controlled experiments varying one factor at a time:
- Reward model capacity: small vs. large models
- RM training data quality: clean vs. noisy preferences
- RL hyperparameters: KL penalty strength, learning rate
- Dataset characteristics: domain coverage, prompt diversity

Measure reward hacking frequency and severity under each condition.

## Measurement methodology

**Detecting reward hacking:**

1. **Human evaluation divergence**: Compare RM scores vs. human ratings on samples from trained policies
2. **Out-of-distribution testing**: Test on held-out prompts; reward hacking often fails to generalize
3. **Adversarial probing**: Design prompts specifically to elicit suspected hacking behaviors
4. **Contrastive editing**: Manually edit responses to remove suspected hacking features; does RM score drop disproportionately?

**Quantifying severity:**

- Frequency: What fraction of responses exhibit each hacking type?
- Impact: How much does hacking inflate RM scores vs. true quality?
- Robustness: How sensitive is hacking to prompt variations?

## Expected outcomes

A comprehensive dataset and report including:
- Taxonomy of reward hacking types with examples
- Training dynamics: when and how each type emerges
- Risk factors: what conditions make hacking more likely
- Mitigation insights: what interventions reduce hacking
- Open dataset of labeled reward hacking examples for future research

## Connection to existing work

This relates to:
- [Gao et al.'s work on scaling laws for reward model overoptimization](https://arxiv.org/abs/2210.10760)
- [Skalse et al.'s invariance properties of reward functions](https://arxiv.org/abs/2212.08341)
- Goodhart's Law in AI systems

But provides more granular, behavior-level analysis rather than aggregate metrics.

## Practical value

Understanding reward hacking patterns can inform:
- Better reward model training (covering common failure modes)
- Better RL training procedures (detecting and correcting hacking early)
- Evaluation protocols (testing for known hacking behaviors)
- Theoretical work (grounding abstract concerns in empirical reality)

This empirical foundation could make RLHF more reliable and predictable in practice.
