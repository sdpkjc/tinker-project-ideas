# Learning from negative preferences: What not to do

Current RLHF focuses on learning what good responses look like from positive and comparative preferences. But humans often have strong intuitions about what's *definitely wrong*—responses that are harmful, nonsensical, or severely off-topic—even when they're uncertain about what the *best* response would be. Standard preference modeling treats bad responses as simply "less preferred," but this may not capture the strength of negative examples.

This project explores **negative preference learning**: explicitly training models to avoid clearly unacceptable behaviors, separate from learning to pursue positive objectives.

## Motivation

Consider these scenarios:

**Scenario A**: User asks for cooking advice
- Response 1: Detailed recipe with clear steps (excellent)
- Response 2: Brief recipe with fewer details (okay)
- Response 3: Random poetry about unicorns (completely unacceptable)

**Scenario B**: User asks for medical information
- Response 1: Careful, accurate health information with disclaimers (excellent)
- Response 2: Somewhat less detailed but still accurate (okay)
- Response 3: Dangerous medical misinformation (completely unacceptable)

Standard preference ranking treats these as ordered lists: 1 > 2 > 3. But there's a *categorical difference* between "Response 2 is worse than 1" and "Response 3 is unacceptable." The former is about quality, the latter is about constraints.

## Proposed approach

Instead of learning a single reward model R(x, y), learn two components:

1. **Constraint function C(x, y)**: Binary or low-valued for unacceptable responses, high-valued for acceptable ones
2. **Quality function Q(x, y)**: Ranks acceptable responses by quality

Final reward: R(x, y) = C(x, y) · Q(x, y)

This factorization enforces that unacceptable responses receive low reward regardless of their "quality" in other dimensions.

## Data collection

Ask annotators to label responses in two stages:

**Stage 1 (Constraint)**: Is this response acceptable? (Binary: yes/no)
- Fast to annotate
- High inter-annotator agreement expected
- Focus on egregious failures: off-topic, harmful, incoherent

**Stage 2 (Quality)**: Among acceptable responses, rank by quality
- More nuanced judgment
- Only applied to responses that passed Stage 1

This separates the easier annotation task (identifying unacceptable responses) from the harder one (fine-grained quality ranking).

## Training procedure

1. **Train constraint function C**:
   - Binary classification on acceptable vs. unacceptable responses
   - Use a calibrated output (probability) to allow gradations
   - Heavily penalize false negatives (marking unacceptable as acceptable)

2. **Train quality function Q**:
   - Standard preference learning (Bradley-Terry model or similar)
   - Only train on responses labeled as acceptable
   - Can use pairwise comparisons as usual

3. **RL with factored reward**:
   - Use R(x, y) = C(x, y) · Q(x, y) as reward
   - Consider using a hard constraint: reject samples where C(x, y) < threshold
   - This creates a "safe" exploration region for the policy

## Research questions

1. **Does factorization improve safety?**
   - Does explicit negative preference learning reduce harmful outputs?
   - How does test-time performance compare to standard RLHF on safety evals?

2. **Data efficiency**:
   - Can we achieve better safety with fewer annotations by focusing human effort on identifying unacceptable responses?
   - Is inter-annotator agreement higher for negative preferences than positive ones?

3. **Generalization**:
   - Do learned constraints generalize to new types of harmful behavior?
   - Does the model learn general principles ("be on-topic," "avoid harm") or memorize specific bad patterns?

4. **Quality-safety trade-offs**:
   - Does strongly enforcing constraints harm helpfulness or creativity?
   - Can we tune the constraint threshold to control the safety-quality trade-off?

## Extensions

**Active learning for negatives**: Use the current policy to generate candidates, then ask annotators to identify unacceptable responses. This focuses annotation budget on the policy's most likely failures.

**Adversarial negative generation**: Train a model to generate maximally unacceptable responses (high Q but low C), then use these as hard negative examples.

**Compositional constraints**: Factor C into multiple constraint functions for different types of badness (harmfulness, off-topic, incoherence), allowing fine-grained control.

**Meta-learning constraint boundaries**: Different contexts may have different constraint boundaries (creative writing allows more freedom than medical advice). Learn to infer context-appropriate constraints.

## Connection to AI safety

This approach aligns with "avoiding bad outcomes" safety paradigms, complementing "achieving good outcomes" approaches. The factorization makes constraints explicit and auditable, rather than implicit in a single reward function.

By focusing human oversight on identifying unacceptable behaviors (which may be easier than defining optimal behaviors), we might achieve better safety properties with more efficient use of human annotation resources.
