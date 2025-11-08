# Training models to verify their own chain-of-thought reasoning

Chain-of-thought (CoT) prompting ([Wei et al., 2022](https://arxiv.org/abs/2201.11903)) improves reasoning by having models generate intermediate steps. However, these reasoning traces often contain errors that propagate to incorrect final answers. While process reward models (PRMs) can catch errors, they require expensive human annotation of reasoning steps.

This project explores training models to verify their own reasoning traces without external supervision, creating a self-contained generate-and-verify system.

## Motivation

Current approaches to verification:
- **Outcome verification**: Check if final answer is correct (requires ground truth)
- **Process reward models**: Judge intermediate steps (requires human annotation per step)
- **Self-consistency**: Generate multiple reasoning traces and take majority vote (expensive at inference)

Desired: A model that can generate a reasoning trace and then evaluate its own steps for logical validity, without needing ground truth or human step annotations.

## Proposed approach

Train a model with dual capabilities: **generation** and **verification**, using only outcome-level supervision.

### Phase 1: Generate diverse reasoning traces

For each problem, generate multiple CoT solutions with varied reasoning paths:
1. Use different temperatures, prompts, or model checkpoints
2. Collect both successful (correct final answer) and failed traces
3. This creates a dataset of (problem, reasoning_trace, outcome) tuples

### Phase 2: Train verification through consistency

Key insight: If a reasoning step appears consistently in successful traces but rarely in failed traces, it's likely valid. Use this to create step-level labels automatically:

1. **Alignment**: Match similar reasoning steps across different traces for the same problem (using semantic similarity)
2. **Consistency scoring**: For each step, compute:
   - P(step | successful trace) - P(step | failed trace)
   - Steps with high scores are likely valid, low scores likely invalid
3. **Step labeling**: Label steps as positive/negative based on consistency scores
4. **Train verifier**: Fine-tune model to predict step validity, using the format:
   - Input: "Problem: {problem}\nStep: {step}\nIs this step correct? (Yes/No)"
   - Output: Yes/No based on consistency-derived label

### Phase 3: Generate-and-verify

At inference time:
1. Generate CoT reasoning trace
2. After each step (or at the end), model verifies its own reasoning
3. If verification fails, model can:
   - Backtrack and try alternative reasoning
   - Indicate uncertainty in final answer
   - Generate multiple traces and select the one with highest verification scores

## Alternative: Contrastive verification training

Instead of explicit step labels, train the verifier to distinguish valid and invalid reasoning traces end-to-end:

1. Generate pairs of reasoning traces: one successful, one failed (for the same problem)
2. Train model to predict which trace is more likely to be correct
3. Use techniques from [Constitutional AI](https://arxiv.org/abs/2212.08073): model critiques its own reasoning and generates revisions

## Implementation details

- **Architecture options**:
  - Single model with special tokens to switch between generation and verification modes
  - Two separate fine-tuned models (generator and verifier) from the same base
  - Cross-attention between reasoning steps and verification queries

- **Training data**:
  - Use datasets with verifiable answers: math (MATH, GSM8K), code (HumanEval), logic puzzles
  - Generate 10-50 reasoning traces per problem with varied sampling strategies
  - Focus on problems where the model has partial but inconsistent success

## Key research questions

- Can models learn to verify reasoning accuracy using only outcome supervision?
- Does self-verification improve accuracy compared to standard CoT?
- How does verification quality scale with the number of diverse traces generated?
- Can models learn general verification skills that transfer to new problem domains?
- Does verification training improve the quality of generated reasoning traces?
- When self-verification disagrees with the model's initial answer, which should we trust?

## Evaluation

1. **Verification accuracy**: How well does the verifier identify incorrect reasoning steps?
   - Test on human-annotated step-level data (if available)
   - Measure correlation between verification scores and step correctness

2. **End-task performance**: Does generate-and-verify improve final answer accuracy?
   - Compare against: vanilla CoT, self-consistency, PRM-guided generation

3. **Error analysis**: What types of reasoning errors can the verifier catch vs. miss?
   - Logical fallacies, arithmetic errors, unstated assumptions, etc.

## Connection to scalable oversight

This project relates to the challenge of scalable oversight in AI alignment: as models become more capable, verifying their reasoning becomes harder. Self-verification could be a step toward models that can explain and justify their own reasoning in a way that's trustworthy and verifiable.

If models can reliably verify their own reasoning, this could enable:
- Iterative refinement of reasoning through self-critique
- Uncertainty quantification based on verification confidence
- Training signal for RL without external reward models
- Interpretability through explicit verification rationales

This combines the benefits of chain-of-thought reasoning (interpretability, improved accuracy) with self-verification (catching errors without human oversight).
