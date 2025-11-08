# Cross-model debate tournament: Learning from diverse model perspectives

Current multi-agent debate systems typically use multiple samples from the same model or slight variants (temperature changes, prompt modifications). This limits diversity because all agents share the same underlying knowledge, biases, and failure modes. What if we construct a debate system using fundamentally different models—varying in size, architecture, training data, and capabilities?

This project proposes a **cross-model debate tournament** where heterogeneous models engage in structured argumentation, and we use the debate process as training signal for all participants.

## Core motivation

Different models have different strengths:
- Large models have more knowledge but can be overconfident
- Small models are more uncertain but sometimes avoid large model blind spots
- Models trained on different data have different biases
- Different model families (e.g., different base models, different post-training) have different reasoning styles

Current debate work doesn't exploit this heterogeneity. By pitting diverse models against each other, we might:
1. Extract more reliable truth-finding through diverse perspectives
2. Allow smaller models to learn from larger ones through debate
3. Make larger models more calibrated by exposing them to challenging counterarguments
4. Discover when model diversity leads to genuine insight vs. when models just have different biases

## Tournament structure

**Participants**: Select a diverse pool of models
- Different scales: 1B, 7B, 30B, 70B+ parameter models
- Different families: Multiple base model families (Llama, Mistral, Qwen, etc.)
- Different training: Base models, instruction-tuned, RLHF-trained
- Specialist vs. generalist: Code-specialized, math-specialized, general-purpose

**Debate format**: For each question, randomly assign models to positions
1. Round 1: Each model presents its answer with reasoning (200 tokens)
2. Round 2: Each model critiques opponents' answers (150 tokens)
3. Round 3: Each model gives final answer incorporating debate (100 tokens)
4. Judgment: Determine winner via:
   - Ground truth (for verifiable questions)
   - Super-judge model (large, capable model not in tournament)
   - Human evaluation (sample-based)

**Cross-pollination**: Each model receives training signal from debates:
- Positive examples: Its winning arguments and reasoning
- Negative examples: Its losing arguments
- Learning examples: Winning arguments from opponents that this model lost to

## Training signal extraction

From each debate, extract:

1. **Direct learning**: If model A lost to model B on question Q, add B's argument as a high-quality training example for A

2. **Contrastive learning**: Train model A to distinguish its winning arguments from its losing arguments (supervised by debate outcomes)

3. **Argument mining**: Extract specific reasoning steps that led to wins/losses, use these for process reward modeling

4. **Calibration data**: When a small model successfully challenges a large model, this is particularly valuable—it reveals cases where large models are overconfident

5. **Combination strategies**: Learn when to defer to other models (if model A consistently loses to model B on math questions, A can learn to express uncertainty in math)

## Tournament iterations

**Iteration 0**: Initial tournament with base set of models

**Iteration N → N+1**:
1. Run debates, collect outcomes
2. Fine-tune each model on its personalized training data from debates
3. Add updated models to tournament alongside originals
4. Track: Do models improve? Do debate dynamics shift?

**Meta-level questions**:
- Do smaller models catch up to larger ones over iterations?
- Do models learn to exploit specific weaknesses of opponents?
- Does the tournament converge to consensus or maintain diversity?
- Are there "rock-paper-scissors" dynamics where model A beats B, B beats C, C beats A?

## Evaluation dimensions

**Individual model improvement**:
- Accuracy on verifiable questions (math, code, factual QA)
- Calibration (confidence vs. correctness)
- Robustness to adversarial questions

**Debate-level metrics**:
- Truth-finding rate: How often does debate identify correct answer?
- Consensus emergence: Do models converge on answers after debate?
- Argument quality: Can humans identify which arguments are more sound?

**Diversity metrics**:
- Perspective diversity: Do different models raise genuinely different considerations?
- Failure mode overlap: Do different models make independent errors?
- Novel reasoning: Do debates produce reasoning paths not present in any individual model?

## Research questions

1. **Does heterogeneity improve truth-finding?**
   - Are debates among diverse models more accurate than homogeneous debates?
   - Which types of diversity matter most (scale, training, architecture)?

2. **Can small models learn from large ones via debate?**
   - Does exposure to larger models' reasoning improve small models beyond standard distillation?
   - Are there sweet spots where model size difference enables productive learning?

3. **Do large models benefit?**
   - Can smaller models successfully challenge large models, improving their calibration?
   - Do large models learn better argumentation strategies from facing diverse opponents?

4. **Emergent specialization**:
   - Do models naturally develop "expertise" in areas where they perform well in debates?
   - Can we identify which model to trust for which question types based on tournament history?

5. **Debate dynamics**:
   - Do tournament incentives lead to honest argumentation or strategic manipulation?
   - How do we prevent models from learning to "play to the judge" rather than seeking truth?

## Extensions

**Human-in-the-loop**: Add human debaters to the tournament, creating a hybrid human-AI debate system

**Recursive tournaments**: Winners of one tournament level compete in a higher-level tournament with stronger opponents

**Task specialization**: Run separate tournaments for different domains (math, code, reasoning, creative tasks) and compare dynamics

**Debate formats**: Experiment with different structures (Oxford-style, judicial cross-examination, collaborative refinement)

**Adversarial debate**: Some debates have adversarial setups where models are rewarded for winning rather than truth-finding—study how this affects training dynamics

This project combines ideas from multi-agent debate, ensemble learning, model distillation, and game theory, potentially revealing new ways to extract value from diverse model populations.
