# Empirical analysis of attention patterns in chain-of-thought reasoning

Chain-of-thought (CoT) prompting improves reasoning by generating intermediate steps, but we lack detailed understanding of how models actually use these steps mechanistically. Do models genuinely reason through the steps, or do they generate them as a side effect while reasoning in other ways?

Recent interpretability work ([Wang et al., 2023](https://arxiv.org/abs/2305.00050)) shows that attention patterns can reveal information flow in transformers. This empirical study systematically analyzes attention patterns during CoT reasoning to understand:
- Whether models actually attend to their own reasoning steps
- Which steps are most important
- How attention patterns differ between correct and incorrect reasoning
- Whether we can predict reasoning success from attention patterns

## Core research questions

1. **Do models attend to their reasoning steps?**
   - When generating step N, how much attention goes to steps 1..N-1?
   - Contrast with baseline: attention in non-CoT generation

2. **Which reasoning steps matter most?**
   - Do certain steps receive disproportionate attention?
   - Do "key insights" show distinctive attention patterns?

3. **How do attention patterns differ in success vs. failure?**
   - Can we distinguish correct and incorrect reasoning by attention?
   - Are there attention "signatures" of reasoning errors?

4. **Do attention patterns predict reasoning quality?**
   - Can we predict final answer correctness from intermediate attention?
   - Early warning signals for reasoning failure?

5. **How do attention patterns evolve during training?**
   - Do models learn to attend to reasoning steps through RL or fine-tuning?
   - Comparison: base model vs. reasoning-trained model

## Experimental framework

### Data collection

**Datasets**: Problems with verifiable answers
- Math: GSM8K, MATH
- Logic: ARC reasoning, proof problems
- Code: HumanEval with test-driven reasoning

**Models**: Compare models of varying reasoning ability
- Base pre-trained models (weak reasoning)
- Instruction-tuned models (medium reasoning)
- RLHF models trained on reasoning (strong reasoning)
- Compare across sizes: 7B, 13B, 34B, 70B

**Generation setup**:
- For each problem, generate multiple CoT solutions
- Separate: correct solutions (right answer) vs. incorrect (wrong answer)
- Extract attention weights at each reasoning step during generation

### Attention extraction methodology

For each generated token during CoT:
- Extract attention weights from all layers and heads
- Focus on attention to previous tokens (especially previous reasoning steps)
- Aggregate attention patterns across:
  - Heads (which heads attend where?)
  - Layers (which layers show reasoning attention?)
  - Token positions (which previous tokens get attended to?)

### Analysis 1: Attention flow during reasoning

**Metric**: Attention concentration on reasoning steps

For each token in step N, compute:
- `attention_to_reasoning = Σ attention[step i] for i < N`
- `attention_to_problem = Σ attention[problem tokens]`
- `attention_to_other = remaining attention`

**Hypothesis**: In successful reasoning, attention_to_reasoning should be high

**Visualization**: Attention flow diagrams showing information propagation through reasoning steps

### Analysis 2: Critical step identification

**Method**:
For each reasoning trace, identify which previous steps receive most attention when generating the final answer

**Questions**:
- Do models focus on specific "key" steps (e.g., the problem setup, a crucial intermediate result)?
- Do different models identify the same key steps?
- Do key steps correlate with human judgment of important reasoning?

**Visualization**: Heatmaps showing attention from final answer tokens to all previous reasoning steps

### Analysis 3: Attention patterns in correct vs. incorrect reasoning

**Method**:
Contrast attention patterns between:
- Traces leading to correct answers
- Traces leading to incorrect answers

**Metrics**:
- Attention entropy (do correct traces have more focused attention?)
- Attention to specific step types (computation steps, logical steps, etc.)
- Attention backtracking (do incorrect traces show less coherent attention flow?)

**Analysis**:
- Train classifier: Can we predict correctness from attention patterns?
- Feature importance: Which attention features most distinguish correct/incorrect?

### Analysis 4: Layer-wise and head-wise decomposition

**Question**: Which layers and attention heads are responsible for reasoning attention?

**Method**:
- Separate analysis for each layer (1-32+) and head (1-32+)
- Identify "reasoning heads" that specifically attend to previous reasoning steps
- Test: Are reasoning heads consistent across problems?

**Connection to interpretability**:
- Do reasoning heads emerge through training?
- Can we ablate reasoning heads to impair reasoning?
- Similar to [circuit analysis](https://distill.pub/2020/circuits/) in vision models

### Analysis 5: Training dynamics of attention

**Setup**:
- Take checkpoints throughout reasoning training (SFT or RL on reasoning tasks)
- Analyze attention patterns at each checkpoint

**Questions**:
- How do attention patterns change as model learns to reason?
- Do reasoning-specific attention patterns emerge?
- Is there a critical point where attention "clicks" into reasoning mode?

## Practical predictions and applications

**Application 1: Early error detection**

If incorrect reasoning has distinctive attention patterns:
- Monitor attention during generation
- Flag anomalous patterns for human review or regeneration
- Improve reliability of reasoning systems

**Application 2: Reasoning scaffolding**

If we identify critical attention patterns:
- Design prompts that encourage beneficial attention flows
- Structure reasoning steps to facilitate important attention patterns
- Format optimization for reasoning quality

**Application 3: Model selection**

If attention patterns predict reasoning capability:
- Quickly evaluate new models by attention analysis
- Select models with strongest reasoning attention patterns
- Cheaper than extensive behavioral testing

**Application 4: Interpretability**

Understanding attention in reasoning:
- Explains how models use CoT steps mechanistically
- Grounds interpretability in attention—a measurable, mechanistic property
- Helps distinguish genuine reasoning from superficial pattern matching

## Key measurements to report

For each model and dataset:

1. **Attention to reasoning steps**: Mean attention weight from answer tokens to reasoning tokens
2. **Attention concentration**: Entropy of attention distribution (lower = more focused)
3. **Critical step attention**: Attention to top-3 most-attended reasoning steps
4. **Correctness prediction**: Accuracy of classifier trained on attention patterns
5. **Layer analysis**: Which layers show strongest reasoning attention
6. **Head analysis**: Number and properties of reasoning-specialized heads

## Expected insights

This study could reveal:
- Whether CoT actually involves attention-based reasoning or is superficial
- Mechanistic understanding of how reasoning works in transformers
- Practical signals for monitoring and improving reasoning quality
- Differences between models in how they use reasoning steps

## Limitations and challenges

**Challenge**: Attention is not the only mechanism for information flow
- Residual connections bypass attention
- Feed-forward layers process information without attention

**Mitigation**: Complement attention analysis with activation patching and other interpretability techniques

**Challenge**: Correlation vs. causation
- Attention patterns correlate with reasoning, but do they cause it?

**Mitigation**: Ablation studies—modify attention patterns and measure impact on reasoning

## Extensions

**Causal attention intervention**:
- Artificially strengthen/weaken attention to certain reasoning steps
- Measure impact on final answer quality
- Establish causal role of attention in reasoning

**Cross-model comparison**:
- Compare attention patterns across different model families
- Do all models reason the same way, or are there different strategies?

**Cross-domain transfer**:
- Do attention patterns learned in math transfer to code or logic?
- Are there universal reasoning attention patterns?

This empirical analysis would provide unprecedented mechanistic insight into how language models actually use chain-of-thought reasoning, moving beyond behavioral observations to understanding the underlying computational mechanisms.
