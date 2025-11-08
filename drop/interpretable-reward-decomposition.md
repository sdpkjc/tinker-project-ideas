# Interpretable reward decomposition: Explaining why responses are preferred

Reward models in RLHF are black boxes—they output a scalar score but don't explain why one response is better than another. This opacity creates problems: difficulty debugging reward model failures, inability to verify that RM captures intended preferences, and challenges in building trust with users.

This project develops methods for interpretable reward models that decompose overall preference into understandable factors and provide natural language explanations, making RLHF more transparent and debuggable.

## Motivation

**Current RM**: (prompt, response) → scalar score

**Problems with opacity**:
- Can't debug: Why did RM prefer response A over B?
- Can't verify: Does RM reward what we think it rewards?
- Can't trust: Users/developers can't understand RM decisions
- Can't improve: Hard to identify RM weaknesses

**Desired**: Interpretable RM that explains its judgments
- "Response A is better because: [+0.3 more accurate, +0.2 more helpful, -0.1 less concise]"
- Enables debugging, verification, and trust

## Approaches to interpretable reward decomposition

### Approach 1: Explicit factor decomposition

Train RM to output scores for multiple interpretable dimensions:

**Architecture**:
```
(prompt, response) → RM → {
  accuracy_score: 0.8,
  helpfulness_score: 0.9,
  safety_score: 1.0,
  conciseness_score: 0.6,
  formatting_score: 0.7
} → weighted_sum → overall_score
```

**Training**:
- Collect dimension-specific annotations (expensive but gold standard)
- Or use weak supervision (heuristics, prompted LLMs) for dimension scores
- Train RM to predict both overall preference and dimension scores
- Multi-task learning with shared representations

**Benefits**:
- Clear decomposition into interpretable factors
- Can adjust weights for different use cases
- Easy to debug (check dimension scores)

**Challenges**:
- Defining dimensions (which factors matter?)
- Collecting dimension-specific labels (expensive)
- Factors may not be independent

### Approach 2: Natural language explanations

Train RM to generate explanations alongside scores:

**Architecture**:
```
(prompt, response_A, response_B) → RM → {
  preference: A > B,
  explanation: "Response A provides more accurate information and directly addresses the question, while Response B is vaguer and less helpful."
}
```

**Training**:
- Collect human explanations for preferences (expensive)
- Or train RM to generate explanations via:
  - **Self-rationalization**: RM generates explanation, check if it predicts the preference
  - **Synthetic explanations**: Use prompted LLM to explain preferences, train RM on these

**Benefits**:
- Natural language is inherently interpretable
- Flexible (can explain any aspect)
- Useful for users and developers

**Challenges**:
- Generating faithful explanations (not post-hoc rationalization)
- Verifying explanation quality

### Approach 3: Attention-based interpretation

Use model internals to explain decisions:

**Method**:
1. RM attends to specific parts of response when scoring
2. Extract attention weights: which tokens/spans matter most?
3. Visualize: Highlight parts of response that most influence score

**Technique**: Attention rollout, integrated gradients, LIME

**Benefits**:
- No additional training needed (use existing RM)
- Grounded in actual model computation
- Can identify surprising focus areas

**Challenges**:
- Attention may not be faithful to true reasoning
- Hard to interpret (which attention heads/layers matter?)
- Technical, not user-friendly

### Approach 4: Contrastive explanations

Explain preference through minimal contrastive edits:

**Method**:
1. Given A > B, find minimal change to B that would make B > A
2. Example: "Changing 'Paris' to 'Rome' in response B would make it correct"
3. This identifies what aspects matter for preference

**Implementation**:
- Search over edits to response B
- Find edits that flip RM's preference
- Present: "Response A is better because [aspect X], as shown by [counterfactual edit]"

**Benefits**:
- Contrastive explanations are cognitively natural
- Identifies specific decision factors
- Useful for understanding RM sensitivities

### Approach 5: Concept-based interpretation

Learn interpretable concepts and attribute scores to them:

**Method**:
1. Discover or specify interpretable concepts:
   - Factual accuracy
   - Logical coherence
   - Helpfulness
   - Conciseness
   - Safety

2. Train concept detectors: Does response exhibit concept C?
3. Learn attribution: How much does concept C contribute to overall score?
4. Decompose: score = Σ attribution(concept_i) × presence(concept_i)

**Inspired by**: [Concept Activation Vectors](https://arxiv.org/abs/1711.11279)

**Benefits**:
- Grounded in human-understandable concepts
- Quantifiable attribution
- Can test causal role of concepts

## Training methodologies

### Training approach 1: Multi-task learning

**Setup**:
- Main task: Predict overall preference (A > B)
- Auxiliary tasks: Predict dimension scores, generate explanations
- Shared representation, multiple heads

**Loss**:
```
L_total = L_preference + λ₁·L_dimensions + λ₂·L_explanations
```

**Benefit**: Joint training encourages interpretability

### Training approach 2: Explanation-supervised learning

**Setup**:
1. Collect (prompt, response_A, response_B, preference, explanation) tuples
2. Train RM to predict preference
3. Verify: Does RM internals align with explanation?
4. Penalize if RM prediction relies on features not mentioned in explanation

**Benefit**: Ensures RM reasoning matches human reasoning

### Training approach 3: Self-consistency training

**Setup**:
1. RM generates explanation for preference
2. Mask response, provide only explanation
3. RM should still predict same preference from explanation alone
4. If not, explanation is not faithful

**Benefit**: Encourages faithful explanations

## Evaluation methodology

### Metric 1: Explanation quality

**Human evaluation**:
- Show humans (response_A, response_B, RM's explanation)
- Ask: Does explanation match your preference? Is it helpful?
- Measure: Agreement, helpfulness ratings

### Metric 2: Faithfulness

**Test**: Do explanations reflect actual RM reasoning?

**Method**:
- Ablation: Remove factors mentioned in explanation, check if score changes as expected
- Counterfactual: Edit response to address explanation, check if score improves
- Consistency: Do similar explanations lead to similar predictions?

### Metric 3: Decomposition accuracy

**For explicit factor decomposition**:
- Collect ground truth dimension scores from humans
- Measure: Accuracy of RM's dimension predictions
- Test: Does dimension decomposition sum to overall score correctly?

### Metric 4: Debugging utility

**Practical test**:
- Give developers interpretable RM
- Task: Identify and fix RM failures
- Measure: Time to identify bug, success rate
- Compare: Interpretable vs. black-box RM

### Metric 5: User trust

**Study**:
- Deploy system with/without explanations
- Measure: User trust, satisfaction, task success
- Question: Do explanations improve user experience?

## Applications of interpretable RMs

### Application 1: Debugging reward models

**Use case**: RM gives surprising scores

**With interpretability**:
- Check dimension scores: Which factor is wrong?
- Read explanation: Does reasoning make sense?
- Identify root cause faster

### Application 2: Verifying alignment

**Use case**: Ensure RM captures intended preferences

**With interpretability**:
- Check if RM focuses on right factors
- Example: Does safety RM actually detect unsafe content?
- Provides evidence of alignment

### Application 3: User control and personalization

**Use case**: Different users have different preferences

**With interpretability**:
- Show dimension scores to users
- Let users adjust dimension weights
- Personalized RM based on user feedback

### Application 4: Training data improvement

**Use case**: Identify weaknesses in preference dataset

**With interpretability**:
- Check what factors RM learns vs. should learn
- Identify missing dimensions in training data
- Guide targeted data collection

### Application 5: Policy debugging

**Use case**: Policy after RLHF behaves unexpectedly

**With interpretability**:
- Check what RM rewards in policy outputs
- Identify if policy is exploiting unintended RM features
- Diagnose reward hacking

## Key research questions

1. **Can we make RMs interpretable without sacrificing accuracy?**
   - Trade-off between interpretability and performance?

2. **Are explanations faithful?**
   - Do they reflect actual RM reasoning or post-hoc rationalization?

3. **What level of interpretation is useful?**
   - Dimension scores? Natural language? Attention? Counterfactuals?
   - User studies to determine preferences

4. **Can interpretability improve RLHF outcomes?**
   - Better debugging → better RMs → better policies?

5. **Do users trust interpretable RMs more?**
   - Does interpretability increase adoption and satisfaction?

## Practical implementation

**Phase 1: Dimension decomposition**
- Define key dimensions (accuracy, helpfulness, safety, etc.)
- Train multi-head RM predicting each dimension
- Test: Does decomposition help debugging?

**Phase 2: Explanation generation**
- Fine-tune RM to generate natural language explanations
- Evaluate explanation quality and faithfulness
- Test: Are explanations useful for developers?

**Phase 3: User interface**
- Build interface showing dimension scores and explanations
- Deploy to users
- Collect feedback on interpretability value

**Phase 4: Integration with RLHF**
- Use interpretable RM for full RLHF pipeline
- Monitor: Does interpretability improve training process?

## Extensions

**Hierarchical explanations**:
- High-level summary + detailed breakdown
- Adapt explanation depth to user expertise

**Interactive interpretation**:
- Users can query: "Why did you prefer A over B?"
- RM generates explanation on demand

**Contrastive interpretation**:
- Compare two RMs' decisions side-by-side
- Explain: Why do these RMs disagree?

**Temporal interpretation**:
- Track how RM reasoning changes during training
- Visualize: What does RM learn over time?

**Cross-cultural interpretation**:
- Do explanations differ across languages/cultures?
- Adapt interpretation to cultural context

By making reward models interpretable, we can build more trustworthy, debuggable, and verifiable RLHF systems—critical for safety, reliability, and user adoption.
