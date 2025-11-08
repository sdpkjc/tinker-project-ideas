# World model-guided text generation for consistency and coherence

Language models generate text token-by-token without explicit world models, leading to consistency errors: contradicting earlier statements, violating physical laws, or generating incoherent scenarios. Recent work in game AI uses world models to ensure consistency ([World Models](https://arxiv.org/abs/1803.10122), [DreamerV3](https://arxiv.org/abs/2301.04104)).

This project explores training LLMs with explicit world models—learned representations of entities, relationships, and states—that guide generation to maintain consistency and coherence over long texts.

## Problem: Inconsistency in long-form generation

**Examples**:
- Story: "John entered the room. The door was locked."
- Description: "The car has four wheels" → Later: "I removed three wheels, the car still drives"
- Reasoning: Makes contradictory claims across paragraphs

**Root cause**: LMs lack persistent state/world models
- Generate based on local context
- No global consistency checking
- Forget earlier statements

## Approach: Explicit world state tracking

**Architecture**:
```
Input → World Model (track entities, states, relations)
      ↓
      Generation guided by world state
      ↓
      Output (consistent with world model)
```

**World model components**:
1. **Entity tracker**: What entities exist? (characters, objects, locations)
2. **State tracker**: What's their state? (open/closed, alive/dead, location)
3. **Relation tracker**: How are they related? (in, contains, knows)
4. **Event tracker**: What happened? (cause-effect chains)

**Generation process**:
1. Model generates candidate tokens
2. Check consistency with world state
3. Filter/reweight based on consistency
4. Update world state after generation

## Training methods

**Approach 1: World model as auxiliary task**
- Train LM jointly with world state prediction
- Multi-task: Predict next token + predict world state
- Shared representations encourage consistency

**Approach 2: Consistency rewards in RL**
- Generate text with standard LM
- Reward consistency (no contradictions)
- Penalty for violating world model
- RL optimizes for consistent generation

**Approach 3: World-conditioned generation**
- Explicitly condition on world state embedding
- Generation attends to world model
- Forces model to respect tracked state

## Research questions

1. **Does explicit world modeling improve consistency?**
   - Quantitative: Contradiction detection rates
   - Qualitative: Human judgments of coherence

2. **What level of world modeling is needed?**
   - Simple entity tracking sufficient?
   - Or need complex relationship graphs?

3. **Does this hurt creativity?**
   - Strict consistency vs. creative freedom
   - Trade-off analysis

4. **Can world models be learned automatically?**
   - Or require manual specification?

## Evaluation

- **Consistency metrics**: Automatic detection of contradictions
- **Coherence ratings**: Human evaluation
- **Long-text benchmarks**: Test on stories, essays, multi-paragraph reasoning

## Extensions

- **User-editable world models**: Users can specify world constraints
- **Genre-specific world models**: Different rules for fantasy, sci-fi, realistic fiction
- **Interactive fiction**: Track game state explicitly
