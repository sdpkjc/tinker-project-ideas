# Grounding LLM outputs through verifiable tool use

LLMs hallucinate because they lack grounding—connection to external reality. Tool-augmented LMs ([Toolformer](https://arxiv.org/abs/2302.04761), [ReAct](https://arxiv.org/abs/2210.03629)) can call external tools (search, calculator), providing grounding. This project develops training methods that encourage models to use tools for verification, reducing hallucinations through checkable actions.

## Problem: Ungrounded generation

**Current**:
```
User: "What is 17 × 23?"
Model: "391" (wrong, hallucinated)
```

**With tool grounding**:
```
User: "What is 17 × 23?"
Model: [Calls calculator(17, 23)] → "391"
Model: "17 × 23 = 391"
```

Tool provides ground truth, prevents hallucination

## Tool-use training

### Approach 1: Reward tool use that improves accuracy

**RL objective**:
```
R = Accuracy + β·(Tool_use_helped)
```

**Encourage**:
- Using tools when uncertain
- Trusting tool outputs
- Integrating tool results correctly

**Discourage**:
- Ignoring tool outputs
- Unnecessary tool calls
- Wrong tool for task

### Approach 2: Supervised tool-use demonstrations

**Data collection**:
- Expert demonstrations of tool-augmented problem-solving
- Annotate: When to call tool, which tool, how to use result

**Training**:
- Supervised learning on tool-use trajectories
- Learn policy: when/how to use tools

### Approach 3: Tool-use-as-curriculum

**Progressive training**:
1. **Phase 1**: Simple tasks, no tools needed
2. **Phase 2**: Medium tasks, tools available but optional
3. **Phase 3**: Hard tasks, tools necessary

**Goal**: Learn to recognize when tools are needed

### Approach 4: Verification loops

**Training process**:
1. Model generates answer
2. Model generates verification plan (which tools to use)
3. Execute verification
4. If verification fails, regenerate
5. Train on successful verification loops

## Tool types

**Deterministic tools** (provide ground truth):
- Calculator (arithmetic)
- Code execution (logical correctness)
- Database queries (factual lookup)
- API calls (external data)

**Probabilistic tools** (reduce uncertainty):
- Search engines (information retrieval)
- Fact-checking APIs (claim verification)
- Simulation (outcome prediction)

**Verification tools** (check consistency):
- Grammar checkers
- Logic provers
- Unit tests

## Evaluation

### Metric 1: Hallucination rate

**Measure**: Factual accuracy on verifiable claims
- With vs. without tool access
- Expected: Lower hallucinations with tools

### Metric 2: Appropriate tool use

**Measure**:
- Precision: Tool use when needed
- Recall: Recognition of when tools help
- Efficiency: Not overusing tools

### Metric 3: Trust calibration

**Measure**: Does model trust tool outputs?
- Or ignore/overwrite them?

### Metric 4: Verification effectiveness

**Measure**: Rate of catching errors through tool verification

## Research questions

1. **Does tool grounding reduce hallucinations?**
   - How much improvement?

2. **Can models learn when to use tools?**
   - Or overuse/underuse?

3. **Do models trust tool outputs?**
   - Or exhibit confirmation bias?

4. **What tools are most valuable?**
   - Calculator? Search? Code execution?

5. **Trade-offs**: Latency, cost, accuracy?

## Extensions

- **Tool learning**: Models learn to use new tools zero-shot
- **Tool creation**: Models create custom tools for tasks
- **Multi-tool reasoning**: Combine multiple tools for complex verification
- **Adversarial tool use**: Robust to incorrect tool outputs
- **Private tools**: Tool use without leaking sensitive data
