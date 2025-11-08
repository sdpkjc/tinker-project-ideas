# Task-agnostic capability probing: Automated discovery of model abilities

Current model evaluation relies on pre-defined benchmarks (MMLU, HumanEval), but these only test known capabilities. Models may have latent abilities that exist but aren't tested. This project develops automated probing methods to discover model capabilities without pre-specifying tasks, inspired by open-ended learning in game AI ([Open-Ended Learning Leads to Generally Capable Agents](https://arxiv.org/abs/2107.12808)).

## Motivation

**Problem with fixed benchmarks**:
- Only test what we think to test
- Miss unexpected capabilities
- Benchmarks become stale
- Models may game specific benchmarks

**Desired**: Automatic capability discovery
- Systematically explore what model can do
- Find strengths and weaknesses automatically
- Discover surprising abilities

## Proposed approaches

### Approach 1: Compositional task generation and testing

**Method**:
1. Define primitive capabilities: {reasoning, math, code, language, knowledge}
2. Generate composite tasks combining primitives
3. Test model on generated tasks
4. Map: Which combinations does model handle?

**Example tasks**:
- "Math + code": Implement numerical algorithm
- "Reasoning + knowledge": Answer multi-hop questions
- "Language + creativity": Write metaphorical poetry

**Output**: Capability map showing performance on all combinations

### Approach 2: Adversarial task search

**Method**:
1. Generator creates tasks designed to reveal model weaknesses
2. Test model on generated tasks
3. Identify failure modes
4. Generator creates similar tasks to explore boundaries

**Benefits**: Automatically finds edge cases and limitations

### Approach 3: Skill tree discovery

**Method**:
1. Start with simple tasks (model solves easily)
2. Gradually increase difficulty until model fails
3. Find boundary between can/cannot do
4. Build skill tree: What skills are prerequisites for others?

**Output**: Hierarchical map of capabilities

### Approach 4: Open-ended task environment

**Method**:
1. Define rich, open-ended environment (e.g., text-based world)
2. Model explores and attempts various tasks
3. Track: What tasks does model discover and solve?
4. Measure: Coverage of task space

**Inspired by**: Open-ended RL research

## Evaluation

- **Coverage**: How much of capability space is explored?
- **Discovery**: What unexpected capabilities are found?
- **Comparison**: How does this compare to fixed benchmarks?

## Applications

- **Model selection**: Find model best suited for your needs
- **Training guidance**: Identify weak areas to target
- **Capability documentation**: Comprehensive capability catalog
