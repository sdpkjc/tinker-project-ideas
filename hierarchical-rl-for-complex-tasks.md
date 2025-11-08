# Hierarchical RL for compositional complex task completion

Standard RLHF optimizes token-by-token generation with a reward given after the full response. For complex multi-step tasks (extended reasoning, multi-file code generation, long-form writing), this approach is inefficient—the model receives sparse feedback only at the end, making credit assignment difficult and learning slow.

This project explores hierarchical reinforcement learning for LLMs: decomposing complex tasks into subtasks with intermediate rewards, enabling efficient learning on challenging compositional tasks, inspired by [hierarchical RL](https://arxiv.org/abs/1604.06057) and [options framework](https://people.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf).

## Motivation

**Problem with flat RL on complex tasks**:

Example: "Implement a full web application with authentication"
- Single reward at end (after 10,000+ tokens)
- Hard to determine which parts contributed to success/failure
- Sparse signal makes learning inefficient
- Model may fail early but only get feedback at the end

**Hierarchical approach**:
Break into subtasks:
1. Design database schema → Intermediate reward
2. Implement authentication → Intermediate reward
3. Create API endpoints → Intermediate reward
4. Build frontend → Intermediate reward
5. Write tests → Intermediate reward

**Benefits**:
- Dense rewards (feedback at each subtask)
- Better credit assignment (know which subtask succeeded/failed)
- Reusable skills (subtasks transferable to other complex tasks)
- Curriculum learning (learn simple subtasks first)

## Hierarchical RL framework for LLMs

### Hierarchy levels

**Level 0: Token generation** (standard LM)
- Primitive action: Generate next token
- Low-level policy: π(token | context)

**Level 1: Subtask completion** (skills/options)
- Mid-level policy: Complete subtask (e.g., "write authentication function")
- Temporal abstraction: Generate tokens until subtask done
- Reward: Subtask success

**Level 2: Task planning** (high-level strategy)
- High-level policy: Decompose task into subtask sequence
- Meta-policy: Select which subtask to tackle next
- Reward: Overall task success

### Two-level hierarchical RL

**High-level policy** (manager):
- Input: Complex task description
- Output: Sequence of subtasks
- Objective: Maximize overall task reward

**Low-level policy** (worker):
- Input: Subtask description + context
- Output: Subtask completion (generate response for subtask)
- Objective: Maximize subtask reward

**Training**:
- Train low-level policy on subtasks (easier learning problem)
- Train high-level policy to select/sequence subtasks (meta-learning)

## Proposed approaches

### Approach 1: Fixed subtask decomposition

**Method**:
Manually define task decomposition into subtasks

**Example** (code generation):
1. Understand requirements → Generate specification
2. Design architecture → Generate design doc
3. Implement core logic → Generate code
4. Add error handling → Generate error handling code
5. Write tests → Generate tests

**Training**:
- Train separate models (or LoRA adapters) for each subtask
- Or train single model with subtask prompts
- Reward each subtask independently

**Benefits**:
- Explicit structure, interpretable
- Can design curriculum (easy subtasks first)
- Reusable subtask models

**Challenges**:
- Manual decomposition required per task type
- May not be optimal decomposition
- Fixed structure (not adaptive)

### Approach 2: Learned task decomposition

**Method**:
Learn to decompose tasks automatically

**High-level planner**:
```
Input: Complex task
Output: [Subtask_1, Subtask_2, ..., Subtask_N]
```

**Training**:
- Supervise with expert decompositions (if available)
- Or use RL to learn decomposition that maximizes task success
- Reward: Overall task completion

**Benefits**:
- Flexible, adapts to task
- Discovers efficient decompositions
- Generalizes to new task types

**Challenges**:
- Hard learning problem (credit assignment for decomposition quality)
- Requires meta-learning

### Approach 3: Options/skills framework

**Method**:
Learn reusable skills (temporally extended actions)

**Skill library**:
- Skill 1: "Implement function given signature"
- Skill 2: "Fix bug given error message"
- Skill 3: "Write docstring for code"
- Skill 4: "Generate test cases"

**Meta-policy**:
- Selects which skill to apply at each step
- Sequences skills to complete complex task

**Training**:
1. Pre-train skills on skill-specific datasets
2. Train meta-policy via RL to compose skills
3. Fine-tune skills end-to-end

**Benefits**:
- Modular, reusable skills
- Transfer across tasks
- Efficient learning (learn skills once, reuse often)

**Inspired by**: Options framework in RL, skill learning in robotics

### Approach 4: Recursive task decomposition

**Method**:
Recursively decompose tasks until atomic subtasks

**Algorithm**:
```
def solve(task):
  if task is atomic:
    generate response directly
  else:
    subtasks = decompose(task)
    results = [solve(subtask) for subtask in subtasks]
    combine(results)
```

**Example** (write research paper):
- Top level: [Introduction, Related Work, Method, Experiments, Conclusion]
- Mid level: Introduction → [Motivation, Problem Statement, Contributions]
- Low level: Motivation → [Why problem important, Why existing solutions insufficient]

**Training**:
- Train decomposer: task → subtasks
- Train combiner: subtask results → combined result
- Train base solver: atomic task → response

**Benefits**:
- Natural hierarchical structure
- Arbitrary depth
- Compositional generalization

### Approach 5: Hindsight relabeling with subgoals

**Method**:
Learn from "hindsight" by relabeling goals

**Standard RL**: Try to complete task, get reward based on success

**Hindsight**: Even if task fails, learn from what was achieved
- Attempted: "Build full web app" (failed)
- Relabel: "Build authentication module" (succeeded)
- Reward achievement of intermediate subgoal

**Training**:
- Attempt complex tasks (may fail)
- Identify what subtasks were successfully completed
- Reward model for those achievements
- Learn incrementally toward full task

**Benefits**:
- Learn from failures
- Dense learning signal
- Automatic curriculum (naturally progresses from easy to hard)

**Inspired by**: [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495)

## Implementation strategies

### Strategy 1: Prompt-based hierarchy

Use prompts to specify hierarchy:

**Example**:
```
High-level: "Break this task into steps: [task]"
Model: "Step 1: ..., Step 2: ..., Step 3: ..."

Low-level: "Complete step 1: [subtask_1]"
Model: [subtask_1 completion]
```

**Benefits**:
- No architecture changes
- Works with existing models
- Flexible

### Strategy 2: Multi-stage generation

Generate response in stages with intermediate rewards:

**Generation process**:
1. Generate outline/plan → Reward_1
2. Generate detailed content → Reward_2
3. Refine and polish → Reward_3

**Training**:
- RL at each stage with stage-specific rewards
- Final reward combines all stages

### Strategy 3: Tree-structured generation

Generate response as tree:
- Root: Overall task
- Branches: Subtasks
- Leaves: Atomic outputs

**Training**:
- Reward at each node (subtask)
- Propagate rewards up tree
- Learn to build successful trees

## Evaluation methodology

### Metric 1: Complex task success rate

**Test tasks**:
- Multi-file code projects
- Long-form structured documents
- Multi-step reasoning problems
- Composite creative projects

**Measure**: Completion rate, quality

**Compare**: Hierarchical RL vs. flat RL

### Metric 2: Sample efficiency

**Question**: Does hierarchical RL learn faster?

**Method**:
- Plot: Training steps vs. task success rate
- Compare learning curves

**Hypothesis**: Hierarchical RL more sample-efficient (dense rewards)

### Metric 3: Subtask reusability

**Test**: Can learned subtasks transfer to new tasks?

**Method**:
- Train on task set A
- Test on task set B (different tasks but overlapping subtasks)
- Measure: Transfer performance

### Metric 4: Credit assignment quality

**Analysis**: Do rewards reach appropriate subtasks?

**Method**:
- Inject known errors in specific subtasks
- Check: Does training fix those subtasks specifically?
- Measure: Targeted improvement

### Metric 5: Decomposition quality

**For learned decomposition**:
- Compare learned decomposition vs. expert decomposition
- Measure: Similarity, efficiency, success rate

## Experimental framework

### Experiment 1: Hierarchical vs. flat RL

**Setup**:
- Select complex tasks requiring multiple steps
- Train with flat RL (single end reward)
- Train with hierarchical RL (intermediate rewards)

**Measure**: Success rate, sample efficiency, final quality

### Experiment 2: Hierarchy depth effects

**Question**: How many hierarchy levels are optimal?

**Test**:
- 1-level (flat)
- 2-level (subtasks)
- 3-level (sub-subtasks)

**Measure**: Performance vs. training cost

### Experiment 3: Fixed vs. learned decomposition

**Compare**:
- Hand-designed task decomposition
- Learned task decomposition

**Question**: Can model learn better decomposition than human-designed?

### Experiment 4: Domain specificity

**Test across domains**:
- Code generation
- Long-form writing
- Multi-step reasoning
- Creative projects

**Question**: Where does hierarchical RL help most?

## Key research questions

1. **Does hierarchical RL improve complex task performance?**
   - How much improvement over flat RL?

2. **What hierarchy structure is optimal?**
   - Depth, breadth, decomposition strategy?

3. **Do learned skills transfer?**
   - Can skills learned on one task apply to others?

4. **Is learned decomposition better than fixed?**
   - Or is human-designed structure sufficient?

5. **What types of tasks benefit most?**
   - When is hierarchy necessary vs. helpful vs. unnecessary?

## Practical considerations

**Computational cost**:
- Hierarchical RL may be more complex
- But potentially more sample-efficient
- Trade-off analysis needed

**Interpretability**:
- Hierarchical structure may be more interpretable
- Can visualize task decomposition
- Easier debugging (identify failing subtasks)

## Extensions

**Continual learning of skills**:
- Continuously add new skills to library
- Never stop learning new capabilities

**User-specified hierarchy**:
- Users can specify task decomposition
- Model follows user's structure

**Adaptive hierarchy**:
- Adjust hierarchy depth based on task difficulty
- Simple tasks: Shallow hierarchy
- Complex tasks: Deep hierarchy

**Multi-agent hierarchy**:
- Different agents for different subtasks
- Specialized experts for each hierarchy level

This approach could unlock LLM capabilities on complex, multi-step tasks that are currently out of reach for standard RLHF, enabling AI systems that can tackle ambitious, structured projects end-to-end.
