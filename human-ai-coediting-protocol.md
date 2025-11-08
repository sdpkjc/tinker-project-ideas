# Human-AI co-editing protocols: Collaborative content creation

Current LLM interaction is generate-then-edit: model produces output, human edits. Real collaboration involves iterative co-creation where both parties contribute. This project develops protocols for human-AI co-editing—fluid, turn-by-turn collaborative content creation with clear roles and communication patterns.

## Vision: Collaborative writing/coding

**Instead of**:
```
Human: "Write a function to..."
AI: [Generates full function]
Human: [Edits function]
```

**Co-editing**:
```
Human: "Let's write a function to sort a list"
AI: "Sure! Let's start with the signature: def sort_list(items: list):"
Human: "Good. Now let's handle the base case"
AI: "if len(items) <= 1: return items"
Human: "Perfect. Now the recursive case..."
[Back-and-forth continues]
```

**Benefits**:
- Higher quality (catch errors early)
- User maintains control
- Learning opportunity (user sees AI's reasoning)
- More engaging

## Protocol design

### Turn-taking rules

**Who goes when**:
- User initiates or requests AI contribution
- AI contributes specific components
- Clear handoff signals: "Your turn" / "What next?"

**Contribution sizes**:
- Small chunks (1-2 sentences, 5-10 lines of code)
- Not full documents/programs
- Allows frequent feedback

### Communication patterns

**Modes of interaction**:
1. **Suggestion mode**: AI proposes, human accepts/rejects
2. **Completion mode**: Human starts, AI continues
3. **Critique mode**: AI critiques human's contribution
4. **Question mode**: AI asks questions to clarify intent

**Explicit signaling**:
- [AI proposes], [Human edits], [AI explains]
- Clear whose turn, what mode

### Conflict resolution

**When disagreement**:
- AI explains reasoning
- Human makes final decision
- AI adapts to human's choice

**Example**:
```
AI: "I suggest using quicksort here"
Human: "I prefer mergesort for stability"
AI: "Got it, using mergesort. That ensures stable sorting."
```

## Training for collaboration

### Reward functions for co-editing

**Reward good collaboration**:
- Small, well-scoped contributions (not monologues)
- Responsive to human edits
- Asks clarifying questions when needed
- Defers to human judgment

**Penalize poor collaboration**:
- Overwriting human contributions
- Ignoring context from previous turns
- Too verbose or too terse

### Data collection

**Collect human-AI co-editing sessions**:
- Hire annotators to co-edit with models
- Record interaction patterns
- Label good/bad collaboration moments

**Train on collaboration data**:
- Supervised learning on good co-editing behaviors
- RL with collaboration rewards

### Multi-agent training

**Train via self-play**:
- Two models co-edit as human and AI
- Learn collaborative dynamics
- Transfer to human-AI setting

## Evaluation

### Metric 1: Collaboration quality

**Measure**:
- Turn appropriateness (right contribution at right time)
- Responsiveness to human input
- Balance (not AI-dominated or human-dominated)

### Metric 2: Output quality

**Final artifact quality**:
- Code correctness, document coherence
- Compare: Co-editing vs. generate-then-edit

### Metric 3: User experience

**User studies**:
- Satisfaction ratings
- Perceived collaboration quality
- Preference: Co-editing vs. traditional

### Metric 4: Efficiency

**Time to completion**:
- Faster or slower than alternatives?
- Quality-time trade-off

## Research questions

1. **Is co-editing better than generate-then-edit?**
   - Quality? Efficiency? User satisfaction?

2. **What collaboration protocols work best?**
   - Turn-taking rules? Communication patterns?

3. **Can models learn to collaborate?**
   - Or requires hand-engineered protocols?

4. **Do users prefer collaborative interaction?**
   - Or find it tedious?

5. **What tasks benefit most from co-editing?**
   - Creative writing? Code? Planning?

## Applications

- **Long-form writing**: Essays, reports, stories
- **Code development**: Pair programming with AI
- **Design**: Iterative refinement of designs
- **Planning**: Collaborative task planning

## Extensions

- **Multi-party collaboration**: Human + multiple AIs
- **Asynchronous co-editing**: Google Docs-style real-time editing
- **Role specialization**: AI takes different roles (editor, critic, supporter)
- **Learning from collaboration**: Model improves through co-editing sessions
