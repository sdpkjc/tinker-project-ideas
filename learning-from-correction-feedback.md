# Learning from correction feedback: Iterative response refinement

Standard RLHF learns from pairwise preferences: given two responses, which is better? However, real user feedback is often richer—users provide corrections, suggestions, and iterative refinement rather than binary comparisons. This correction feedback is more informative but underutilized in current RLHF methods.

This project explores training methods that leverage correction feedback: users edit/refine model outputs, and the model learns directly from these edits, providing a more efficient and natural learning signal than binary preferences.

## Motivation

**Standard preference feedback**:
```
User: "Explain photosynthesis"
Model: [Response A], [Response B]
User: "A is better"
```

**Limitation**: Binary signal, doesn't explain why or how to improve

**Correction feedback**:
```
User: "Explain photosynthesis"
Model: [Response]
User: *Edits response* "Actually, chlorophyll absorbs red and blue light, not green..."
```

**Benefit**: Directly shows what to change, more informative per interaction

**Analogy**: Learning from corrections is like supervised learning with demonstrations, while preference learning is like learning from comparisons alone

## Types of correction feedback

### Type 1: Direct edits
User modifies model output:
- Add missing information
- Fix factual errors
- Improve clarity/style
- Remove unnecessary content

### Type 2: Verbal corrections
User provides textual feedback:
- "This is wrong because..."
- "You should mention..."
- "Too verbose, be more concise"

### Type 3: Iterative refinement
Multi-turn improvement:
- User requests modifications
- Model generates revised version
- Repeat until satisfactory

### Type 4: Accept/reject with explanation
User accepts parts and rejects parts:
- "This part is good: [...], but this part is wrong: [...]"
- Selective feedback on different response components

## Proposed training approaches

### Approach 1: Correction-to-preference conversion

Convert correction feedback into preference pairs:

**Method**:
1. User corrects response A → produces A'
2. Create preference: A' > A
3. Train reward model on (A', A) preferences
4. Use RM for standard RLHF

**Benefits**:
- Compatible with existing RLHF pipeline
- Simple conversion
- Leverages existing infrastructure

**Limitations**:
- Discards rich edit information (only uses final corrected version)
- Doesn't learn what specifically to change

### Approach 2: Edit-based supervised learning

Train model to generate corrected outputs directly:

**Method**:
1. Collect (prompt, initial_response, corrected_response) tuples
2. Fine-tune model to generate corrected versions:
   ```
   Input: [prompt] + [initial_response]
   Output: [corrected_response]
   ```
3. Model learns transformation from error → correction

**Benefits**:
- Directly learns from corrections
- Captures specific edit patterns
- Efficient learning signal

**Challenges**:
- Requires correction data (more expensive than preferences)
- May not generalize to novel error types

### Approach 3: Diff-based learning

Learn from the "diff" between initial and corrected responses:

**Method**:
1. Compute edit distance / diff:
   - Deletions: What to remove
   - Insertions: What to add
   - Substitutions: What to replace

2. Train model to predict edits:
   ```
   Input: [prompt] + [initial_response]
   Output: [edit_operations]
   ```

3. Apply learned edits to improve responses

**Benefits**:
- Fine-grained learning (token/phrase level)
- Efficient representation (only changes, not full text)
- Interpretable (can see what model learned to fix)

**Inspired by**: Code review systems, text editing research

### Approach 4: Reward modeling from corrections

Train reward model to score based on correction quality:

**Method**:
1. Measure "correction distance": How much did user need to edit?
2. Responses requiring fewer corrections → Higher reward
3. Train RM: R(response) ∝ -correction_distance
4. Use for RLHF

**Benefits**:
- Quantifies quality through corrections
- Can combine with preference data
- Natural metric (less editing = better quality)

### Approach 5: Iterative self-correction

Train model to iteratively improve its own outputs:

**Method**:
1. Model generates initial response
2. Model generates critique of its response
3. Model generates improved version based on critique
4. Train on human corrections as supervision for this process

**Benefits**:
- Model learns to self-improve
- Applicable at inference (no human needed)
- Captures iterative refinement process

**Inspired by**: [Self-Refine](https://arxiv.org/abs/2303.17651), Constitutional AI

### Approach 6: Contrastive correction learning

Learn by contrasting original and corrected responses:

**Method**:
1. Embed original response → e_original
2. Embed corrected response → e_corrected
3. Train: Model should generate responses closer to e_corrected
4. Contrastive loss: Pull toward corrected, push from original

**Benefits**:
- Learns direction of improvement in latent space
- Generalizes beyond specific edits
- Representation learning approach

## Data collection strategies

### Strategy 1: Crowdsourced corrections

**Method**:
- Generate responses from current policy
- Ask crowd workers to correct/improve them
- Collect (original, corrected) pairs

**Challenges**:
- More expensive than preference collection
- Quality control (are corrections actually better?)
- Annotator expertise (corrections may not be correct)

### Strategy 2: Expert corrections

**Method**:
- In specialized domains (medical, legal, technical), get expert corrections
- Higher quality but more expensive

**Use case**: High-stakes domains where accuracy critical

### Strategy 3: User corrections in deployment

**Method**:
- Allow deployed users to edit model outputs
- "Improve this response" feature
- Collect real corrections from actual users

**Benefits**:
- Real user needs
- Continuous improvement
- Authentic feedback

**Privacy**: Need user consent for data collection

### Strategy 4: Synthetic corrections

**Method**:
- Use strong model to generate corrections of weak model outputs
- Or use model to inject errors, then correct them

**Benefits**:
- Scalable, cheap
- No human annotation needed

**Challenges**:
- Quality of synthetic corrections
- May not reflect real user needs

### Strategy 5: Semi-supervised learning

**Method**:
- Small seed set of human corrections
- Use model to generate pseudo-corrections for unlabeled data
- Train on mixture of real and pseudo corrections

**Benefits**:
- Leverages unlabeled data
- Reduces annotation cost

## Evaluation methodology

### Metric 1: Correction quality

**Test**:
- Generate responses from trained model
- Have humans correct them
- Measure: How much correction needed?

**Metric**: Edit distance, time to correct, annotator satisfaction

**Goal**: Trained model requires fewer corrections

### Metric 2: Specific error reduction

**Test**:
- Identify common error types in corrections (factual errors, verbosity, etc.)
- After training on corrections, measure: Error rates for each type

**Goal**: Targeted improvement on corrected error types

### Metric 3: Preference win rate

**Comparison**:
- Model trained on corrections vs. model trained on preferences
- Head-to-head human evaluation

**Question**: Does correction-based training produce better outputs?

### Metric 4: Sample efficiency

**Question**: Are corrections more efficient than preferences?

**Method**:
- Plot: Number of training examples vs. performance
- Compare: Corrections vs. preferences
- Measure: How many corrections = how many preferences in terms of final quality?

### Metric 5: Generalization to novel errors

**Test**:
- Train on corrections for error types {A, B, C}
- Test on error type D (not in training)

**Question**: Does model learn general correction ability, or just specific fixes?

## Experimental framework

### Experiment 1: Correction vs. preference learning

**Setup**:
- Collect both corrections and preferences on same prompts
- Train separate models with each feedback type
- Control for annotation cost

**Measure**: Final model quality, sample efficiency

### Experiment 2: Correction type effectiveness

**Compare**:
- Direct edits
- Verbal corrections
- Iterative refinement
- Accept/reject with explanation

**Question**: Which type of correction most useful for learning?

### Experiment 3: Hybrid correction + preference

**Test**:
- Train on corrections only
- Train on preferences only
- Train on combination

**Question**: Are they complementary?

### Experiment 4: Domain-specific corrections

**Test**: In specialized domains (code, math, writing)
- Collect domain-specific corrections
- Measure domain-specific improvement

**Question**: Do corrections help more in some domains?

## Key research questions

1. **Are corrections more efficient than preferences?**
   - How many corrections = how many preferences?
   - Cost-benefit analysis

2. **What training method best uses corrections?**
   - Direct SFT, diff-based, reward modeling, or other?

3. **Do corrections enable better generalization?**
   - More specific feedback → Better learning?

4. **Can we collect corrections at scale?**
   - Feasibility and cost compared to preferences

5. **Do users prefer providing corrections?**
   - User experience: Easier to correct or compare?

## Practical considerations

**Interface design**:
- Easy editing tools for corrections
- Suggest where corrections might be needed
- Track edit patterns for analysis

**Quality control**:
- Verify corrections are actually improvements
- Filter low-quality corrections
- Multiple annotators for important corrections

**Privacy**:
- User corrections may contain sensitive info
- Need consent and data handling policies

## Extensions

**Active correction learning**:
- Identify responses most beneficial to correct
- Request corrections strategically (like active learning)

**Correction-guided generation**:
- Use past corrections to guide future generation
- Model learns common correction patterns

**Collaborative correction**:
- Multiple users refine same response
- Aggregate corrections for better training signal

**Real-time correction loop**:
- User corrects → Model updates immediately (continual learning)
- Personalized improvement

**Correction explanation**:
- Ask users why they made corrections
- Learn not just what to change but why

This approach could make RLHF more efficient and natural by aligning training with how users actually provide feedback in practice—through edits and refinements rather than binary comparisons.
