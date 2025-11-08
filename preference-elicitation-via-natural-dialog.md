# Preference elicitation through natural dialog: Conversational feedback collection

Standard preference collection shows users two responses and asks "which is better?"—an unnatural interaction. Real users express preferences through natural language: "I liked this part but that part was wrong" or "Too verbose, be more concise." This project develops conversational interfaces for preference elicitation that collect richer feedback through natural dialog.

## Current problems with preference collection

**Binary comparison interface**:
```
Response A: [text]
Response B: [text]
Question: "Which is better?"
Choices: [A] [B] [Tie]
```

**Limitations**:
- Unnatural (not how people give feedback)
- Limited information (just A vs B)
- Cognitively demanding (compare full responses)
- No explanation of why
- Forced choice (even if both bad or both good)

## Proposed: Conversational preference elicitation

**Natural dialog interface**:
```
System: [Shows response]
User: "This is mostly good, but the second paragraph is confusing"
System: "What would make it clearer?"
User: "It should explain X before discussing Y"
System: "Got it. Was the first paragraph helpful?"
User: "Yes, very clear and concise"
```

**Benefits**:
- Natural interaction (how users actually give feedback)
- Richer information (specific strengths/weaknesses)
- Explanations (why preferred)
- Lower cognitive load (discuss one response at a time)
- Nuanced feedback (not just binary)

## System design

### Component 1: Dialog policy for elicitation

Train policy to ask good questions:

**Questions types**:
- "What did you think of [response]?"
- "Was [specific part] helpful?"
- "How could this be improved?"
- "Which part was best/worst?"
- "Would you prefer more/less detail?"

**Goal**: Maximize information gain per question
- Active learning: Ask about uncertain aspects
- Conversational: Natural follow-up questions

### Component 2: Feedback interpretation

Parse natural language feedback into structured preferences:

**Input**: Free-form user comments
**Output**: Structured labels
- Overall rating
- Dimension scores (accuracy, helpfulness, clarity, etc.)
- Specific strengths/weaknesses
- Suggested improvements

**Method**:
- Train classifier to extract structured info from dialog
- Or use LLM to parse feedback

### Component 3: Preference aggregation

Combine information from dialog into training signal:

**Aggregate across**:
- Multiple turns in conversation
- Multiple users
- Different question types

**Output**: Preference dataset for reward model training

## Training and evaluation

**Data collection**:
- Deploy conversational interface
- Collect dialogs about model outputs
- Compare to standard binary preferences

**Evaluation**:
- Information per annotation minute (efficiency)
- User satisfaction (easier to provide feedback?)
- Reward model quality (richer feedback → better RM?)

## Research questions

1. **Is conversational elicitation more efficient?**
   - More information per user interaction?

2. **Do users prefer natural dialog?**
   - Better experience than forced comparisons?

3. **Does richer feedback improve RMs?**
   - Better final model quality?

4. **What questions are most informative?**
   - Optimal dialog strategy?

5. **Can we automate dialog policy?**
   - Learn to ask good questions?

## Extensions

- **Multi-modal feedback**: Voice, pointing, editing
- **Personalized dialogs**: Adapt questions to user expertise
- **Iterative refinement**: User guides model improvements through conversation
- **Explanation elicitation**: "Why did you prefer this?"
