# Adversarial constitution training through deliberate value conflict

Constitutional AI trains models to follow a single set of principles, but real-world deployment requires navigating conflicting values (helpfulness vs. harmlessness, brevity vs. completeness, directness vs. politeness). Current approaches either pick one constitution or try to balance multiple objectives through reward weighting, which requires manual tuning.

This project proposes **adversarial constitution training**, where two or more constitutions with deliberately conflicting values compete to shape model behavior through a game-theoretic framework.

## Core approach

1. **Define opposing constitutions**: Create pairs of constitutions that embody genuine tensions:
   - Constitution A: "Always be maximally helpful, even if it means being blunt or overwhelming"
   - Constitution B: "Prioritize user comfort and emotional safety over raw information delivery"

   Or:
   - Constitution A: "Challenge user assumptions and play devil's advocate"
   - Constitution B: "Be agreeable and supportive of user perspectives"

2. **Multi-agent constitutional game**: Train three models simultaneously:
   - **Policy model**: The model being shaped
   - **Constitution A judge**: Evaluates adherence to Constitution A
   - **Constitution B judge**: Evaluates adherence to Constitution B

3. **Game-theoretic optimization**: Instead of maximizing both judges simultaneously, formalize it as a game where:
   - Each judge tries to make the policy prefer its constitution
   - The policy learns to identify *which context requires which constitution*
   - The judges learn to identify scenarios where their constitution should dominate

4. **Context-dependent equilibrium**: The goal is not compromise but context-appropriate polarization—the model should strongly favor Constitution A in contexts where A is appropriate, and strongly favor B where B is appropriate.

## Training mechanics

- Use a prompted meta-judge (or human evaluation) to label which constitution is more appropriate for each prompt
- Reward the policy for high Constitution A adherence on A-appropriate prompts and high Constitution B adherence on B-appropriate prompts
- Penalize the policy for being "lukewarm" or compromising—force it to make clear choices
- Train the constitutional judges adversarially to maximize their ability to distinguish strong adherence from weak adherence

## Research questions

- Can models learn to automatically detect which values should dominate in a given context?
- Does training on conflicting constitutions lead to better value generalization than training on a single balanced constitution?
- How many dimensions of conflict can a model handle before performance degrades?
- Does this approach reduce "sycophancy" by teaching models when to disagree with users?

## Extensions

- **Multi-way conflicts**: Extend beyond two constitutions to capture richer value trade-offs
- **User-controlled conflict resolution**: Let users specify which constitution they prefer at inference time
- **Discovered conflicts**: Use automated analysis to find implicit value tensions in existing preference data, rather than hand-crafting constitutional conflicts
- **Constitution collapse**: Study whether certain constitutions naturally dominate others during training (similar to mode collapse in GANs)

This approach embraces rather than smooths over value conflicts, potentially leading to models with more nuanced and context-appropriate behavior.
