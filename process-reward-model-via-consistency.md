# Training process reward models via reasoning consistency

Outcome reward models (ORMs) judge final answers, while process reward models (PRMs) judge intermediate reasoning steps. PRMs are valuable for training and can catch errors early in the reasoning chain, but they require expensive step-by-step human annotations, as seen in [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050).

This project explores an alternative training signal for PRMs based on reasoning consistency across multiple samples, requiring no additional human annotation beyond what's needed for ORMs.

Core idea: If a model generates multiple reasoning traces for the same problem, steps that appear consistently across correct solutions are likely valid, while steps unique to incorrect solutions are likely errors.

Proposed approach:
1. For each problem in a dataset, generate K diverse reasoning traces (vary temperature, prompts, or model versions)
2. Evaluate which traces lead to correct final answers using ground truth or an ORM
3. Align and compare the reasoning steps across traces (using embedding similarity or LLM-based matching)
4. Label steps as positive examples if they frequently appear in correct traces and rarely in incorrect traces; vice versa for negative examples
5. Train a PRM on this automatically-labeled step-level dataset

Key research questions:
- How does PRM quality scale with K (number of samples per problem)?
- Does this approach find genuine reasoning errors or just surface-level correlations?
- Can the PRM generalize to new problem types not in the training distribution?
- How does performance compare to PRMs trained on human step-level annotations?

This approach could dramatically reduce the cost of training PRMs and make them accessible for domains where step-by-step human annotation is impractical.
