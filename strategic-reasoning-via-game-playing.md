# Training strategic reasoning through competitive game-playing

Current LLM training focuses on single-turn accuracy, but strategic reasoning—planning ahead, anticipating opponents, optimizing under uncertainty—requires different skills. Competitive game-playing has driven major AI advances (Chess, Go, StarCraft). This project trains LLMs for strategic reasoning through game-playing environments.

## Motivation

**Strategic reasoning gaps in LLMs**:
- Poor at multi-step planning
- Don't anticipate consequences
- Struggle with adversarial scenarios
- Lack game-theoretic reasoning

**Game-playing as training ground**:
- Clear objectives
- Adversarial dynamics
- Require planning ahead
- Rich feedback (win/loss)

**Games to use**:
- **Simple**: Tic-tac-toe, Connect Four
- **Medium**: Chess, Poker (LLM plays via text)
- **Complex**: Negotiation games, Diplomacy
- **Open-ended**: Text-based strategy games

## Training approach

### Self-play RL

**Method**:
1. Model plays against itself
2. Learn from wins/losses
3. Improve strategy iteratively

**Benefits**: AlphaGo-style emergent strategy

### Multi-agent competition

**Method**:
1. Population of models
2. Models play against each other
3. Selection pressure for strategic ability

**Benefits**: Diversity, robust strategies

### Curriculum of games

**Method**:
1. Start with simple games (learn basics)
2. Progress to complex games (strategic depth)
3. Transfer learned reasoning to non-game tasks

### Reward shaping for reasoning

**Rewards**:
- Win/loss (sparse)
- Intermediate: Good moves even if game not won
- Strategy metrics: Board control, resource advantage

**Goal**: Learn strategic reasoning, not just winning specific games

## Transfer to general reasoning

**Hypothesis**: Strategic reasoning transfers to:
- Planning tasks
- Multi-step problem-solving
- Adversarial robustness
- Negotiation and persuasion

**Evaluation**:
- Train on games
- Test on non-game strategic reasoning tasks
- Measure: Does game-playing improve general reasoning?

## Research questions

1. **Does game-playing improve strategic reasoning?**
   - Transfer to general tasks?

2. **What games are most effective?**
   - Simple games sufficient?
   - Or need complex games?

3. **How much game training is needed?**
   - Data efficiency?

4. **Does diversity matter?**
   - Train on multiple game types?

5. **Can we explain emergent strategies?**
   - Interpret what model learns?

## Evaluation

- **Game performance**: ELO ratings
- **Transfer tasks**: Planning benchmarks, adversarial reasoning
- **Strategy quality**: Expert evaluation of game-play
- **Generalization**: Novel games not in training

## Extensions

- **Human vs. AI**: Models play against humans
- **Cooperative games**: Multi-agent cooperation
- **Open-ended games**: Minecraft-style environments
- **Meta-gaming**: Learn to learn new games quickly
