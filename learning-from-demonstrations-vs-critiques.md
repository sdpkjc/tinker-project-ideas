# Comparing learning from demonstrations vs. critiques: Which feedback type is more efficient?

RLHF learns from critiques (A is better than B), while supervised learning learns from demonstrations (here's a good example). Both have trade-offs. This empirical study systematically compares learning efficiency, quality, and failure modes of different feedback types to guide data collection strategies.

## Feedback types

### Type 1: Demonstrations
```
Input: Prompt
Output: High-quality response (gold standard)
```
Standard supervised learning

### Type 2: Pairwise preferences
```
Input: Prompt, Response A, Response B
Output: A > B (or B > A or tie)
```
Standard RLHF

### Type 3: Critiques
```
Input: Prompt, Response
Output: "This is good because... but could improve by..."
```
Verbal feedback

### Type 4: Rankings
```
Input: Prompt, [Response1, Response2, ..., ResponseN]
Output: Ranking (best to worst)
```
More informative than pairs

### Type 5: Scores
```
Input: Prompt, Response
Output: Quality score (1-10)
```
Cardinal vs. ordinal

### Type 6: Edits
```
Input: Prompt, Response (imperfect)
Output: Edited response (improved)
```
Correctional feedback

## Research questions

1. **Sample efficiency**: Which feedback type learns fastest?
   - 100 demonstrations vs. 1000 preferences?

2. **Quality ceiling**: Which achieves best final performance?

3. **Robustness**: Which is most robust to noise?

4. **Cost-effectiveness**: Quality per annotation dollar?

5. **Task dependence**: Does optimal feedback type vary by task?

6. **Combination**: Are feedback types complementary?

## Experimental framework

### Setup

**Fixed task**: Instruction following

**Data budget**: $10,000 annotation budget

**Allocations**:
- Demonstrations only
- Preferences only
- Critiques only
- Rankings only
- Edits only
- Optimal combination (learned)

**Train**: Models with each allocation

**Evaluate**: Final model quality

### Measurements

**Sample efficiency**:
- Plot: Training examples vs. performance
- Find: How many examples needed to reach target quality?

**Cost efficiency**:
- Plot: Annotation cost vs. performance
- Account for: Different feedback types cost differently

**Quality ceiling**:
- Given unlimited data of each type, which achieves best performance?

**Failure modes**:
- What errors does each feedback type make?
- Characterize systematic failures

## Expected insights

**Hypotheses** (to be tested):
- Demonstrations: Data-efficient for capability, but expensive to collect
- Preferences: Cheap to collect, but less informative per example
- Critiques: Rich information, but annotation consistency issues
- Rankings: More efficient than pairwise preferences
- Edits: Best for targeted improvement, but expensive

## Practical recommendations

**Output**: Decision tree for data collection

```
If budget < $X: Collect preferences
If need high quality: Collect demonstrations
If have capable base model: Collect preferences + few demonstrations
If need robustness: Collect mix of feedback types
```

## Extensions

- **Active learning**: Adaptively choose feedback type per example
- **Hybrid training**: Methods that use multiple feedback types jointly
- **Sequential collection**: Collect one type, then another (curriculum)
- **Quality prediction**: Predict which feedback type would be most valuable
