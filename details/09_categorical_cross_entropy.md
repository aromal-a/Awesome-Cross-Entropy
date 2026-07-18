# Categorical Cross-Entropy (Softmax Loss)

Categorical Cross-Entropy is applied when targets are mutually exclusive classes.

## Concept
It pushes the model to maximize the probability of the correct class while minimizing the rest.

## Diagram
```mermaid
flowchart TD
    Softmax[Softmax Layer] --> TargetProb[Correct Class Probability]
    TargetProb --> NegLog[Negative Log-Likelihood]
    NegLog --> Loss[Final Loss Scalar]
```

[Back to README](../README.md)
