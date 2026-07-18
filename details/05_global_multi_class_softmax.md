# Global Multi-Class Softmax Categorical Era

Standard categorical cross-entropy with global Softmax scales neural networks to handle multi-class target domains (e.g. classification, token predictions).

## Concept
Evaluates the complete probability vector over the entire vocabulary or category space.

## Diagram
```mermaid
flowchart TD
    Net[Deep Neural Network] --> Logits[Global Logits Vector]
    Logits --> Softmax[Global Softmax Denominator Sum]
    Softmax --> Loss[Categorical Cross-Entropy]
```

[Back to README](../README.md)
