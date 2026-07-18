# Label Smoothing Regularization

Label smoothing regularizes networks against overconfidence by distributing a small probability mass uniformly across all classes.

## Mathematical Formulation
$$y_i^{\text{smoothed}} = y_i(1 - \epsilon) + \frac{\epsilon}{K}$$

## Diagram
```mermaid
flowchart LR
    OneHot[One-Hot Targets] --> Smooth[Label Smoothing Kernel]
    Smooth --> Dist[Soft targets]
    Dist --> CrossEntropy[Cross-Entropy Loss]
```

[Back to README](../README.md)
