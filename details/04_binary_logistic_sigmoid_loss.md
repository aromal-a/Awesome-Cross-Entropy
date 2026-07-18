# Binary Logistic & Sigmoid Loss Era

The Binary Logistic & Sigmoid loss was the cornerstone of early statistical classification methods and single-axis class boundary models.

## Concept
Using the sigmoid function to squeeze a single output logit into a binary probability value:
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

## Diagram
```mermaid
flowchart LR
    Input[Input Feature Vector] --> Model[Linear Classifier z]
    Model --> Sigmoid[Sigmoid Activation]
    Sigmoid --> Prob[Binary Probability]
```

[Back to README](../README.md)
