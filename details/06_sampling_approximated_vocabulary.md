# Sampling-Approximated Vocabulary Shortcut Era

To bypass the expensive $O(|\mathcal{V}|)$ computation of global Softmax, approximate techniques like Noise-Contrastive Estimation (NCE) and Negative Sampling were developed.

## Concept
Instead of normalising over all words, classify the target word against $k$ noise samples.

## Diagram
```mermaid
flowchart TD
    Target[Target Word Logit] --> BinaryClass[Binary Logistic Classifiers]
    Noise[K Noise Logits] --> BinaryClass
    BinaryClass --> Loss[Approximate Cross-Entropy]
```

[Back to README](../README.md)
