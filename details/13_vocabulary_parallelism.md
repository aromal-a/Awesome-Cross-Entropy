# The Trillion-Token Vocabulary Memory Wall

Vocabulary Parallelism shards the final linear layers and loss calculation across multiple GPUs to reduce peak memory consumption.

## Concept
Split the massive vocabulary projection matrix and calculate partial softmax denominators across GPU partitions.

## Diagram
```mermaid
flowchart TD
    GPU1[GPU 1: Part of Vocab] -->|Partial Exp Sum| AllReduce[All-Reduce Collective]
    GPU2[GPU 2: Part of Vocab] -->|Partial Exp Sum| AllReduce
    AllReduce --> Loss[Global Cross-Entropy Loss]
```

[Back to README](../README.md)
