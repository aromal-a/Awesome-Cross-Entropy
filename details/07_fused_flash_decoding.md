# Fused Flash-Decoding Online Sequence Era

Modern training clusters use register-fused execution kernels to perform online softmax and cross-entropy computations directly within GPU SRAM.

## Concept
Bypasses the write-back of large intermediate logit tensors to High Bandwidth Memory (HBM).

## Diagram
```mermaid
flowchart LR
    SRAM[GPU SRAM Registers] -->|Fused Kernel| Softmax[Online Softmax]
    Softmax -->|Fused Causal Mask| Loss[Fused Cross-Entropy]
    Loss -->|Minimal Footprint| Output[Gradients & Loss]
```

[Back to README](../README.md)
