# Lesson 0.6 — GPU Memory Bandwidth

## 1. Definition

**Memory Bandwidth** = Amount of data that can be transferred between GPU memory and the GPU per second.

Measured in:

```text
GB/s
TB/s
```

Example:

```text
Memory Bandwidth = 1 TB/s
```

means the memory system can theoretically transfer about **1 TB of data per second**.

---

## 2. Compute vs Memory

```text
              GPU
               │
       ┌───────┴────────┐
       │                │
    Compute           Memory
    FLOPS             Bandwidth
       │                │
       └───────┬────────┘
               │
         Actual Performance
```

High compute capability is not enough if the GPU cannot supply data fast enough.

---

## 3. Simple Example

Suppose:

```text
Data = 1 TB
Memory Bandwidth = 1 TB/s
```

The theoretical minimum transfer time is approximately:

```text
1 TB
────── = 1 second
1 TB/s
```

Actual execution will usually take longer because of other overheads.

---

## 4. Memory-Bound Workload

A workload is **memory-bound** when data movement is the primary bottleneck.

```text
Memory
   ↓
Load
   ↓
Small computation
   ↓
Store
   ↓
Memory
```

Examples:

* Element-wise operations
* Some reductions
* Some normalization operations
* Some LLM decode workloads

---

## 5. Compute-Bound Workload

A workload is **compute-bound** when arithmetic computation is the primary bottleneck.

```text
Load data
   ↓
Heavy computation
   ↓
Heavy computation
   ↓
Store result
```

Large matrix multiplications are often compute-intensive.

---

## 6. Memory Capacity vs Memory Bandwidth

These are different.

### Memory Capacity

How much data can fit in GPU memory.

```text
80 GB VRAM
```

### Memory Bandwidth

How quickly data can move.

```text
3 TB/s
```

Think:

```text
Capacity  → How much can I store?

Bandwidth → How fast can I move it?
```

---

## 7. Why It Matters for LLMs

LLM inference needs to access:

```text
Model Weights
KV Cache
Activations
Intermediate Data
```

During decode:

```text
Read weights / KV cache
        ↓
      Compute
        ↓
    Next Token
```

Memory bandwidth can become a major bottleneck, especially for some decode workloads.

---

## 8. Connection to Arithmetic Intensity

$$
Arithmetic\ Intensity =
\frac{FLOPs}{Bytes\ Transferred}
$$

### Low Arithmetic Intensity

```text
More memory traffic
       ↓
Memory-bound
```

### High Arithmetic Intensity

```text
More computation per byte
       ↓
Potentially compute-bound
```

---

## 9. Key Takeaways

1. Memory bandwidth measures **data transfer rate**.
2. Memory capacity measures **how much data can be stored**.
3. High TFLOPS does not guarantee high performance.
4. A GPU can be limited by memory bandwidth.
5. Memory-bound workloads spend significant time moving data.
6. Compute-bound workloads spend significant time performing calculations.
7. LLM inference can be heavily affected by memory bandwidth.
8. Arithmetic intensity helps determine whether a workload is likely to be compute-bound or memory-bound.

### Mental Model

```text
GPU Performance
      │
      ├── Compute → FLOPS
      │
      ├── Memory → Bandwidth
      │
      └── Memory → Capacity
```

> **The GPU must have enough compute capability AND enough memory bandwidth to keep its compute resources efficiently utilized.**
