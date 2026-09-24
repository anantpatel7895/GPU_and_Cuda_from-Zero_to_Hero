# Phase 00 — GPU Fundamentals

# Lessons 0.9–0.13 — Quick Notes + Practical GPU Examples

---

# Lesson 0.9 — SIMD vs SIMT

## SIMD

**SIMD = Single Instruction, Multiple Data**

One instruction operates on multiple data elements.

```text
Instruction
    │
    ├── Data 1
    ├── Data 2
    ├── Data 3
    └── Data 4
```

Common CPU examples:

```text
SSE
AVX
AVX2
AVX-512
```

---

## SIMT

**SIMT = Single Instruction, Multiple Threads**

Used by NVIDIA GPUs.

```text
Instruction
    │
    ├── Thread 0
    ├── Thread 1
    ├── Thread 2
    ├── Thread 3
    └── ...
```

NVIDIA groups threads into **warps**.

```text
1 Warp = 32 Threads
```

---

## SIMD vs SIMT

| SIMD                       | SIMT                             |
| -------------------------- | -------------------------------- |
| CPU-oriented model         | GPU-oriented model               |
| Vector lanes               | Threads                          |
| Vector instruction         | Thread execution model           |
| AVX/AVX-512                | CUDA                             |
| Good for CPU vectorization | Good for massive GPU parallelism |

### Key Takeaway

> SIMD operates on multiple data elements using vector instructions; SIMT executes many threads using the GPU execution model.

---

# Lesson 0.10 — GPU Memory Basics

GPU memory is hierarchical.

```text
GPU
│
├── Registers
│
├── Shared Memory / L1
│
├── L2 Cache
│
└── Global Memory / VRAM
```

| Memory        | Speed        | Capacity   | Scope  |
| ------------- | ------------ | ---------- | ------ |
| Registers     | Very fast    | Very small | Thread |
| Shared Memory | Very fast    | Small      | Block  |
| L1 Cache      | Fast         | Small      | SM     |
| L2 Cache      | Slower       | Larger     | GPU    |
| Global Memory | High latency | Large      | GPU    |

---

## Practical Example — Matrix Multiplication

Suppose:

```text
A = 4096 × 4096
B = 4096 × 4096
```

A naïve implementation repeatedly reads data from global memory.

A better CUDA kernel can use:

```text
Global Memory
      ↓
Shared Memory
      ↓
CUDA Threads
      ↓
Compute
```

The same matrix values can be reused by multiple threads.

This reduces expensive global-memory traffic.

---

## Practical Example — LLM Inference

GPU memory may contain:

```text
┌──────────────────────┐
│ Model Weights        │
├──────────────────────┤
│ KV Cache             │
├──────────────────────┤
│ Activations          │
├──────────────────────┤
│ Temporary Buffers    │
└──────────────────────┘
```

If the model requires more memory than the GPU has:

```text
Model
  ↓
GPU Memory
  ↓
OUT OF MEMORY
```

Possible solutions:

* Quantization
* Tensor parallelism
* Pipeline parallelism
* CPU offloading
* Smaller batch size
* Smaller context
* KV-cache optimization

---

# Lesson 0.11 — NVIDIA GPU Overview

Simplified NVIDIA GPU:

```text
                 GPU
                  │
      ┌───────────┼───────────┐
      │           │           │
     SM          SM          SM
      │           │           │
   Threads     Threads     Threads
```

An SM contains resources such as:

* CUDA Cores
* Tensor Cores
* Warp Schedulers
* Registers
* Shared Memory
* L1 Cache

---

## Practical Example — CUDA Kernel

Suppose we want:

```python
C[i] = A[i] + B[i]
```

Conceptually:

```text
GPU
│
├── SM 0
│    ├── Thread 0 → C[0]
│    ├── Thread 1 → C[1]
│    └── ...
│
├── SM 1
│    ├── Thread N → C[N]
│    └── ...
│
└── SM N
```

Millions of elements can be distributed across GPU threads.

---

# Lesson 0.12 — GPU Execution Lifecycle

CUDA execution:

```text
CPU
 │
 │ Launch Kernel
 ▼
CUDA Runtime
 │
 ▼
GPU
 │
 ▼
Grid
 │
 ├── Block 0
 ├── Block 1
 ├── Block 2
 └── Block N
       │
       ▼
    Threads
       │
       ▼
     Warps
       │
       ▼
      SMs
```

A CUDA kernel is a function executed on the GPU.

Conceptually:

```python
kernel<<<grid, block>>>(...)
```

---

## Practical Example — Vector Addition

Given:

```text
A = [1, 2, 3, 4]
B = [5, 6, 7, 8]
```

We want:

```text
C = [6, 8, 10, 12]
```

GPU mapping:

```text
Thread 0 → C[0] = A[0] + B[0]
Thread 1 → C[1] = A[1] + B[1]
Thread 2 → C[2] = A[2] + B[2]
Thread 3 → C[3] = A[3] + B[3]
```

For millions of elements, the same pattern scales across many threads.

---

# Lesson 0.13 — GPU Performance Mental Model

When a GPU workload is slow, analyze:

```text
                 GPU Workload
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
       Compute      Memory      Parallelism
          │           │           │
       FLOPS       Bandwidth    Occupancy
          │           │           │
          └───────────┼───────────┘
                      ▼
                 Performance
```

---

## 1. Compute

Ask:

```text
Is the GPU doing enough useful computation?
```

Look at:

```text
FLOPS
Tensor Core utilization
CUDA Core utilization
```

---

## 2. Memory

Ask:

```text
Can data reach the compute units fast enough?
```

Consider:

```text
Memory bandwidth
Memory latency
Cache behavior
Memory access pattern
Global memory traffic
```

---

## 3. Parallelism

Ask:

```text
Is there enough parallel work?
```

Consider:

```text
Threads
Blocks
Warps
Occupancy
Batch size
```

---

# Practical GPU Example 1 — Vector Addition

Operation:

```python
C = A + B
```

Suppose:

```text
A = 100 million elements
B = 100 million elements
```

Each element requires:

```text
Read A
Read B
Add
Write C
```

There is relatively little computation.

Therefore:

```text
Large Data
+
Small Computation
        ↓
Low Arithmetic Intensity
        ↓
Likely Memory-Bound
```

### Optimization focus

```text
Memory access
Memory bandwidth
Coalescing
Kernel efficiency
```

---

# Practical GPU Example 2 — Matrix Multiplication

```text
C = A × B
```

Suppose:

```text
A = 4096 × 4096
B = 4096 × 4096
```

Approximate FLOPs:

$$
2 \times 4096^3
$$

≈

```text
137 billion FLOPs
```

This is a large amount of computation.

```text
Large Computation
       ↓
High Arithmetic Intensity
       ↓
Potentially Compute-Bound
       ↓
Tensor Cores become important
```

### Optimization focus

```text
Tensor Cores
Tiling
Shared Memory
Memory Reuse
Kernel Optimization
Mixed Precision
```

---

# Practical GPU Example 3 — Softmax

Softmax:

$$
softmax(x_i)
=
\frac{e^{x_i}}
{\sum_j e^{x_j}}
$$

Conceptually:

```text
Input
 ↓
Find Max
 ↓
Subtract Max
 ↓
Exp
 ↓
Sum
 ↓
Divide
 ↓
Output
```

This involves:

* Reduction
* Element-wise operations
* Multiple memory accesses

Therefore it can be more memory-sensitive than a large GEMM.

### Optimization focus

```text
Memory access
Kernel fusion
Shared memory
Efficient reductions
```

---

# Practical GPU Example 4 — LayerNorm

Simplified:

$$
y =
\frac{x-\mu}
{\sqrt{\sigma^2+\epsilon}}
\gamma+\beta
$$

Flow:

```text
Input
 ↓
Mean
 ↓
Variance
 ↓
Normalize
 ↓
Scale
 ↓
Shift
```

It involves reductions and element-wise operations.

Therefore LayerNorm can be memory-sensitive.

### Optimization focus

```text
Reduce memory traffic
Fuse operations
Efficient reduction
Use shared memory/registers
```

---

# Practical GPU Example 5 — Transformer Attention

```text
Q = XWQ
K = XWK
V = XWV
```

Then:

$$
Attention =
softmax(QK^T/\sqrt{d_k})V
$$

Flow:

```text
X
│
├── Q Projection
├── K Projection
└── V Projection
       │
       ▼
     Q × Kᵀ
       │
       ▼
    Softmax
       │
       ▼
   Attention × V
```

Major operations:

```text
GEMM
GEMM
Reduction
Element-wise operations
GEMM
```

This is why attention requires sophisticated GPU kernels.

---

# Practical GPU Example 6 — LLM Prefill

Suppose:

```text
Prompt = 2,000 tokens
```

The model processes the prompt.

```text
2,000 tokens
     ↓
Parallel Transformer computation
     ↓
KV Cache generated
     ↓
First output token
```

Prefill often has substantial matrix computation.

Typical focus:

```text
Tensor Core utilization
GEMM efficiency
Batching
Memory utilization
```

---

# Practical GPU Example 7 — LLM Decode

After prefill:

```text
Generate token 1
      ↓
Generate token 2
      ↓
Generate token 3
      ↓
...
```

Each decoding step needs access to:

```text
Model Weights
+
KV Cache
```

Therefore:

```text
Decode
  ↓
Memory Access
  ↓
Compute
  ↓
Next Token
```

Decode can become memory-bandwidth-sensitive.

Typical optimization areas:

```text
KV Cache
Memory Bandwidth
Continuous Batching
Quantization
Kernel Fusion
Speculative Decoding
```

---

# Practical GPU Example 8 — Batch Size

Suppose an LLM server processes:

```text
Batch = 1
```

GPU may not be fully utilized.

Increasing:

```text
Batch = 8
Batch = 16
Batch = 32
```

can increase available parallel work.

Conceptually:

```text
Batch ↑
  ↓
Parallel Work ↑
  ↓
GPU Utilization ↑
  ↓
Throughput ↑
```

But:

```text
Batch ↑
  ↓
Memory Usage ↑
  ↓
Queueing ↑
  ↓
Latency may ↑
```

Production systems therefore need to balance latency and throughput.

---

# Practical GPU Example 9 — Quantization

Suppose a model has:

```text
7B parameters
```

Approximate weight memory:

```text
FP32:
7B × 4 bytes ≈ 28 GB

FP16:
7B × 2 bytes ≈ 14 GB

INT8:
7B × 1 byte ≈ 7 GB
```

Actual runtime memory will be higher.

Conceptually:

```text
FP32
 ↓
FP16
 ↓
INT8 / lower precision
 ↓
Memory usage ↓
```

Benefits can include:

* Lower memory footprint
* Lower memory bandwidth requirements
* Higher inference throughput

But accuracy and hardware support must be considered.

---

# Practical GPU Example 10 — CPU vs GPU

### CPU

Good for:

```text
Small workload
Sequential logic
Branch-heavy code
Data preprocessing
API handling
Scheduling
```

### GPU

Good for:

```text
Large matrix operations
Tensor operations
Massive parallel workloads
Deep learning
LLM inference
LLM training
```

Production architecture often uses both:

```text
CPU
│
├── API
├── Scheduling
├── Preprocessing
└── Orchestration
       │
       ▼
GPU
│
├── Model computation
├── Tensor operations
├── Attention
└── Inference
```

---

# Practical GPU Example 11 — Why GPU Utilization Can Be Low

Suppose:

```text
GPU Utilization = 30%
```

Possible reasons:

```text
CPU bottleneck
     ↓
GPU waits

Memory bottleneck
     ↓
GPU compute waits

Small batch
     ↓
Not enough parallelism

Kernel launch overhead
     ↓
GPU idle

Synchronization
     ↓
GPU waits
```

Therefore:

> **GPU utilization alone does not identify the bottleneck.**

Use profiling tools such as:

```text
NVIDIA Nsight Systems
NVIDIA Nsight Compute
PyTorch Profiler
nvidia-smi
```

---

# Practical GPU Example 12 — Production LLM Server

A simplified production architecture:

```text
                 Users
                   │
                   ▼
              API Gateway
                   │
                   ▼
              LLM Server
                   │
          ┌────────┴────────┐
          ▼                 ▼
      Scheduler          KV Cache
          │                 │
          └────────┬────────┘
                   ▼
             CUDA Kernels
                   │
          ┌────────┴────────┐
          ▼                 ▼
      Tensor Cores      GPU Memory
          │                 │
          └────────┬────────┘
                   ▼
                NVIDIA GPU
```

The system must optimize:

```text
Latency
Throughput
GPU utilization
Memory
KV Cache
Batching
Kernel efficiency
Cost
```

---

# Phase 00 — Practical Mental Model

When you encounter any GPU workload, ask these questions:

```text
1. How much parallelism exists?

2. How many FLOPs are required?

3. How much data must be moved?

4. What is the arithmetic intensity?

5. Is it compute-bound or memory-bound?

6. Is the GPU memory sufficient?

7. Are memory accesses efficient?

8. Is the workload using Tensor Cores effectively?

9. Is the batch size large enough?

10. Is CPU/GPU communication a bottleneck?
```

---

# Phase 00 — Final Mental Model

```text
                       Workload
                          │
                          ▼
                  Parallel Operations
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
           Compute                  Memory
           FLOPS                    Bandwidth
              │                       │
              └───────────┬───────────┘
                          ▼
                 Arithmetic Intensity
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
        Compute-bound            Memory-bound
              │                       │
              └───────────┬───────────┘
                          ▼
                   Kernel Efficiency
                          │
                          ▼
                  Actual Performance
```

> **The goal of GPU engineering is not simply to use a GPU. The goal is to keep the right GPU resources busy while minimizing unnecessary computation, memory movement, synchronization, and communication.**

---

# Phase 00 Complete

## What You Should Now Understand

* GPU architecture basics
* CPU vs GPU
* SIMD vs SIMT
* Threads, warps and SMs
* FLOPS
* Precision
* Memory bandwidth
* GPU memory hierarchy
* Compute-bound workloads
* Memory-bound workloads
* Arithmetic intensity
* Roofline model
* CUDA execution lifecycle
* Tensor Cores
* Transformer GPU workloads
* LLM prefill
* LLM decode
* KV cache
* Batch size
* GPU utilization
* Basic GPU optimization strategy

## Next Phase

# Phase 01 — GPU Architecture

```text
GPU Architecture
       ↓
Streaming Multiprocessor
       ↓
Warp Scheduler
       ↓
CUDA Cores
       ↓
Tensor Cores
       ↓
Registers
       ↓
Shared Memory
       ↓
L1 / L2 Cache
       ↓
Memory Controllers
       ↓
HBM / GDDR
```

Phase 01 will move from **GPU fundamentals** into the actual **NVIDIA hardware execution architecture**.
