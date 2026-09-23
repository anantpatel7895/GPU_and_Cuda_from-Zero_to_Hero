# Lesson 0.1 — GPU Fundamentals

## Questions & Answers

This document contains the answers to the key questions from **Lesson 0.1 — What is a GPU?**

The objective is not just to memorize definitions, but to build the mental model required for CUDA, GPU optimization, and LLM inference.

---

# 1. What is a GPU?

## Answer

A **GPU (Graphics Processing Unit)** is a massively parallel processor designed to execute a large number of similar operations concurrently.

Unlike a CPU, which has a relatively small number of powerful and complex cores, a GPU contains many parallel execution resources optimized for high-throughput workloads.

A simplified view:

```text
CPU

┌───────────────────────────┐
│                           │
│  Core 0   Core 1          │
│  Core 2   Core 3          │
│                           │
│  Large Cache              │
│  Complex Control Logic    │
│                           │
└───────────────────────────┘
```

GPU:

```text
GPU

┌─────────────────────────────────────┐
│                                     │
│  SM 0   SM 1   SM 2   SM 3          │
│                                     │
│  SM 4   SM 5   SM 6   SM 7          │
│                                     │
│  SM 8   SM 9   SM 10  SM 11         │
│                                     │
│          ... many SMs ...           │
│                                     │
└─────────────────────────────────────┘
```

Each **SM (Streaming Multiprocessor)** can execute many threads.

We will study SMs in detail in Phase 1.

### Important

A GPU is not simply:

> "A CPU with more cores."

It is a different architecture optimized for **massive parallelism and throughput**.

---

# 2. Why is a GPU good for deep learning?

Deep-learning workloads contain enormous amounts of parallel mathematical computation.

Common operations include:

```text
Matrix Multiplication
Vector Operations
Convolution
Attention
Softmax
LayerNorm
MLP
Embedding Operations
```

Consider:

```python
C = A @ B
```

For matrix multiplication, many elements of `C` can be calculated independently.

Conceptually:

```text
C[0][0]  ──┐
C[0][1]  ──┤
C[0][2]  ──┤
C[0][3]  ──┤
C[1][0]  ──┤
C[1][1]  ──┤
   ...     │
C[N][N]  ──┘
            │
            ▼
       Parallel Work
```

A GPU is designed to process this kind of workload efficiently.

### Neural networks are dominated by mathematical operations

For example, a Transformer contains:

```text
Input
  │
  ▼
Embedding
  │
  ▼
Q/K/V Projections
  │
  ▼
Matrix Multiplication
  │
  ▼
Attention
  │
  ▼
Softmax
  │
  ▼
MLP
  │
  ▼
Output
```

Most of the expensive operations involve tensors and can be parallelized.

Therefore:

```text
Deep Learning
      │
      ▼
Large Tensor Operations
      │
      ▼
Massive Parallelism
      │
      ▼
GPU
```

### Important qualification

Not every deep-learning operation automatically becomes faster on a GPU.

Performance depends on:

* amount of parallel work
* memory access patterns
* GPU memory bandwidth
* computation required
* kernel launch overhead
* synchronization
* data transfer between CPU and GPU

---

# 3. CPU vs GPU?

The most important difference is their **design objective**.

| Characteristic                | CPU                     | GPU                                   |
| ----------------------------- | ----------------------- | ------------------------------------- |
| Primary goal                  | Low latency             | High throughput                       |
| Number of execution resources | Relatively few          | Very many                             |
| Individual execution resource | Powerful/complex        | Simpler/massively parallel            |
| Parallelism                   | Moderate                | Massive                               |
| Control flow                  | Excellent               | Less suitable for irregular branching |
| Cache                         | Large and sophisticated | Optimized differently for throughput  |
| Memory bandwidth              | Lower                   | Very high                             |
| Sequential workloads          | Excellent               | Usually poor fit                      |
| Matrix operations             | Good                    | Excellent                             |
| Neural networks               | Good                    | Excellent                             |
| General-purpose workloads     | Excellent               | Workload dependent                    |

### CPU mental model

```text
Task
 │
 ▼
CPU
 │
 ├── Core 0
 ├── Core 1
 ├── Core 2
 └── Core 3
```

### GPU mental model

```text
Large Parallel Workload
          │
          ▼
 ┌──────────────────────┐
 │                      │
 │  Thousands of        │
 │  threads             │
 │                      │
 └──────────────────────┘
          │
          ▼
       GPU
```

### Example

Suppose we need:

```python
C[i] = A[i] + B[i]
```

for 10 million elements.

The calculations are independent:

```text
C[0] = A[0] + B[0]
C[1] = A[1] + B[1]
C[2] = A[2] + B[2]
...
C[9999999] = A[9999999] + B[9999999]
```

This is highly parallel and therefore well suited to GPU execution.

---

# 4. Why is matrix multiplication suitable for GPUs?

Matrix multiplication is one of the most important workloads in deep learning.

Given:

```text
A = M × K

B = K × N

C = M × N
```

we calculate:

```text
C = A × B
```

Each output element is:

```text
C[i][j] =
    A[i][0] × B[0][j]
  + A[i][1] × B[1][j]
  + ...
  + A[i][K-1] × B[K-1][j]
```

The key property is:

> Different output elements can generally be calculated independently.

For example:

```text
C[0][0] ──────────────► Thread
C[0][1] ──────────────► Thread
C[0][2] ──────────────► Thread
C[0][3] ──────────────► Thread

C[1][0] ──────────────► Thread
C[1][1] ──────────────► Thread
C[1][2] ──────────────► Thread
C[1][3] ──────────────► Thread

...
```

This maps naturally to GPU parallelism.

## Why GPUs are especially good at it

Matrix multiplication involves huge numbers of:

```text
Multiply
   +
Add
```

operations.

Modern NVIDIA GPUs also contain **Tensor Cores**, specialized hardware designed to accelerate matrix operations used heavily by AI workloads.

Therefore:

```text
Matrix Multiplication
        │
        ├── Massive parallelism
        │
        ├── Large number of arithmetic operations
        │
        └── Tensor Core acceleration
                    │
                    ▼
                  GPU
```

This is one of the fundamental reasons GPUs dominate modern deep-learning workloads.

---

# 5. What is throughput vs latency?

These concepts are extremely important for production AI systems.

## Latency

Latency is the amount of time required to complete one operation or request.

Example:

```text
Request
   │
   ▼
Processing
   │
   ▼
Response

Time = 50 ms
```

Latency:

```text
50 ms
```

For an LLM API, examples include:

```text
Time To First Token (TTFT)
Time Per Output Token (TPOT)
End-to-end request latency
```

---

## Throughput

Throughput measures how much work can be completed per unit of time.

For example:

```text
100 requests
────────────
1 second
```

Throughput:

```text
100 requests/second
```

For LLMs:

```text
tokens/second
requests/second
```

are common throughput metrics.

---

## Simple comparison

Imagine a GPU server processing requests.

### System A

```text
1 request → 10 ms
```

Latency:

```text
10 ms
```

### System B

```text
100 requests → 1 second
```

Throughput:

```text
100 requests/sec
```

A system can have excellent throughput while individual requests have higher latency.

Therefore:

```text
Latency
   =
How quickly does one job finish?

Throughput
   =
How much work can we complete over time?
```

---

# 6. What does FLOPS measure?

**FLOPS = Floating Point Operations Per Second**

It measures how many floating-point arithmetic operations a processor can theoretically or actually perform per second, depending on the context.

Examples:

```text
1 GFLOPS  = 10⁹ FLOPS
1 TFLOPS  = 10¹² FLOPS
1 PFLOPS  = 10¹⁵ FLOPS
```

Suppose a GPU is rated at:

```text
100 TFLOPS
```

This means its relevant peak floating-point capability is approximately:

```text
100 × 10¹²
```

floating-point operations per second under the specified numerical format and hardware conditions.

---

## Why FLOPS matters for AI

Deep learning performs enormous numbers of floating-point operations.

For example:

```text
Matrix Multiplication
        │
        ▼
Multiply + Add
        │
        ▼
Billions/Trillions of Operations
```

Therefore FLOPS is useful for understanding computational capability.

---

## But FLOPS is NOT the same as performance

This is extremely important.

Suppose:

```text
GPU A
100 TFLOPS

GPU B
200 TFLOPS
```

It does **not** automatically mean GPU B will be 2× faster for every workload.

Why?

Because performance can be limited by:

```text
Memory bandwidth
Memory latency
Kernel launch overhead
Synchronization
Data transfer
Cache behavior
Occupancy
Communication
```

This leads to one of the most important GPU concepts:

> **Peak FLOPS is only one part of GPU performance.**

---

# 7. What does memory bandwidth measure?

Memory bandwidth measures how quickly data can be transferred between memory and the processor.

It is usually expressed as:

```text
GB/s
TB/s
```

For example:

```text
Memory bandwidth = 1 TB/s
```

means the memory subsystem can theoretically transfer approximately:

```text
1 TB
```

of data per second under the relevant conditions.

---

## Why memory bandwidth matters

Consider:

```python
C = A + B
```

For each element:

```text
Read A
Read B
Add
Write C
```

Only one arithmetic operation is performed, but multiple memory accesses are required.

Therefore the workload can become limited by memory movement.

Conceptually:

```text
Memory
  │
  │ Data movement
  ▼
GPU
  │
  │ Addition
  ▼
Result
```

If the GPU spends most of its time waiting for data, having more arithmetic units won't necessarily solve the problem.

---

# 8. FLOPS vs Memory Bandwidth

This distinction is fundamental.

```text
             GPU Performance
                    │
          ┌─────────┴─────────┐
          │                   │
       Compute              Memory
          │                   │
        FLOPS             Bandwidth
          │                   │
          ▼                   ▼
   Arithmetic work      Data movement
```

A workload can be:

### Compute-bound

```text
GPU spends most of its time
performing arithmetic.
```

### Memory-bound

```text
GPU spends most of its time
moving/waiting for data.
```

We'll study this in much greater depth in:

**Lesson 0.7 — Compute-bound vs Memory-bound**

---

# 9. What is the difference between a GPU and CUDA?

This is one of the most important distinctions.

## GPU

A GPU is **hardware**.

Examples:

```text
NVIDIA A10
NVIDIA A100
NVIDIA H100
NVIDIA H200
NVIDIA B200
```

The GPU contains physical hardware such as:

```text
SMs
CUDA Cores
Tensor Cores
Registers
Caches
Memory Controllers
```

---

## CUDA

CUDA is NVIDIA's **GPU computing platform and programming ecosystem**.

It includes:

```text
CUDA Programming Model
CUDA Runtime
CUDA APIs
CUDA Libraries
CUDA Compiler Toolchain
CUDA Toolkit
```

Simplified:

```text
GPU
=
Hardware

CUDA
=
Software platform used to program and utilize
NVIDIA GPUs for general-purpose computation
```

---

## Example

Imagine:

```text
NVIDIA A10
```

is the physical machine.

CUDA provides the software environment that allows applications to execute computation on that GPU.

---

# 10. What is the relationship between PyTorch, CUDA and the GPU?

Think in layers.

```text
┌─────────────────────────────┐
│          PyTorch            │
│       ML Framework          │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│            CUDA             │
│ Runtime / APIs / Libraries  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│       NVIDIA GPU            │
│ Hardware / SMs / Memory     │
└─────────────────────────────┘
```

Suppose we write:

```python
import torch

x = torch.randn(
    4096,
    4096,
    device="cuda"
)
```

At a high level:

```text
Python
  │
  ▼
PyTorch
  │
  ▼
CUDA APIs / Runtime
  │
  ▼
NVIDIA Driver
  │
  ▼
GPU
  │
  ▼
GPU Memory
```

Then if we perform:

```python
y = x @ x
```

PyTorch dispatches an appropriate GPU implementation for the operation.

The actual computation eventually executes on the GPU.

---

## Important distinction

PyTorch does **not** equal CUDA.

PyTorch is a machine-learning framework.

CUDA is NVIDIA's GPU computing platform.

The GPU is the hardware.

```text
PyTorch
   │
   │ uses
   ▼
CUDA ecosystem
   │
   │ targets
   ▼
NVIDIA GPU
```

---

# 11. Why doesn't having more GPU cores automatically guarantee better performance?

This is one of the most important GPU engineering questions.

Because GPU performance is not determined only by the number of compute cores.

Consider:

```text
GPU A
10,000 execution units

GPU B
15,000 execution units
```

It does not follow that:

```text
GPU B = 1.5 × performance
```

because other bottlenecks may dominate.

---

## Reason 1 — Memory bandwidth

Suppose the workload is memory-bound.

```text
Compute:
██████
Memory:
████████████████████
```

Adding more compute resources won't help much because the GPU is waiting for data.

---

## Reason 2 — Poor memory access

Suppose threads access memory inefficiently:

```text
Thread 0 → random address
Thread 1 → random address
Thread 2 → random address
Thread 3 → random address
```

The GPU may waste memory bandwidth.

Efficient GPU programming requires concepts such as:

```text
Memory Coalescing
Cache Locality
Shared Memory
Memory Alignment
```

---

## Reason 3 — Insufficient parallelism

A GPU needs enough work to keep its execution resources busy.

If the workload is:

```text
Very small
```

the GPU may be underutilized.

```text
GPU capacity:
████████████████████

Work:
██
```

Most of the GPU remains idle.

---

## Reason 4 — Kernel launch overhead

Launching GPU kernels has overhead.

If we repeatedly execute tiny operations:

```text
Launch
Compute
Finish

Launch
Compute
Finish

Launch
Compute
Finish
```

the launch overhead can become significant.

This is one reason **kernel fusion** can improve performance.

Instead of:

```text
Kernel A
Kernel B
Kernel C
```

we may be able to perform:

```text
Fused Kernel
```

---

## Reason 5 — Synchronization

Threads sometimes need to wait for each other.

For example:

```text
Thread 0 ────────┐
Thread 1 ────────┤
Thread 2 ────────┤──► Synchronization
Thread 3 ────────┘
                         │
                         ▼
                     Continue
```

Too much synchronization can reduce performance.

---

## Reason 6 — Warp divergence

GPUs execute threads in groups called **warps**.

If threads within a warp take different branches:

```text
Thread 0 → if
Thread 1 → if
Thread 2 → else
Thread 3 → else
```

the GPU may need to execute different paths separately.

This reduces efficiency.

We will study this in Phase 4.

---

## Reason 7 — Register and shared-memory limitations

A kernel may require a large amount of:

```text
Registers
Shared Memory
```

This can reduce how many blocks/warps can simultaneously reside on an SM.

That can reduce occupancy and affect performance.

---

## Reason 8 — Communication

In multi-GPU systems:

```text
GPU 0
  │
  │ communication
  ▼
GPU 1
```

The bottleneck may be:

```text
PCIe
NVLink
NCCL
Synchronization
```

rather than GPU computation.

This becomes extremely important for:

```text
Tensor Parallelism
Pipeline Parallelism
Distributed Training
```

---

# 12. The complete answer to Question 10

The correct mental model is:

```text
GPU Performance
       │
       ├── Compute capability
       │
       ├── Memory bandwidth
       │
       ├── Memory latency
       │
       ├── Cache behavior
       │
       ├── Parallelism
       │
       ├── Occupancy
       │
       ├── Warp efficiency
       │
       ├── Kernel launch overhead
       │
       ├── Synchronization
       │
       ├── Data transfer
       │
       └── Inter-GPU communication
```

Therefore:

> **More GPU cores provide more potential compute capacity, but actual performance depends on whether the workload can effectively utilize that capacity without being limited by memory, synchronization, communication, or other bottlenecks.**

---

# 🎯 Final Summary

The 10 questions can be reduced to this mental model:

```text
                         GPU
                          │
             ┌────────────┴────────────┐
             │                         │
          Compute                    Memory
             │                         │
          FLOPS                    Bandwidth
             │                         │
             └────────────┬────────────┘
                          │
                          ▼
                     Parallelism
                          │
             ┌────────────┴────────────┐
             │                         │
          Threads                    Warps
             │                         │
          Blocks                     SMs
             │                         │
             └────────────┬────────────┘
                          │
                          ▼
                     Workload
                          │
                ┌─────────┴─────────┐
                │                   │
          Compute-bound        Memory-bound
```

And the software stack:

```text
Application
    │
    ▼
PyTorch
    │
    ▼
CUDA
    │
    ▼
NVIDIA Driver
    │
    ▼
GPU Hardware
    │
    ├── SMs
    ├── CUDA Cores
    ├── Tensor Cores
    ├── Cache
    └── GPU Memory
```

---

# 🧠 Interview One-Liners

| Question                                    | Short Answer                                                                                           |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| What is a GPU?                              | A massively parallel processor optimized for high-throughput workloads.                                |
| Why is GPU good for deep learning?          | Deep learning contains large amounts of parallel tensor computation.                                   |
| CPU vs GPU?                                 | CPU prioritizes low latency and general-purpose execution; GPU prioritizes parallel throughput.        |
| Why is MatMul good for GPUs?                | Many output elements require independent multiply-add operations that can execute in parallel.         |
| Latency vs throughput?                      | Latency is time per operation/request; throughput is amount of work completed per unit time.           |
| What is FLOPS?                              | Floating-point operations per second.                                                                  |
| What is memory bandwidth?                   | The rate at which data can be transferred through the memory subsystem.                                |
| GPU vs CUDA?                                | GPU is hardware; CUDA is NVIDIA's GPU computing platform and software ecosystem.                       |
| PyTorch → CUDA → GPU?                       | PyTorch dispatches GPU operations through CUDA/software layers to execute on NVIDIA GPU hardware.      |
| Why don't more cores guarantee performance? | Workloads may be limited by memory, parallelism, synchronization, communication, or other bottlenecks. |

---

# 🔑 Most Important Takeaways

### 1.

> **GPU ≠ faster CPU**

GPU and CPU are optimized for different workload characteristics.

### 2.

> **GPU performance = compute + memory + parallelism + efficient execution**

### 3.

> **FLOPS alone does not determine performance.**

### 4.

> **Memory bandwidth is often as important as compute capability.**

### 5.

> **CUDA is software; GPU is hardware.**

### 6.

> **PyTorch is a framework that uses lower-level GPU software such as CUDA to execute workloads on NVIDIA GPUs.**

### 7.

> **The first question when optimizing a GPU workload should be: "What is the bottleneck?"**

Not:

> "How can I use more GPU cores?"

---

# 📌 Next Lesson

## Lesson 0.2 — CPU vs GPU: Architecture & Execution Model

We will go one level deeper:

```text
CPU
 │
 ├── Core
 ├── ALU
 ├── Cache
 ├── Branch Predictor
 └── Out-of-Order Execution


GPU
 │
 ├── SM
 ├── CUDA Cores
 ├── Tensor Cores
 ├── Warp
 ├── Warp Scheduler
 └── Memory Hierarchy
```

Then we'll introduce the critical concepts:

```text
Instruction-Level Parallelism
        ↓
Data-Level Parallelism
        ↓
SIMD
        ↓
SIMT
        ↓
CUDA execution model
```

These concepts will become the foundation for understanding **threads → warps → blocks → SMs → CUDA kernels**.
