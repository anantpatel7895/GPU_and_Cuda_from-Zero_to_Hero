# Phase 01 — GPU Architecture

## Goal

Understand how an NVIDIA GPU is structured internally and how CUDA workloads actually execute on the hardware.

```text
GPU
│
├── SMs
│   ├── Warp Schedulers
│   ├── CUDA Cores
│   ├── Tensor Cores
│   ├── Registers
│   ├── Shared Memory
│   └── L1 Cache
│
├── L2 Cache
│
├── Memory Controllers
│
└── HBM / GDDR
```

---

# Lesson 1.1 — GPU Architecture Overview

A GPU consists of multiple **Streaming Multiprocessors (SMs)**.

```text
GPU
│
├── SM 0
├── SM 1
├── SM 2
├── ...
└── SM N
```

Each SM executes many threads organized into warps.

```text
GPU
 ↓
SM
 ↓
Warp
 ↓
Threads
```

### Key Point

> A GPU is a collection of parallel execution resources rather than one giant processor.

---

# Lesson 1.2 — Streaming Multiprocessor (SM)

The **SM** is one of the most important hardware units in an NVIDIA GPU.

Conceptually:

```text
SM
│
├── Warp Schedulers
├── CUDA Cores
├── Tensor Cores
├── Registers
├── Shared Memory
└── L1 Cache
```

The exact organization differs between GPU architectures.

### Practical Example

If a GPU has many SMs:

```text
SM 0 → Blocks
SM 1 → Blocks
SM 2 → Blocks
...
SM N → Blocks
```

Multiple blocks can execute concurrently across SMs.

---

# Lesson 1.3 — CUDA Cores

CUDA Cores are general-purpose arithmetic execution units.

They execute operations such as:

```text
Addition
Multiplication
FMA
Integer operations
Other instructions
```

Simplified:

```text
Thread
  ↓
Instruction
  ↓
CUDA Core
```

### Important

CUDA Core count alone does **not** determine GPU performance.

Performance also depends on:

* Clock frequency
* Memory bandwidth
* Architecture
* Tensor Cores
* Workload
* Kernel efficiency
* Precision

---

# Lesson 1.4 — Tensor Cores

Tensor Cores are specialized hardware designed for high-throughput matrix operations.

```text
Matrix Multiply
      ↓
Tensor Core
      ↓
Matrix Multiply-Accumulate
```

They are particularly important for:

* Deep learning
* Transformers
* LLM inference
* LLM training

### Practical Example

Transformer:

```text
X × W
```

can be mapped to optimized matrix-multiply kernels that use Tensor Cores when supported by the datatype and hardware.

```text
FP16 / BF16 / FP8
        ↓
Tensor Core
        ↓
High AI Throughput
```

---

# Lesson 1.5 — Warp

A **warp** is a group of 32 CUDA threads on NVIDIA GPUs.

```text
Warp
│
├── Thread 0
├── Thread 1
├── ...
└── Thread 31
```

The GPU schedules execution at the warp level.

```text
Block
 ↓
Warps
 ↓
Threads
```

### Why 32?

32 is the NVIDIA-defined warp width.

It is a hardware execution unit, not a universal law of parallel computing.

---

# Lesson 1.6 — Warp Scheduler

The warp scheduler selects eligible warps for execution.

Suppose:

```text
Warp A → waiting for memory
Warp B → ready
Warp C → waiting
Warp D → ready
```

The scheduler can execute:

```text
Warp B
   ↓
Warp D
   ↓
Warp B
   ↓
...
```

This helps hide memory latency.

### Key Idea

> GPUs hide latency by keeping many warps available and switching execution to work that is ready.

---

# Lesson 1.7 — Warp Divergence

Consider:

```python
if thread_id % 2 == 0:
    do_A()
else:
    do_B()
```

Threads in the same warp may take different branches.

Conceptually:

```text
Warp
│
├── Thread 0 → A
├── Thread 1 → B
├── Thread 2 → A
├── Thread 3 → B
└── ...
```

The GPU may need to execute the different paths separately.

```text
Path A
  ↓
Path B
```

This reduces efficiency.

### Key Point

> Keep threads within a warp following similar execution paths whenever practical.

---

# Lesson 1.8 — Registers

Registers are very fast storage associated with threads.

```text
Thread
  │
  └── Registers
```

They hold values used during computation.

Example:

```text
a
b
c = a × b
```

may use registers for temporary values.

### Important Trade-off

Registers are fast but limited.

If a kernel uses too many registers:

```text
Registers / Thread ↑
        ↓
Fewer active warps
        ↓
Occupancy may ↓
```

So:

> More registers can improve per-thread computation but reduce the number of simultaneously active threads.

---

# Lesson 1.9 — Shared Memory

Shared memory is fast memory accessible by threads in the same CUDA block.

```text
Block
│
├── Thread 0
├── Thread 1
├── Thread 2
└── Thread N
        │
        ▼
   Shared Memory
```

It is commonly used for:

* Tiling
* Data reuse
* Reducing global-memory access
* Cooperation between threads

### Practical Example — Matrix Multiplication

Instead of repeatedly reading from global memory:

```text
Global Memory
      ↓
Shared Memory
      ↓
Multiple Threads
      ↓
Reuse Data
```

This can significantly reduce memory traffic.

---

# Lesson 1.10 — L1 and L2 Cache

GPUs also use caches to reduce expensive memory accesses.

Simplified:

```text
Thread
  ↓
Registers
  ↓
L1 / Shared Memory
  ↓
L2 Cache
  ↓
Global Memory
```

Generally:

```text
Closer to compute
      ↓
Lower latency
      ↓
Smaller capacity
```

and:

```text
Farther from compute
      ↓
Higher latency
      ↓
Larger capacity
```

The exact cache organization depends on the NVIDIA GPU architecture.

---

# Lesson 1.11 — Global Memory

Global memory is the large GPU memory space accessible by GPU kernels.

Depending on the GPU:

```text
GDDR
HBM
```

may be used.

LLM workloads store things such as:

```text
Model Weights
KV Cache
Activations
Input/Output
Temporary Data
```

### Key Problem

Global memory has much higher latency than registers.

Therefore:

> Efficient kernels try to maximize data reuse and minimize unnecessary global-memory traffic.

---

# Lesson 1.12 — Memory Access and Coalescing

GPU threads often access memory in parallel.

Good access pattern:

```text
Thread 0 → Address 0
Thread 1 → Address 1
Thread 2 → Address 2
Thread 3 → Address 3
```

This allows memory accesses to be efficiently combined.

Poor pattern:

```text
Thread 0 → Address 0
Thread 1 → Address 1000
Thread 2 → Address 500
Thread 3 → Address 9000
```

This can result in inefficient memory transactions.

### Key Term

> **Coalesced memory access**

is an important GPU optimization technique.

---

# Lesson 1.13 — CUDA Block

A CUDA kernel launches a grid of blocks.

```text
Grid
│
├── Block 0
├── Block 1
├── Block 2
└── Block N
```

A block contains threads:

```text
Block
│
├── Thread 0
├── Thread 1
├── ...
└── Thread N
```

Threads within a block can:

* Share shared memory
* Synchronize
* Cooperate on computation

---

# Lesson 1.14 — Grid → Block → Warp → Thread

The complete CUDA hierarchy:

```text
GPU
 │
 └── Grid
      │
      ├── Block
      │    │
      │    ├── Warp
      │    │    ├── Thread
      │    │    ├── Thread
      │    │    └── ...
      │    │
      │    └── Warp
      │
      └── Block
```

Important:

```text
Grid
 ↓
Blocks
 ↓
Warps
 ↓
Threads
```

---

# Lesson 1.15 — Block Scheduling

Blocks are scheduled onto available SMs.

```text
Grid
│
├── Block 0 ──► SM 0
├── Block 1 ──► SM 1
├── Block 2 ──► SM 2
└── Block 3 ──► Available SM
```

The programmer generally does not hard-code:

```text
Block 0 → SM 0
```

The GPU scheduler decides where blocks execute.

### Important Design Principle

Blocks should generally be independent so that they can execute on any suitable SM.

---

# Lesson 1.16 — Occupancy

**Occupancy** is roughly the ratio of active warps on an SM to the maximum number of active warps supported by that SM.

Conceptually:

```text
Maximum active warps = 64
Active warps = 48

Occupancy = 48 / 64
           = 75%
```

Higher occupancy can help hide latency.

But:

> **Higher occupancy does not automatically mean higher performance.**

A kernel may perform better with lower occupancy if it has:

* Better data reuse
* More registers
* Better instruction efficiency
* Higher arithmetic intensity

---

# Lesson 1.17 — Synchronization

Threads sometimes need to coordinate.

For example:

```text
Thread 0 ─┐
Thread 1 ─┤
Thread 2 ─┼──► Shared computation
Thread 3 ─┘
              │
              ▼
        Synchronization
              │
              ▼
          Continue
```

CUDA provides synchronization mechanisms such as block-level synchronization.

Synchronization has a cost.

Too much synchronization can reduce performance.

---

# Lesson 1.18 — Practical Matrix Multiplication Architecture

Consider:

```text
C = A × B
```

A common optimized strategy:

```text
Global Memory
      │
      ▼
Load Tiles
      │
      ▼
Shared Memory
      │
      ▼
Registers
      │
      ▼
Tensor/CUDA Cores
      │
      ▼
Output
```

This reduces repeated global-memory access.

The general principle:

> Move data closer to computation and reuse it as much as possible.

---

# Lesson 1.19 — Practical Transformer Example

Transformer linear layer:

```text
Y = XW
```

Execution conceptually:

```text
Input X
   │
   ▼
Global Memory
   │
   ▼
CUDA Kernel
   │
   ▼
Load / Tile Data
   │
   ▼
Shared Memory / Registers
   │
   ▼
Tensor Cores
   │
   ▼
Output
```

This is why GPU architecture knowledge matters for LLM inference.

---

# Lesson 1.20 — Practical LLM Decode Example

During autoregressive decoding:

```text
Request
   ↓
Scheduler
   ↓
Decode Kernel
   ↓
Read Weights / KV Cache
   ↓
GPU Computation
   ↓
Next Token
```

Potential bottlenecks:

```text
Memory Bandwidth
KV Cache Access
Kernel Launch
Batch Size
Synchronization
```

This is why LLM inference engines optimize more than just matrix multiplication.

---

# Phase 01 — Practical Examples

## Example 1 — Vector Addition

```text
C[i] = A[i] + B[i]
```

Characteristics:

```text
Parallelism → High
Computation → Low
Memory Traffic → High
```

Likely:

```text
Memory-bound
```

---

## Example 2 — Matrix Multiplication

```text
C = A × B
```

Characteristics:

```text
Parallelism → High
Computation → Very High
Data Reuse → High
```

Likely:

```text
Compute-intensive
```

Excellent workload for Tensor Cores.

---

## Example 3 — LayerNorm

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

Characteristics:

```text
Reduction
+
Element-wise operations
+
Memory traffic
```

Can become memory-sensitive.

---

## Example 4 — Attention

```text
Q × Kᵀ
   ↓
Softmax
   ↓
Attention × V
```

Contains both:

```text
Compute-heavy GEMMs
+
Memory/reduction operations
```

The bottleneck can change depending on sequence length, batch size, GPU, and implementation.

---

## Example 5 — LLM Serving

```text
Users
  ↓
API
  ↓
Scheduler
  ↓
Continuous Batching
  ↓
CUDA Kernels
  ↓
Tensor Cores
  ↓
GPU Memory
```

Important resources:

```text
Compute
Memory Bandwidth
Memory Capacity
KV Cache
SM Utilization
Kernel Efficiency
```

---

# Phase 01 — Most Important Concepts

Before moving to CUDA programming, you should understand:

* [ ] GPU contains multiple SMs
* [ ] SM executes CUDA blocks
* [ ] Blocks contain threads
* [ ] Threads execute in warps
* [ ] NVIDIA warp = 32 threads
* [ ] Warp scheduler selects ready warps
* [ ] CUDA Cores perform general arithmetic
* [ ] Tensor Cores accelerate matrix operations
* [ ] Registers are private to threads
* [ ] Shared memory is shared within a block
* [ ] L1/L2 caches reduce memory latency
* [ ] Global memory provides large GPU storage
* [ ] Coalesced access improves memory efficiency
* [ ] Too many registers can reduce occupancy
* [ ] Synchronization has a cost
* [ ] Occupancy is useful but is not the same as performance
* [ ] Blocks are scheduled dynamically onto SMs
* [ ] Data reuse is critical for performance

---

# Phase 01 — Final Architecture Mental Model

```text
                         NVIDIA GPU
                              │
              ┌───────────────┴───────────────┐
              │                               │
             SM                              SM
              │                               │
       ┌──────┼──────┐                 ┌──────┼──────┐
       │      │      │                 │      │      │
   Scheduler CUDA  Tensor          Scheduler CUDA  Tensor
             Cores  Cores                   Cores  Cores
       │
       ├── Registers
       │
       ├── Shared Memory
       │
       └── L1 Cache
              │
              ▼
           L2 Cache
              │
              ▼
       Memory Controllers
              │
              ▼
          HBM / GDDR
```

CUDA execution:

```text
Kernel
  ↓
Grid
  ↓
Blocks
  ↓
Warps
  ↓
Threads
  ↓
SM
  ↓
CUDA / Tensor Cores
  ↓
Registers / Shared Memory
  ↓
L1 / L2
  ↓
Global Memory
```

## Core Principle

> **GPU performance comes from efficiently scheduling massive numbers of threads while keeping computation fed with data through the memory hierarchy.**

---

# Phase 01 Complete

Next:

# Phase 02 — CUDA Programming

We move from understanding the hardware to actually programming it:

```text
CUDA
 │
 ├── Kernel
 ├── Thread
 ├── Block
 ├── Grid
 ├── Thread Indexing
 ├── Memory Management
 ├── Synchronization
 ├── Shared Memory
 ├── Streams
 ├── Events
 └── CUDA Errors
```

Then we will build actual CUDA programs rather than only studying architecture.
