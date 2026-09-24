# Lesson 0.2 — CPU vs GPU

## Questions & Answers

This document contains the answers to the key questions from:

> **Lesson 0.2 — CPU vs GPU: Architecture & Execution Model**

The objective is to understand the architectural and execution differences between CPUs and GPUs, and to build the foundation required for CUDA programming.

---

# 1. What is the architectural difference between CPU and GPU?

The fundamental difference is what each processor is optimized for.

### CPU

A CPU is designed primarily for:

* Low latency
* General-purpose workloads
* Complex control flow
* Sequential execution
* Branch-heavy workloads
* Instruction-level parallelism

A simplified CPU:

```text
                     CPU
                      │
          ┌───────────┼───────────┐
          │           │           │
        Core 0      Core 1      Core 2
          │           │           │
       Complex     Complex     Complex
       execution   execution   execution
          │           │           │
          └───────────┼───────────┘
                      │
                  Large Cache
```

Each CPU core is relatively complex.

---

### GPU

A GPU is designed primarily for:

* High throughput
* Massive parallelism
* Large-scale numerical computation
* Tensor operations
* High memory bandwidth

Simplified:

```text
                         GPU
                          │
          ┌───────────────┼───────────────┐
          │               │               │
         SM              SM              SM
          │               │               │
     Many Threads    Many Threads    Many Threads
          │               │               │
          └───────────────┼───────────────┘
                          │
                       L2 Cache
                          │
                          ▼
                     GPU Memory
```

### Key difference

```text
CPU
→ Few powerful and complex execution resources
→ Optimize for low latency

GPU
→ Many parallel execution resources
→ Optimize for high throughput
```

---

# 2. Why is GPU optimized for throughput rather than latency?

A GPU is designed to process a **large amount of work simultaneously**.

Consider:

```text
Task 1
Task 2
Task 3
Task 4
...
Task 10000
```

These tasks may be distributed across many GPU threads.

```text
                GPU
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
    Thread      Thread    Thread
       0           1         2
       │           │         │
      Task 0      Task 1    Task 2
```

The objective is not necessarily to make one individual task finish as quickly as possible.

The objective is:

> **Complete as much total work as possible per unit of time.**

That is throughput.

---

## Latency

Latency answers:

> How long does one operation take?

Example:

```text
Request
   │
   ▼
Processing
   │
   ▼
Response

50 ms
```

---

## Throughput

Throughput answers:

> How much work can we complete per unit of time?

Example:

```text
100 requests / second
```

---

## Why GPU favors throughput

A GPU contains many execution resources that can remain busy when there is sufficient parallel work.

```text
Large workload
      │
      ▼
Thousands of threads
      │
      ▼
Many warps
      │
      ▼
Multiple SMs
      │
      ▼
High throughput
```

This is particularly useful for:

* Matrix multiplication
* Neural networks
* Image processing
* Embeddings
* Attention
* Large tensor operations

---

# 3. What is Instruction-Level Parallelism?

**Instruction-Level Parallelism (ILP)** means executing multiple independent instructions concurrently or in an overlapping manner.

Suppose we have:

```text
Instruction A → Add
Instruction B → Multiply
Instruction C → Load
Instruction D → Compare
```

If these instructions do not depend on each other, a modern CPU may execute some of them concurrently.

Conceptually:

```text
Instruction A ───────► Execution
Instruction B ───────► Execution
Instruction C ───────► Execution
Instruction D ───────► Execution
```

CPU architectures use sophisticated mechanisms to exploit ILP.

Examples include:

* Out-of-order execution
* Instruction scheduling
* Multiple execution units
* Speculative execution
* Branch prediction

---

## Example

Consider:

```python
a = b + c
x = y * z
p = q - r
```

There may be no dependency between these operations.

Therefore the processor can potentially overlap their execution.

---

## Important distinction

ILP focuses on:

> **Parallelism between instructions.**

It does not necessarily mean processing many independent data elements using thousands of threads.

That is where Data-Level Parallelism becomes important.

---

# 4. What is Data-Level Parallelism?

**Data-Level Parallelism (DLP)** means performing the same operation on multiple pieces of data.

Example:

```python
C = A + B
```

Suppose:

```text
A = [1, 2, 3, 4]
B = [5, 6, 7, 8]
```

Then:

```text
C = [6, 8, 10, 12]
```

The same operation:

```text
addition
```

is applied independently to multiple elements.

```text
A[0] + B[0]
A[1] + B[1]
A[2] + B[2]
A[3] + B[3]
```

This is Data-Level Parallelism.

---

## DLP is extremely important for deep learning

Neural networks repeatedly perform operations over large tensors:

```text
Tensor
  │
  ├── Element 0
  ├── Element 1
  ├── Element 2
  ├── ...
  └── Element N
```

Many operations can be performed independently.

Therefore:

```text
Large Tensor
     ↓
Data-Level Parallelism
     ↓
GPU
```

---

# 5. What is SIMD?

SIMD means:

> **Single Instruction, Multiple Data**

One instruction operates on multiple data values.

Conceptually:

```text
                 ADD instruction
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
      A[0]            A[1]           A[2]
        +              +              +
      B[0]            B[1]           B[2]
        │              │              │
        ▼              ▼              ▼
      C[0]            C[1]           C[2]
```

A vectorized CPU instruction can operate on several data elements simultaneously.

---

## CPU SIMD

Modern CPUs provide vector instruction sets such as:

```text
SSE
AVX
AVX2
AVX-512
```

For example:

```text
A = [A0 A1 A2 A3]
B = [B0 B1 B2 B3]

A + B

    ↓

C = [C0 C1 C2 C3]
```

A vector instruction processes multiple values.

---

# 6. What is SIMT?

SIMT means:

> **Single Instruction, Multiple Threads**

It is the fundamental execution model exposed by NVIDIA CUDA programming.

Instead of thinking primarily in terms of vector lanes, the programmer thinks in terms of threads.

Example:

```text
Instruction: ADD

Thread 0 → A[0] + B[0]
Thread 1 → A[1] + B[1]
Thread 2 → A[2] + B[2]
Thread 3 → A[3] + B[3]
...
```

Conceptually:

```text
                  ADD
                   │
       ┌───────────┼───────────┐
       ▼           ▼           ▼
    Thread 0    Thread 1    Thread 2
       │           │           │
    A[0]+B[0]   A[1]+B[1]   A[2]+B[2]
```

CUDA allows us to program these threads.

---

# 7. SIMD vs SIMT?

They are related but conceptually different.

| Feature                  | SIMD                             | SIMT                                |
| ------------------------ | -------------------------------- | ----------------------------------- |
| Full form                | Single Instruction Multiple Data | Single Instruction Multiple Threads |
| Commonly associated with | CPU vector units                 | NVIDIA GPU programming model        |
| Programmer abstraction   | Vector/data lanes                | Threads                             |
| Execution                | Vector operation                 | Threads grouped into warps          |
| Typical hardware         | CPU vector units                 | GPU execution hardware              |
| Programming style        | Vectorization                    | Thread-based parallel programming   |

---

## SIMD

Think:

```text
One instruction
       │
       ▼
Multiple data lanes
```

## SIMT

Think:

```text
One instruction
       │
       ▼
Multiple threads
       │
       ▼
Warp
```

---

## Important nuance

SIMT should not be interpreted as:

> "Every thread has completely independent hardware."

The GPU groups threads into **warps**.

This is extremely important.

---

# 8. Why does NVIDIA use the SIMT model?

SIMT gives programmers a convenient abstraction for massive parallelism.

Instead of manually managing vector registers and hardware lanes, the programmer can write:

```cpp
int idx =
    blockIdx.x * blockDim.x
    + threadIdx.x;

C[idx] = A[idx] + B[idx];
```

Conceptually:

```text
Thread 0 → C[0]
Thread 1 → C[1]
Thread 2 → C[2]
Thread 3 → C[3]
...
```

CUDA handles mapping the logical thread hierarchy onto GPU hardware.

---

## Why is this useful?

It provides:

### 1. Simple programming model

Think in terms of:

```text
Threads
Blocks
Grids
```

rather than directly programming individual hardware execution units.

### 2. Massive parallelism

Thousands of threads can be launched.

### 3. Hardware efficiency

The GPU groups threads into warps and executes them efficiently.

### 4. Scalability

The same kernel can often run across GPUs with different numbers of SMs.

---

# 9. What is a warp?

A **warp** is a group of **32 CUDA threads** that NVIDIA hardware schedules and executes together.

Conceptually:

```text
                    Warp
                      │
 ┌────────────────────┼────────────────────┐
 ▼                    ▼                    ▼
Thread 0           Thread 1           Thread 2
   ...                ...                ...
                                      Thread 31
```

A warp contains:

```text
32 threads
```

---

## Example

Suppose we launch:

```text
256 threads
```

They are organized as:

```text
256 threads
÷
32 threads/warp
=
8 warps
```

So:

```text
Block
 │
 ├── Warp 0 → 32 threads
 ├── Warp 1 → 32 threads
 ├── Warp 2 → 32 threads
 ├── Warp 3 → 32 threads
 ├── Warp 4 → 32 threads
 ├── Warp 5 → 32 threads
 ├── Warp 6 → 32 threads
 └── Warp 7 → 32 threads
```

---

# 10. Why does a warp contain 32 threads?

The important answer is:

> **32 is NVIDIA's hardware-defined warp width for CUDA GPUs.**

It is not because 32 is universally mathematically optimal.

It is a hardware execution design choice that allows NVIDIA GPUs to efficiently group threads for SIMT execution.

A warp gives the GPU a convenient unit for:

* instruction scheduling
* execution
* branch handling
* synchronization behavior
* warp-level operations

---

## Important interview point

Do not answer:

> "A warp has 32 threads because 32 is the fastest."

That's incomplete.

Better answer:

> **A warp consists of 32 threads because NVIDIA defines 32 threads as the fundamental SIMT execution group for its GPU architecture. This allows hardware to efficiently schedule and execute threads together.**

---

## Why should developers care?

Because warp-level behavior directly affects performance.

For example:

```text
32 threads
    │
    ▼
Same instruction
    │
    ▼
Efficient execution
```

But if threads diverge:

```text
Some threads → Path A
Some threads → Path B
```

efficiency can decrease.

---

# 11. What is warp divergence?

Warp divergence occurs when threads within the same warp follow different control-flow paths.

Example:

```cpp
if (idx % 2 == 0)
{
    A[idx] += 1;
}
else
{
    A[idx] *= 2;
}
```

Suppose a warp contains:

```text
Thread 0 → even
Thread 1 → odd
Thread 2 → even
Thread 3 → odd
...
```

The warp now has two execution paths.

Conceptually:

```text
                   Warp
                     │
             ┌───────┴───────┐
             ▼               ▼
          Path A           Path B
       Threads even     Threads odd
             │               │
             └───────┬───────┘
                     ▼
                   Merge
```

The hardware may need to execute the different paths separately while masking inactive threads.

---

## Why is this bad?

Suppose:

```text
16 threads → Path A
16 threads → Path B
```

The GPU may execute:

```text
Path A
  ↓
16 active threads

Path B
  ↓
16 active threads
```

rather than having all 32 threads perform useful work simultaneously.

This reduces efficiency.

---

## Important

Warp divergence does not mean:

> "The program is incorrect."

It means:

> **The execution may be less efficient.**

---

# 12. How does a GPU hide memory latency?

Memory access can be slow compared with computation.

Suppose:

```text
Warp 0
  ↓
Waiting for memory
```

A GPU can maintain many other warps that are ready to execute.

Conceptually:

```text
                         SM
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
       Warp 0          Warp 1          Warp 2
       WAITING          READY           READY
                          │               │
                          └───────┬───────┘
                                  ▼
                              Execute
```

When Warp 0 is waiting, the scheduler can issue instructions from another ready warp.

---

## This is called latency hiding

The GPU does not necessarily make memory latency disappear.

Instead:

> **It uses massive parallelism to perform useful work while some threads are waiting.**

This is a core GPU design principle.

---

## CPU vs GPU

CPU:

```text
Optimize the execution
of a small number of threads.
```

GPU:

```text
Maintain many threads in flight
and switch between ready warps
to hide latency.
```

---

# 13. Why is matrix multiplication GPU-friendly?

Consider:

```text
C = A × B
```

For each output element:

```text
C[i][j]
```

we perform a dot product:

```text
C[i][j] =
A[i][0]B[0][j]
+
A[i][1]B[1][j]
+
...
```

Different output elements can often be calculated independently.

For example:

```text
C[0][0] → Work 1
C[0][1] → Work 2
C[0][2] → Work 3
C[1][0] → Work 4
C[1][1] → Work 5
...
```

This creates enormous parallelism.

---

## Matrix multiplication also has high arithmetic intensity

A well-optimized matrix multiplication can perform a large amount of computation on data loaded into the GPU.

Therefore it can effectively use:

```text
CUDA Cores
Tensor Cores
Shared Memory
Registers
GPU Memory Bandwidth
```

Modern NVIDIA GPUs have specialized Tensor Cores that accelerate matrix operations.

This is one of the primary reasons matrix multiplication is central to GPU-accelerated deep learning.

---

# 14. What is a compute-bound workload?

A workload is **compute-bound** when its performance is primarily limited by the GPU's ability to perform computation.

Conceptually:

```text
                 GPU
                  │
                  ▼
           Compute Units
                  │
                  ▼
             Heavily Busy
```

The workload has enough data available, but the GPU spends most of its time performing arithmetic.

Example:

```text
Large Matrix Multiplication
```

Simplified:

```text
Memory
   │
   ▼
Load Data
   │
   ▼
████████████████████
GPU COMPUTATION
████████████████████
   │
   ▼
Result
```

---

## Typical optimization direction

For compute-bound workloads, investigate:

* Tensor Cores
* better numerical precision
* instruction efficiency
* kernel optimization
* parallelism
* occupancy
* operation fusion

---

# 15. What is a memory-bound workload?

A workload is **memory-bound** when performance is primarily limited by data movement.

Example:

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

The computation is simple.

Memory traffic can dominate.

Conceptually:

```text
GPU
 │
 ▼
Request Data
 │
 ▼
WAIT
 │
 ▼
Data arrives
 │
 ▼
Simple computation
```

---

## Typical optimization direction

For memory-bound workloads, investigate:

* memory coalescing
* cache behavior
* shared memory
* data reuse
* memory layout
* reducing memory transfers
* kernel fusion
* lower precision
* avoiding unnecessary reads/writes

---

# 16. Why doesn't a GPU simply execute every thread independently?

Because GPU hardware is designed around **grouped execution** to achieve very high throughput efficiently.

If every thread had completely independent control hardware, the GPU would need much more hardware dedicated to:

```text
Instruction Fetch
Instruction Decode
Scheduling
Control Logic
```

That would increase hardware cost and reduce the amount of silicon available for arithmetic throughput.

Instead, NVIDIA groups threads into warps.

```text
32 Threads
    │
    ▼
  Warp
    │
    ▼
Shared instruction execution
```

This allows the GPU to dedicate more resources to:

```text
Arithmetic
Parallel execution
Memory throughput
```

---

## Simplified design trade-off

```text
CPU

More complex control
        +
Fewer powerful execution units


GPU

Simpler control per thread
        +
Massive parallel execution
```

This is one of the fundamental architectural trade-offs.

---

# 17. What is an SM?

SM stands for:

> **Streaming Multiprocessor**

An SM is a major execution unit inside an NVIDIA GPU.

A simplified SM:

```text
                       SM
                        │
       ┌────────────────┼────────────────┐
       │                │                │
       ▼                ▼                ▼
   CUDA Cores      Tensor Cores      Warp Scheduler
       │                │                │
       └────────────────┼────────────────┘
                        │
                ┌───────┴────────┐
                ▼                ▼
            Registers      Shared Memory
                │                │
                └───────┬────────┘
                        ▼
                     L1 Cache
```

The exact hardware organization varies across NVIDIA GPU architectures, so this is a conceptual model rather than a literal block diagram of every generation.

---

## What happens inside an SM?

The SM manages execution of many threads.

Those threads are organized into:

```text
Threads
   ↓
Warps
   ↓
Scheduled on SM
```

The SM contains resources needed to execute those threads, including:

* registers
* shared memory
* execution units
* warp schedulers
* load/store resources

---

# 18. What is the relationship between SM → Warp → Thread?

This hierarchy is fundamental.

```text
GPU
 │
 ├── SM 0
 │    │
 │    ├── Warp 0
 │    │    ├── Thread 0
 │    │    ├── Thread 1
 │    │    ├── ...
 │    │    └── Thread 31
 │    │
 │    ├── Warp 1
 │    │    └── 32 Threads
 │    │
 │    └── ...
 │
 ├── SM 1
 │    └── ...
 │
 └── SM N
```

Think about the hierarchy as:

```text
GPU
 ↓
SM
 ↓
Warp
 ↓
Thread
```

---

## Thread

A thread is the smallest logical execution context in CUDA.

For example:

```text
Thread 0 → A[0] + B[0]
Thread 1 → A[1] + B[1]
```

---

## Warp

A warp contains:

```text
32 threads
```

These threads are scheduled/executed together according to NVIDIA's SIMT execution model.

---

## SM

An SM executes and manages many active warps.

Therefore:

```text
SM
 │
 ├── Warp 0
 ├── Warp 1
 ├── Warp 2
 ├── Warp 3
 └── ...
```

The number of active warps depends on the GPU architecture and resource usage.

---

# 19. Why can a CPU outperform a GPU on some workloads?

Because GPUs are not universally faster.

A CPU can outperform a GPU when the workload has characteristics such as:

### 1. Very small workload

If there is not enough parallel work:

```text
GPU capacity
████████████████████

Actual work
██
```

Most GPU resources remain unused.

---

### 2. Highly sequential computation

If:

```text
Task B depends on Task A
Task C depends on Task B
Task D depends on Task C
```

there is limited parallelism.

```text
A
 ↓
B
 ↓
C
 ↓
D
```

A GPU has fewer opportunities to exploit massive parallelism.

---

### 3. Irregular control flow

Example:

```python
if condition_a:
    ...
elif condition_b:
    ...
elif condition_c:
    ...
```

Highly irregular branching can reduce GPU efficiency.

---

### 4. CPU-GPU transfer overhead

Suppose:

```text
CPU
 │
 │ transfer
 ▼
GPU
 │
 │ compute
 ▼
CPU
```

If the computation itself is tiny, data-transfer overhead may dominate.

---

### 5. Workload is latency-sensitive

For a small request:

```text
Launch GPU work
      +
Transfer data
      +
Synchronize
```

may take longer than simply executing the operation on the CPU.

---

### 6. Poor GPU utilization

A GPU is most effective when enough parallel work exists to keep its execution resources busy.

---

# 20. Why can increasing GPU compute resources fail to improve application performance?

This is one of the most important GPU engineering questions.

Suppose we double compute resources:

```text
Before:

Compute Capacity
██████████


After:

Compute Capacity
████████████████████
```

You might expect:

```text
2× performance
```

But this only happens if computation is the bottleneck.

---

## Case 1 — Memory-bound workload

Suppose:

```text
Compute:
████

Memory:
████████████████████
```

Adding compute capacity doesn't solve the memory bottleneck.

---

## Case 2 — Insufficient parallelism

Suppose the workload is tiny:

```text
GPU capacity
████████████████████

Workload
██
```

Additional compute resources remain unused.

---

## Case 3 — Kernel launch overhead

If the application spends a significant amount of time launching kernels:

```text
Launch
  ↓
Compute
  ↓
Launch
  ↓
Compute
  ↓
Launch
```

Increasing GPU arithmetic capacity may not significantly help.

---

## Case 4 — Synchronization

If threads frequently wait:

```text
Compute
  ↓
Synchronize
  ↓
Compute
  ↓
Synchronize
```

additional compute resources may remain idle.

---

## Case 5 — Memory latency

The GPU may spend significant time waiting for data.

```text
Compute
  ↓
Memory request
  ↓
WAIT
  ↓
Data
  ↓
Compute
```

More compute units don't automatically reduce memory latency.

---

## Case 6 — Poor memory access

If threads access memory inefficiently:

```text
Thread 0 → random address
Thread 1 → random address
Thread 2 → random address
Thread 3 → random address
```

memory throughput can become the bottleneck.

---

## Case 7 — Inter-GPU communication

For multi-GPU systems:

```text
GPU 0
 │
 │ NCCL communication
 ▼
GPU 1
```

performance can become limited by:

* PCIe
* NVLink
* NCCL communication
* synchronization
* collective operations

rather than computation.

---

# 21. The General GPU Performance Model

The most useful mental model is:

```text
                       Application
                           │
                           ▼
                    GPU Workload
                           │
            ┌──────────────┼──────────────┐
            │              │              │
         Compute         Memory       Parallelism
            │              │              │
          FLOPS         Bandwidth       Threads
        Tensor Cores      Cache          Warps
        CUDA Cores      Latency          Blocks
            │              │              │
            └──────────────┼──────────────┘
                           │
                           ▼
                    Actual Performance
```

But there are additional factors:

```text
Actual Performance
        │
        ├── Compute
        ├── Memory
        ├── Parallelism
        ├── Occupancy
        ├── Warp efficiency
        ├── Kernel launch overhead
        ├── Synchronization
        ├── CPU-GPU transfers
        └── GPU-GPU communication
```

---

# 22. CPU vs GPU — Complete Comparison

| Feature                       | CPU                        | GPU                                                    |
| ----------------------------- | -------------------------- | ------------------------------------------------------ |
| Primary goal                  | Low latency                | High throughput                                        |
| Core count                    | Relatively low             | Very high parallel execution resources                 |
| Core complexity               | High                       | More specialized/simple per thread                     |
| Control flow                  | Excellent                  | Less efficient with divergence                         |
| Sequential workloads          | Excellent                  | Poor fit                                               |
| Parallel workloads            | Good                       | Excellent                                              |
| Data-level parallelism        | SIMD                       | SIMT                                                   |
| Instruction-level parallelism | Very important             | Also exists, but massive thread parallelism is central |
| Memory bandwidth              | Lower                      | Very high                                              |
| Cache strategy                | Large sophisticated caches | High-throughput memory hierarchy                       |
| Branch prediction             | Sophisticated              | Different execution model                              |
| Matrix multiplication         | Good                       | Excellent                                              |
| Deep learning                 | Good                       | Excellent                                              |
| Small workloads               | Often better               | May have overhead                                      |
| Large tensor workloads        | Good                       | Excellent                                              |
| Multi-threading               | Limited compared with GPU  | Massive                                                |

---

# 23. The Most Important Mental Model

Do not memorize:

```text
CPU = Fast
GPU = Faster
```

Instead remember:

```text
CPU
 │
 ├── Few powerful cores
 ├── Complex control logic
 ├── Strong branch handling
 ├── Large caches
 ├── Low latency
 └── General-purpose workloads


GPU
 │
 ├── Many SMs
 ├── Massive thread parallelism
 ├── Warps
 ├── High memory bandwidth
 ├── High throughput
 └── Tensor/matrix workloads
```

---

# 24. Execution Model

The complete conceptual flow is:

```text
                CPU
                 │
                 │ Launch work
                 ▼
              CUDA
                 │
                 ▼
                GPU
                 │
                 ▼
                Grid
                 │
        ┌────────┴────────┐
        ▼                 ▼
      Block             Block
        │                 │
        ▼                 ▼
      Warps             Warps
        │                 │
        ▼                 ▼
     Threads           Threads
        │                 │
        └────────┬────────┘
                 ▼
              Compute
```

This hierarchy is the foundation for everything that comes later.

---

# 🎯 Interview Quick Answers

| #  | Question                             | Short Answer                                                                                                                    |
| -- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------- |
| 1  | CPU vs GPU architecture?             | CPU uses fewer complex cores for low-latency general workloads; GPU uses many parallel execution resources for high throughput. |
| 2  | Why throughput?                      | GPUs are designed to keep many execution resources busy processing large parallel workloads.                                    |
| 3  | What is ILP?                         | Executing independent instructions concurrently or in an overlapping manner.                                                    |
| 4  | What is DLP?                         | Performing the same operation on multiple data elements simultaneously.                                                         |
| 5  | SIMD?                                | Single Instruction, Multiple Data.                                                                                              |
| 6  | SIMT?                                | Single Instruction, Multiple Threads.                                                                                           |
| 7  | SIMD vs SIMT?                        | SIMD exposes vector/data lanes; SIMT exposes threads grouped into GPU warps.                                                    |
| 8  | Why SIMT?                            | It provides a scalable thread-based programming model while allowing hardware to execute threads efficiently in groups.         |
| 9  | Warp?                                | A group of 32 CUDA threads scheduled/executed together by NVIDIA GPU hardware.                                                  |
| 10 | Why 32 threads?                      | NVIDIA defines 32 as the fundamental warp width for CUDA GPU architectures.                                                     |
| 11 | Warp divergence?                     | When threads in the same warp follow different control-flow paths.                                                              |
| 12 | How hide memory latency?             | By maintaining many active warps and scheduling ready warps while others wait for memory.                                       |
| 13 | Why is MatMul GPU-friendly?          | It contains massive amounts of parallel multiply-add computation and can efficiently use GPU/Tensor Core resources.             |
| 14 | Compute-bound?                       | Performance is primarily limited by computational throughput.                                                                   |
| 15 | Memory-bound?                        | Performance is primarily limited by memory/data movement.                                                                       |
| 16 | Why not independent threads?         | Grouped execution reduces control hardware overhead and allows more hardware resources to be dedicated to parallel computation. |
| 17 | SM?                                  | Streaming Multiprocessor, a major execution unit that manages and executes many GPU threads/warps.                              |
| 18 | SM → Warp → Thread?                  | An SM manages many warps; each warp contains 32 threads; threads are the logical execution contexts.                            |
| 19 | CPU faster sometimes?                | Small, sequential, branch-heavy or transfer-heavy workloads may not utilize the GPU efficiently.                                |
| 20 | Why more GPU resources may not help? | The bottleneck may be memory, synchronization, communication, insufficient parallelism, or kernel overhead rather than compute. |

---

# 🔑 Key Takeaways

## 1. CPU and GPU have different design goals

```text
CPU → Low Latency
GPU → High Throughput
```

---

## 2. GPU performance depends on parallelism

```text
More parallel work
       ↓
More GPU utilization
       ↓
Better throughput
```

---

## 3. SIMD and SIMT are different abstractions

```text
CPU → SIMD → Vector/Data lanes

GPU → SIMT → Threads/Warp
```

---

## 4. Warp is fundamental

```text
32 Threads
     ↓
   Warp
```

Understanding warps is essential for understanding:

* divergence
* memory access
* synchronization
* occupancy
* warp-level operations

---

## 5. SM is the major GPU execution unit

```text
GPU
 ↓
Many SMs
 ↓
Many Warps
 ↓
Many Threads
```

---

## 6. GPU hides latency using parallelism

Instead of making one thread extremely fast:

```text
GPU maintains many threads/warps
              ↓
Some wait
              ↓
Others execute
```

---

## 7. GPU optimization starts with the bottleneck

Always ask:

```text
Is it:

Compute-bound?
Memory-bound?
Latency-bound?
Synchronization-bound?
Communication-bound?
```

Only after identifying the bottleneck should we optimize.

---

# 🧠 Final Mental Model

The entire Lesson 0.2 can be represented as:

```text
                         CPU
                          │
             ┌────────────┴────────────┐
             │                         │
       Few powerful cores        Complex control
             │                         │
             └────────────┬────────────┘
                          │
                          ▼
                     Low Latency


                         GPU
                          │
             ┌────────────┴────────────┐
             │                         │
          Many SMs              High Memory BW
             │
             ▼
           Warps
             │
             ▼
          Threads
             │
             ▼
        Massive Parallelism
             │
             ▼
        High Throughput
```

And the key software/hardware execution relationship is:

```text
Application
    │
    ▼
PyTorch / CUDA
    │
    ▼
CUDA Kernel
    │
    ▼
GPU Grid
    │
    ▼
Blocks
    │
    ▼
Warps
    │
    ▼
Threads
    │
    ▼
SM Execution
```

This is the foundation for the next stage.

---

# 📌 Next Lesson

## Lesson 0.3 — Why GPUs Are Good for Deep Learning

We will now connect the architecture directly to **Machine Learning and Transformers**:

```text
Neural Network
      │
      ▼
Tensor Operations
      │
      ├── Matrix Multiplication
      ├── Vector Operations
      ├── Convolution
      ├── Attention
      └── Element-wise Operations
      │
      ▼
Data-Level Parallelism
      │
      ▼
GPU
```

We will also introduce:

* Why training is GPU-heavy
* Why inference is GPU-heavy
* Forward pass
* Backpropagation
* Matrix multiplication in neural networks
* Tensor operations
* GPU utilization during training vs inference
* Why LLMs require large GPU memory
* Why Tensor Cores matter for Transformers
* Why batch size affects GPU utilization
* Why LLM inference has **prefill vs decode** behavior
* How all of this connects to **CUDA and modern LLM inference engines**
