# Phase 00 — GPU Fundamentals

# Lesson 0.3 — Why GPUs Are Good for Deep Learning

## 1. Core Idea

> **Deep learning is dominated by large-scale tensor operations that expose massive data-level parallelism, which GPUs are designed to exploit.**

The fundamental relationship is:

```text
Deep Learning
      ↓
Tensor Operations
      ↓
Massive Parallelism
      ↓
GPU
      ↓
CUDA Kernels
      ↓
CUDA Cores + Tensor Cores
```

---

# 2. Why Deep Learning Fits GPUs

Neural networks perform large numbers of:

* Matrix multiplications
* Vector operations
* Element-wise operations
* Convolutions
* Reductions
* Normalization
* Softmax
* Attention
* Embedding operations

Many of these operations can execute independently across large numbers of elements.

Example:

```text
C = A + B

C[0] = A[0] + B[0]
C[1] = A[1] + B[1]
C[2] = A[2] + B[2]
...
C[N] = A[N] + B[N]
```

Each element can be processed independently.

Therefore:

```text
Large Tensor
     ↓
Many Independent Operations
     ↓
Many GPU Threads
     ↓
High Throughput
```

---

# 3. Matrix Multiplication

Matrix multiplication is one of the most important operations in deep learning.

For:

```text
A = M × K

B = K × N

C = A × B

C = M × N
```

The computational complexity is approximately:

$$
O(MKN)
$$

Each output element requires computation:

$$
C_{ij} = \sum_k A_{ik}B_{kj}
$$

There are many independent output elements:

```text
C[0][0] ─┐
C[0][1] ─┤
C[0][2] ─┤
C[0][3] ─┤──► Parallel computation
C[1][0] ─┤
C[1][1] ─┤
...       │
C[M][N] ─┘
```

This makes matrix multiplication highly suitable for GPUs.

---

# 4. Neural Network Computation

A simple neural network layer:

$$
Y = XW + b
$$

Typical flow:

```text
Input
  ↓
Matrix Multiplication
  ↓
Bias Addition
  ↓
Activation
  ↓
Matrix Multiplication
  ↓
Output
```

A simple network may perform:

$$
H = XW_1 + b_1
$$

$$
A = ReLU(H)
$$

$$
Y = AW_2 + b_2
$$

The matrix multiplications are computationally significant and highly parallelizable.

---

# 5. Training vs Inference

## Inference

```text
Input
  ↓
Forward Pass
  ↓
Output
```

## Training

```text
Input
  ↓
Forward Pass
  ↓
Prediction
  ↓
Loss
  ↓
Backward Pass
  ↓
Gradients
  ↓
Weight Update
```

Training requires:

* Forward computation
* Backward computation
* Gradient computation
* Parameter updates

Therefore training generally requires substantially more computation and memory than inference.

---

# 6. Forward Pass

For a layer:

$$
Y = XW
$$

The GPU performs the matrix operations required to calculate the output.

Example:

```text
X
│
▼
Matrix Multiply
│
▼
W
│
▼
Y
```

---

# 7. Backward Pass

Training calculates gradients such as:

$$
\frac{\partial L}{\partial W}
$$

and

$$
\frac{\partial L}{\partial X}
$$

These calculations involve additional tensor and matrix operations.

Therefore:

```text
Training
   │
   ├── Forward
   │
   ├── Loss
   │
   ├── Backward
   │
   └── Weight Update
```

---

# 8. Why GPU Is Not Always Faster

GPU does **not** mean automatically faster.

GPU is advantageous when:

```text
Large workload
+
High parallelism
+
Enough computation
```

A tiny operation may be faster on CPU because GPU execution introduces overhead:

```text
CPU
 ↓
GPU Memory Transfer
 ↓
Kernel Launch
 ↓
GPU Computation
 ↓
GPU → CPU Transfer
```

For a very small workload, this overhead may dominate the actual computation.

---

# 9. Batch Size

Batch size determines how many samples are processed together.

### Small batch

```text
Sample 1
   ↓
 GPU
```

GPU may have insufficient work.

### Large batch

```text
Sample 1 ─┐
Sample 2 ─┤
Sample 3 ─┤
Sample 4 ─┤
   ...    ├──► GPU
Sample N ─┘
```

Larger batches can provide more parallelism.

Generally:

```text
Batch Size ↑
     │
     ├── GPU utilization ↑
     ├── Throughput ↑
     ├── Memory usage ↑
     └── Latency may ↑
```

The exact relationship depends on the workload and serving system.

---

# 10. Tensor Cores

Modern NVIDIA GPUs contain specialized hardware called:

> **Tensor Cores**

Tensor Cores are designed to accelerate matrix multiply-accumulate operations used heavily by deep learning.

Conceptually:

```text
NVIDIA GPU
│
├── CUDA Cores
│     └── General-purpose GPU computation
│
└── Tensor Cores
      └── Specialized matrix operations
```

Tensor Cores are extremely important for modern AI workloads.

---

# 11. CUDA Cores vs Tensor Cores

| CUDA Cores                            | Tensor Cores                          |
| ------------------------------------- | ------------------------------------- |
| General-purpose GPU arithmetic        | Specialized matrix operations         |
| Flexible                              | More specialized                      |
| Used by many CUDA kernels             | Optimized for AI/matrix workloads     |
| Important for general GPU computation | Extremely important for deep learning |

Do not think:

```text
Tensor Core = better CUDA Core
```

They are specialized for different workloads.

---

# 12. Numerical Precision

Common numerical formats in modern AI:

```text
FP32
FP16
BF16
TF32
FP8
```

Approximate bytes per value:

| Format | Bytes |
| ------ | ----: |
| FP32   |     4 |
| FP16   |     2 |
| BF16   |     2 |
| FP8    |     1 |

Lower precision can provide:

* Lower memory usage
* Lower memory bandwidth requirements
* Higher throughput
* Better Tensor Core utilization

But precision trade-offs must be considered carefully.

Detailed numerical formats will be covered later in:

```text
Phase 09 — Tensor Cores & Numerical Formats
```

---

# 13. Transformer Workload

Transformers contain many GPU-friendly operations.

For example:

```text
Input
  ↓
Embedding
  ↓
Q/K/V Projections
  ↓
Attention
  ↓
Output Projection
  ↓
MLP
  ↓
Normalization
  ↓
Next Transformer Layer
```

Major operations include:

```text
X × WQ
X × WK
X × WV
Q × Kᵀ
Softmax
Attention × V
X × W1
Activation
X × W2
```

A large portion of Transformer computation is tensor/matrix computation.

---

# 14. Attention

The core attention equation:

$$
Attention(Q,K,V)
=
softmax
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
$$

It contains:

```text
Q × Kᵀ
   ↓
Scaling
   ↓
Softmax
   ↓
Attention × V
```

Important:

* Matrix multiplication → highly parallel
* Element-wise operations → parallel
* Softmax → parallel + reduction
* Attention × V → matrix multiplication

Therefore attention maps well to GPU execution, although particular kernels can become memory-bound depending on implementation and workload shape.

---

# 15. Transformer MLP

A simplified Transformer MLP:

$$
Y = W_2 \sigma(W_1X)
$$

Flow:

```text
X
 ↓
W1
 ↓
Activation
 ↓
W2
 ↓
Y
```

Again, the major operations are matrix multiplications.

---

# 16. GPU Memory Is as Important as GPU Compute

A GPU has two major resources we need to think about:

```text
GPU
│
├── Compute
│
└── Memory
```

High compute capability alone is not sufficient.

A workload can be limited by:

* Compute throughput
* Memory bandwidth
* Memory capacity
* Synchronization
* Kernel launch overhead
* Communication

---

# 17. LLM GPU Memory

During LLM inference, GPU memory is used by:

```text
GPU Memory
│
├── Model Weights
├── KV Cache
├── Activations
├── Temporary Buffers
└── Runtime / CUDA Workspace
```

During training, additional memory is needed for:

```text
GPU Memory
│
├── Model Weights
├── Gradients
├── Activations
├── Optimizer States
└── Temporary Buffers
```

---

# 18. Model Weight Memory

Basic approximation:

$$
Weight\ Memory
\approx
Number\ of\ Parameters
\times
Bytes\ per\ Parameter
$$

Example:

```text
7B parameters
FP16
```

Approximately:

$$
7B \times 2
\approx 14GB
$$

This is only the model weights.

Actual GPU memory requirements are higher because of:

* KV cache
* Activations
* Runtime buffers
* CUDA workspace
* Batch size
* Context length

---

# 19. KV Cache

During autoregressive LLM generation, previously computed Key/Value states are stored.

```text
GPU Memory
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

As:

```text
Sequence Length ↑
Concurrent Requests ↑
```

KV-cache memory generally increases.

This makes KV-cache management a major part of production LLM serving.

---

# 20. Prefill vs Decode

LLM inference has two important stages.

## Prefill

The prompt is processed.

```text
Prompt
 ↓
Token1 Token2 Token3 ... TokenN
 ↓
Parallel processing
```

Prefill is often relatively compute-intensive.

## Decode

The model generates tokens autoregressively:

```text
Token N+1
   ↓
Token N+2
   ↓
Token N+3
   ↓
...
```

Decode can be strongly influenced by:

* Memory bandwidth
* KV-cache access
* Batch size
* Kernel launch overhead
* Scheduling
* Context length

The exact bottleneck depends on the model, hardware, implementation, batching, and precision.

---

# 21. Compute-Bound vs Memory-Bound

## Compute-Bound

The workload spends most of its time performing arithmetic.

```text
Data
 ↓
Compute
 ↓
Compute
 ↓
Compute
```

Example:

```text
Large GEMM
```

## Memory-Bound

The workload is limited by data movement.

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

Examples can include:

* Element-wise operations
* Some reductions
* Some normalization operations
* Some LLM decode workloads

The exact classification depends on workload and implementation.

---

# 22. Arithmetic Intensity

Arithmetic intensity measures computation relative to memory traffic:

$$
Arithmetic\ Intensity
=
\frac{FLOPs}{Bytes\ Transferred}
$$

High arithmetic intensity:

```text
Lots of computation
+
Relatively less data movement
```

Low arithmetic intensity:

```text
Lots of data movement
+
Relatively little computation
```

This concept is fundamental to GPU performance engineering.

---

# 23. GPU Utilization ≠ Performance

A low GPU utilization number does not automatically explain why an application is slow.

Possible bottlenecks:

```text
CPU preprocessing
       ↓
Data transfer
       ↓
Kernel launch
       ↓
GPU computation
       ↓
Memory access
       ↓
Synchronization
       ↓
Communication
```

Therefore:

> **GPU utilization alone is not sufficient to diagnose performance.**

Use profiling tools later to identify the actual bottleneck.

---

# 24. Why LLM Serving Needs GPU-Aware Software

Production LLM serving must optimize:

```text
GPU Compute
+
GPU Memory
+
KV Cache
+
Batching
+
Scheduling
+
Kernel Efficiency
+
Memory Access
```

This is why systems such as:

* vLLM
* SGLang
* TensorRT-LLM

exist.

Conceptually:

```text
Users
  ↓
LLM Server
  ↓
Scheduler
  ↓
Batching
  ↓
CUDA Kernels
  ↓
Tensor Cores / CUDA Cores
  ↓
GPU Memory
```

---

# 25. Continuous Batching

LLM requests have different arrival and completion times.

Instead of waiting for a fixed batch:

```text
Request A ─────────────
Request B ──────
Request C ────────────────
Request D ────────
```

The serving engine dynamically manages active requests.

```text
Requests
   ↓
Scheduler
   ↓
Continuous Batching
   ↓
GPU
```

This helps improve GPU utilization and serving throughput.

---

# 26. The GPU Performance Mental Model

Whenever you analyze a GPU workload, ask:

### 1. How much parallelism exists?

```text
Can thousands of operations run concurrently?
```

### 2. How much computation exists?

```text
Are there enough FLOPs to keep the GPU busy?
```

### 3. How much data movement exists?

```text
How much data must move between memory and compute?
```

### 4. What is the bottleneck?

```text
Compute?
Memory bandwidth?
Memory capacity?
Latency?
Synchronization?
Communication?
```

This mental model is more important than simply memorizing GPU specifications.

---

# 27. Complete Mental Model

```text
                 Deep Learning
                       │
                       ▼
                Tensor Operations
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        GEMM        Attention    Elementwise
          │            │            │
          └────────────┼────────────┘
                       ▼
                Massive Parallelism
                       │
                       ▼
                      GPU
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
        CUDA Cores          Tensor Cores
             │                   │
             └─────────┬─────────┘
                       ▼
                 CUDA Kernels
                       │
                       ▼
                  GPU Memory
                       │
                       ▼
             High AI Throughput
```

---

# 28. Key Takeaways

Remember these 15 points:

1. **Deep learning is tensor-heavy.**
2. **Matrix multiplication is one of the dominant operations.**
3. **Tensor operations expose massive data-level parallelism.**
4. **GPUs are designed to exploit massive parallelism.**
5. **Training requires forward and backward computation.**
6. **Large batch sizes can improve GPU utilization and throughput.**
7. **Tensor Cores accelerate supported matrix operations.**
8. **Mixed precision reduces memory usage and can increase throughput.**
9. **GPU performance depends on both compute and memory.**
10. **LLMs require substantial GPU memory for weights and KV cache.**
11. **KV-cache memory grows with sequence length and concurrency.**
12. **Prefill and decode have different performance characteristics.**
13. **A workload can be compute-bound or memory-bound.**
14. **GPU utilization alone does not explain application performance.**
15. **Production LLM serving requires GPU-aware scheduling, batching, memory management, and optimized kernels.**

---

# 29. Interview Questions

You should be able to answer these before moving forward:

1. Why are GPUs good for deep learning?
2. Why is matrix multiplication so important in neural networks?
3. Why is matrix multiplication highly parallelizable?
4. Why isn't a GPU always faster than a CPU?
5. What happens when batch size increases?
6. What are Tensor Cores?
7. CUDA Core vs Tensor Core?
8. Why does mixed precision improve AI performance?
9. What is compute-bound vs memory-bound?
10. What is arithmetic intensity?
11. Why can GPU utilization be low even when an application is slow?
12. What consumes GPU memory during LLM inference?
13. What is the KV cache?
14. Why are prefill and decode different?
15. Why is continuous batching important?
16. Why do vLLM and SGLang need sophisticated GPU/memory management?
17. Why doesn't increasing GPU compute always improve application performance?
18. What is the relationship between model parameters and GPU memory?
19. Why does context length affect LLM GPU memory usage?
20. What are the major GPU bottlenecks in production LLM inference?

---

# 30. One-Line Interview Answer

If asked:

> **"Why are GPUs good for deep learning?"**

A strong answer is:

> GPUs are well suited for deep learning because neural networks perform large amounts of regular tensor computation, especially matrix multiplications, that expose massive data-level parallelism. GPUs execute these workloads across many parallel threads and specialized hardware such as Tensor Cores, while high memory bandwidth helps feed the computation efficiently. However, actual performance depends on factors such as arithmetic intensity, memory bandwidth, batch size, kernel efficiency, synchronization, and communication.
