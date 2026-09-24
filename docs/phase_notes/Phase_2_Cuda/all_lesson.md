# Phase 02 — CUDA Programming

> Goal: Understand how CUDA programs execute on the GPU and build the foundation required for writing and optimizing real CUDA kernels.

---

# 2.1 CUDA Programming Model

CUDA provides a programming model for executing functions on NVIDIA GPUs.

The basic flow is:

```text
CPU / Host
    │
    │ launch kernel
    ▼
GPU / Device
    │
    ├── Grid
    │    ├── Block
    │    │    ├── Warp
    │    │    │    └── Threads
    │    │    └── ...
    │    └── ...
    │
    └── Execute kernel
```

Important terminology:

| Term   | Meaning                   |
| ------ | ------------------------- |
| Host   | CPU                       |
| Device | GPU                       |
| Kernel | Function executed on GPU  |
| Thread | Smallest execution unit   |
| Block  | Group of threads          |
| Grid   | Collection of blocks      |
| Warp   | 32 threads on NVIDIA GPUs |
| SM     | Streaming Multiprocessor  |

---

# 2.2 CUDA Kernel

A CUDA kernel is a function executed by many GPU threads in parallel.

Conceptually:

```cpp
__global__ void vector_add(...)
{
    // GPU code
}
```

The CPU launches it:

```cpp
vector_add<<<blocks, threads>>>(...);
```

The important difference:

```text
Normal function
    ↓
Executed once

CUDA kernel
    ↓
Executed by thousands/millions of threads
```

For example:

```text
A = [1, 2, 3, 4]

B = [5, 6, 7, 8]

        ↓

C = [6, 8, 10, 12]
```

Each GPU thread can process one element:

```text
Thread 0 → C[0]
Thread 1 → C[1]
Thread 2 → C[2]
Thread 3 → C[3]
```

---

# 2.3 Thread Indexing

The most important CUDA concept is determining:

> Which data element should this thread process?

CUDA provides:

```cpp
threadIdx
blockIdx
blockDim
gridDim
```

For a 1D kernel:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

Meaning:

```text
Global Thread ID
        =
Block ID × Threads per Block
        +
Thread ID inside Block
```

Example:

```text
blockDim.x = 4

Block 0:
threads → 0 1 2 3

Block 1:
threads → 4 5 6 7

Block 2:
threads → 8 9 10 11
```

Therefore:

```cpp
idx = blockIdx.x * blockDim.x + threadIdx.x;
```

---

# 2.4 1D, 2D and 3D Thread Layouts

CUDA supports multidimensional grids and blocks.

## 1D

Useful for:

```text
Vector operations
Token operations
Elementwise operations
```

Example:

```cpp
threadIdx.x
```

---

## 2D

Useful for:

```text
Matrices
Images
Feature maps
```

Example:

```cpp
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

---

## 3D

Useful for:

```text
3D tensors
Volumetric data
Scientific computing
```

The important idea:

```text
Thread indexing
      ↓
Maps GPU threads
      ↓
to data elements
```

---

# 2.5 Vector Addition

One of the simplest CUDA kernels:

```cpp
__global__
void vector_add(
    const float* A,
    const float* B,
    float* C,
    int N
)
{
    int idx =
        blockIdx.x * blockDim.x +
        threadIdx.x;

    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}
```

Launch:

```cpp
int threads = 256;

int blocks =
    (N + threads - 1) / threads;

vector_add<<<blocks, threads>>>(A, B, C, N);
```

Why:

```cpp
(N + threads - 1) / threads
```

?

Because `N` may not be perfectly divisible by the block size.

The bounds check:

```cpp
if (idx < N)
```

prevents out-of-bounds memory access.

---

# 2.6 Host and Device Memory

CPU and GPU have separate memory spaces in the traditional CUDA model.

```text
CPU
 │
 │ Host Memory
 │
 └──────────────┐
                │ PCIe / NVLink
                ▼
             GPU
                │
                │ Device Memory
                ▼
              HBM/GDDR
```

Typical flow:

```text
CPU data
   ↓
cudaMemcpy
   ↓
GPU memory
   ↓
CUDA kernel
   ↓
GPU memory
   ↓
cudaMemcpy
   ↓
CPU data
```

---

# 2.7 cudaMalloc

GPU memory can be allocated using:

```cpp
float* d_A;

cudaMalloc(
    &d_A,
    N * sizeof(float)
);
```

Naming convention:

```text
h_A → host pointer

d_A → device pointer
```

Example:

```cpp
float* h_A;
float* d_A;
```

This convention makes CUDA code much easier to understand.

---

# 2.8 cudaMemcpy

Copy CPU → GPU:

```cpp
cudaMemcpy(
    d_A,
    h_A,
    size,
    cudaMemcpyHostToDevice
);
```

Copy GPU → CPU:

```cpp
cudaMemcpy(
    h_A,
    d_A,
    size,
    cudaMemcpyDeviceToHost
);
```

Direction:

```text
CPU → GPU

cudaMemcpyHostToDevice


GPU → CPU

cudaMemcpyDeviceToHost
```

---

# 2.9 Complete CUDA Program Flow

A typical CUDA program:

```text
1. Allocate host memory
        ↓
2. Initialize data
        ↓
3. Allocate GPU memory
        ↓
4. Copy CPU → GPU
        ↓
5. Launch kernel
        ↓
6. Synchronize/check errors
        ↓
7. Copy GPU → CPU
        ↓
8. Validate result
        ↓
9. Free GPU memory
```

Pseudo-code:

```cpp
allocate_host();

initialize_data();

cudaMalloc();

cudaMemcpy(
    HostToDevice
);

kernel<<<grid, block>>>();

cudaDeviceSynchronize();

cudaMemcpy(
    DeviceToHost
);

validate();

cudaFree();

free_host();
```

This lifecycle is fundamental to CUDA programming.

---

# 2.10 CUDA Synchronization

CUDA operations are often asynchronous.

For example:

```cpp
kernel<<<blocks, threads>>>();
```

does not necessarily mean:

```text
kernel completely finished
```

immediately after the launch.

You can explicitly wait:

```cpp
cudaDeviceSynchronize();
```

Conceptually:

```text
CPU
 │
 ├── launch kernel ────────► GPU
 │
 │ continues
 │
 └── cudaDeviceSynchronize()
              │
              ▼
        wait for GPU
```

Synchronization is useful for:

* Correctness
* Debugging
* Timing
* Coordinating dependent work

But unnecessary synchronization can hurt performance.

---

# 2.11 Synchronization Inside a Block

CUDA provides:

```cpp
__syncthreads();
```

This synchronizes threads within the same block.

Example:

```text
Thread 0 ──┐
Thread 1 ──┤
Thread 2 ──┼── __syncthreads()
Thread 3 ──┤
Thread 4 ──┘
```

All participating threads must reach the synchronization point correctly.

Important:

```text
__syncthreads()
```

does not synchronize arbitrary threads across the entire GPU grid.

Block-level synchronization is one of the reasons CUDA blocks are useful.

---

# 2.12 Shared Memory

Shared memory belongs to a thread block.

```text
Block
 ├── Thread 0
 ├── Thread 1
 ├── Thread 2
 └── Thread 3
       │
       ▼
 Shared Memory
```

Example:

```cpp
__shared__ float tile[256];
```

Threads in the same block can access it.

Shared memory is useful when data is reused by multiple threads.

Typical pattern:

```text
Global Memory
      ↓
Shared Memory
      ↓
Registers
      ↓
Compute
```

This reduces expensive global-memory accesses.

---

# 2.13 Why Shared Memory Matters

Consider matrix multiplication.

Without tiling:

```text
Thread
  ↓
Global Memory
  ↓
Load A
Load B
  ↓
Compute
```

Many threads repeatedly load the same data.

With tiling:

```text
Global Memory
      ↓
Shared Memory
      ↓
Many threads reuse data
      ↓
Registers
      ↓
Compute
```

This improves:

```text
Data reuse
Memory efficiency
Arithmetic intensity
```

---

# 2.14 CUDA Streams

A CUDA stream is a sequence of operations that execute in order.

Example:

```text
Stream 0:

Copy A
  ↓
Kernel A
  ↓
Copy Result
```

Multiple streams can potentially overlap independent work.

```text
Stream 1:
Copy A ───── Kernel A ───── Copy A_Result

Stream 2:
      Copy B ───── Kernel B ───── Copy B_Result
```

This can allow:

```text
Data transfer
      +
Computation
```

to overlap when hardware and memory conditions permit.

---

# 2.15 Why Streams Matter in Production

Consider an inference server:

```text
Request 1
Request 2
Request 3
Request 4
```

Instead of treating every operation as completely serialized, CUDA streams can help organize independent GPU work.

Streams are important for:

* Asynchronous execution
* Pipeline design
* Overlapping transfers and computation
* Multi-request workloads
* GPU inference systems

Modern frameworks such as PyTorch heavily rely on CUDA streams internally.

---

# 2.16 CUDA Events

CUDA events are commonly used for GPU-side timing and synchronization.

Example:

```cpp
cudaEvent_t start, stop;

cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

kernel<<<blocks, threads>>>();

cudaEventRecord(stop);

cudaEventSynchronize(stop);

float milliseconds = 0;

cudaEventElapsedTime(
    &milliseconds,
    start,
    stop
);
```

Why not simply use CPU timing?

Because:

```cpp
kernel<<<...>>>();
```

may be asynchronous.

Therefore:

```text
CPU timer

start
 ↓
launch kernel
 ↓
stop
```

may measure only launch/dispatch time rather than actual GPU execution.

CUDA events measure GPU-side execution more appropriately.

---

# 2.17 CUDA Error Handling

Production CUDA code must check errors.

Basic pattern:

```cpp
cudaError_t err = cudaGetLastError();

if (err != cudaSuccess)
{
    printf(
        "CUDA Error: %s\n",
        cudaGetErrorString(err)
    );
}
```

After synchronization:

```cpp
cudaDeviceSynchronize();
```

you can check:

```cpp
cudaGetLastError();
```

or capture the returned error from the synchronization call.

A robust CUDA codebase should consistently check:

```text
Memory allocation
Memory copies
Kernel launches
Synchronization
API calls
```

---

# 2.18 Common CUDA Errors

### Invalid memory access

Example:

```cpp
C[idx] = A[idx] + B[idx];
```

without:

```cpp
if (idx < N)
```

Potential result:

```text
out-of-bounds memory access
```

---

### Wrong launch configuration

Example:

```text
N = 1000
threads = 256
```

You need enough blocks:

```cpp
blocks =
    (N + threads - 1) / threads;
```

---

### Race condition

Multiple threads modify the same memory location:

```text
Thread 1 ──┐
Thread 2 ──┼──> same variable
Thread 3 ──┘
```

Without proper synchronization or atomic operations, results can become nondeterministic.

---

# 2.19 Atomic Operations

CUDA provides atomic operations for safe updates.

Example:

```cpp
atomicAdd(
    &result,
    value
);
```

Conceptually:

```text
Thread 1 ──┐
Thread 2 ──┤
Thread 3 ──┼── atomicAdd → result
Thread 4 ──┘
```

Atomics are useful for:

* Histograms
* Reductions
* Counters
* Accumulation

But excessive atomic contention can severely reduce performance.

---

# 2.20 CUDA Grid → Block → Warp → Thread

The complete hierarchy:

```text
GPU
 │
 └── Kernel
      │
      └── Grid
           │
           ├── Block
           │    ├── Warp
           │    │    ├── Thread
           │    │    ├── Thread
           │    │    └── ...
           │    │
           │    └── Warp
           │
           └── Block
```

Think:

```text
Grid
 ↓
Blocks
 ↓
Warps
 ↓
Threads
```

And physically:

```text
Blocks
 ↓
scheduled onto SMs
 ↓
warps execute
 ↓
CUDA/Tensor execution units
```

---

# 2.21 Practical Example — Vector Addition

```text
A
│
├── A[0]
├── A[1]
├── A[2]
└── ...

B
│
├── B[0]
├── B[1]
├── B[2]
└── ...

        ↓

GPU Threads

Thread 0 → A[0] + B[0]
Thread 1 → A[1] + B[1]
Thread 2 → A[2] + B[2]
Thread 3 → A[3] + B[3]
...
```

This is highly parallel.

However, the computation per byte is small.

Therefore:

```text
Vector Addition
      ↓
Low arithmetic intensity
      ↓
Often memory-bound
```

---

# 2.22 Practical Example — Matrix Multiplication

Compute:

```text
C = A × B
```

Conceptually:

```text
A[M,K] × B[K,N]
        ↓
     C[M,N]
```

A simple CUDA implementation can assign:

```text
One thread → One C element
```

Each thread computes:

```text
C[row][col]
```

by iterating over K:

```text
C[row][col]

= A[row][0] × B[0][col]
+ A[row][1] × B[1][col]
+ ...
+ A[row][K-1] × B[K-1][col]
```

But this naïve implementation repeatedly reads global memory.

Optimized implementation:

```text
Global Memory
      ↓
Shared-memory tiles
      ↓
Registers
      ↓
Matrix computation
```

Modern production implementations may use:

```text
Tensor Cores
+
optimized tiling
+
shared memory
+
register blocking
+
specialized libraries
```

---

# 2.23 Practical Example — Reduction

Suppose:

```text
A = [1, 2, 3, 4]
```

Need:

```text
sum(A) = 10
```

Parallel reduction:

```text
1   2   3   4
 \ /     \ /
  3       7
    \   /
     10
```

CUDA reduction involves:

```text
Threads
   ↓
Partial sums
   ↓
Shared memory
   ↓
Tree reduction
   ↓
Final result
```

This introduces challenges:

* Synchronization
* Memory access
* Warp efficiency
* Atomic operations
* Load balancing

Reduction is an important CUDA optimization problem.

---

# 2.24 Practical Example — Softmax

Softmax:

```text
softmax(x_i)
=
exp(x_i) / Σ exp(x_j)
```

Implementation requires:

```text
1. Find maximum
2. Subtract maximum
3. exp()
4. Sum
5. Divide
```

This means:

```text
Reduction
+
Elementwise operations
+
Synchronization
+
Memory access
```

A naïve implementation may launch multiple kernels:

```text
Kernel 1 → max
Kernel 2 → exp
Kernel 3 → sum
Kernel 4 → divide
```

Optimized implementations try to reduce:

```text
Kernel launches
Global memory traffic
Synchronization
```

This is why optimized GPU kernels matter heavily in Transformer inference.

---

# 2.25 Practical Example — LayerNorm

LayerNorm roughly performs:

```text
mean
 ↓
variance
 ↓
normalize
 ↓
scale + bias
```

For a tensor:

```text
X
 ↓
mean
 ↓
variance
 ↓
(X - mean) / sqrt(variance + ε)
 ↓
γ × normalized + β
```

It involves:

```text
Reduction
+
Elementwise computation
+
Memory access
```

Therefore efficient LayerNorm implementations focus heavily on:

```text
Memory bandwidth
Kernel fusion
Parallel reduction
Register/shared-memory usage
```

---

# 2.26 Practical Example — Attention

Attention:

```text
Q = XWq
K = XWk
V = XWv

QKᵀ
 ↓
Scaling
 ↓
Softmax
 ↓
× V
```

GPU execution becomes:

```text
Matrix multiplication
        ↓
Matrix multiplication
        ↓
Softmax
        ↓
Matrix multiplication
```

The major performance considerations are:

```text
Tensor Core utilization
Memory bandwidth
Shared memory
Registers
Kernel launches
Synchronization
Sequence length
Batch size
```

This is one of the main reasons optimized attention kernels such as FlashAttention are important.

---

# 2.27 CUDA Kernel Launch Configuration

Typical launch:

```cpp
kernel<<<blocks, threads>>>();
```

For example:

```cpp
int threads = 256;

int blocks =
    (N + threads - 1) / threads;

kernel<<<blocks, threads>>>();
```

The exact optimal block size depends on:

```text
Register usage
Shared memory usage
Kernel characteristics
GPU architecture
Occupancy
Memory behavior
Instruction mix
```

Do not blindly assume:

```text
256 threads = always optimal
```

Benchmark.

---

# 2.28 CUDA Performance Mental Model

When writing a CUDA kernel, ask:

### 1. How much parallelism exists?

```text
Number of elements
Number of independent operations
```

### 2. Is the kernel compute-bound or memory-bound?

```text
Compute
vs
Memory
```

### 3. Are memory accesses coalesced?

```text
Thread 0 → address 0
Thread 1 → address 1
Thread 2 → address 2
...
```

is generally preferable.

### 4. Can data be reused?

Use:

```text
Shared Memory
Registers
Cache
```

### 5. Are warps diverging?

Avoid unnecessary:

```cpp
if/else
```

patterns where threads in the same warp take different paths.

### 6. Is occupancy sufficient?

Check:

```text
Registers
Shared memory
Threads/block
Active warps
```

### 7. Are there unnecessary synchronizations?

Synchronization has a cost.

### 8. Are kernel launches dominating?

Many tiny kernels can cause:

```text
Launch overhead
+
Extra global-memory traffic
```

Kernel fusion can sometimes help.

---

# 2.29 CUDA Kernel Fusion

Suppose we have:

```text
Kernel A
 ↓
Global Memory
 ↓
Kernel B
 ↓
Global Memory
 ↓
Kernel C
```

Each kernel may write/read intermediate results.

Fusion can produce:

```text
Fused Kernel
     ↓
Compute A
     ↓
Compute B
     ↓
Compute C
     ↓
Final output
```

Benefits can include:

```text
Fewer kernel launches
Less global memory traffic
Better cache/register reuse
Lower latency
```

But fusion can also increase:

```text
Register pressure
Code complexity
Occupancy pressure
```

So fusion is an optimization, not automatically better.

---

# 2.30 CUDA vs CPU Execution

CPU:

```text
Few powerful cores
        ↓
Complex control
        ↓
Low-latency execution
```

GPU:

```text
Many parallel execution resources
        ↓
Massive parallelism
        ↓
High throughput
```

CUDA programming therefore requires thinking in terms of:

```text
Data parallelism
```

rather than:

```text
Sequential instructions
```

---

# 2.31 CUDA in PyTorch

You usually don't write raw CUDA for every operation.

PyTorch provides CUDA kernels:

```python
x = torch.randn(
    1024,
    1024,
    device="cuda"
)

y = torch.matmul(x, x)
```

The execution becomes conceptually:

```text
Python
  ↓
PyTorch
  ↓
CUDA backend
  ↓
CUDA kernel
  ↓
GPU
```

For optimized operations, PyTorch may use:

```text
CUDA
cuBLAS
cuDNN
CUTLASS
Tensor Cores
custom kernels
Triton kernels
```

This is why understanding CUDA helps explain what happens underneath PyTorch.

---

# 2.32 CUDA in LLM Inference

Typical LLM inference:

```text
User Request
     ↓
Inference Server
     ↓
Scheduler
     ↓
Batching
     ↓
PyTorch / CUDA
     ↓
GPU
     │
     ├── Model Weights
     ├── Activations
     ├── KV Cache
     └── Temporary Buffers
```

Inside the GPU:

```text
Kernel launch
      ↓
Grid
      ↓
Blocks
      ↓
Warps
      ↓
Tensor/CUDA cores
      ↓
Registers / Shared Memory
      ↓
L1 / L2
      ↓
HBM/GDDR
```

Understanding this hierarchy is essential for:

```text
vLLM
SGLang
TensorRT-LLM
Triton
FlashAttention
Custom CUDA kernels
```

---

# 2.33 CUDA Performance Bottleneck Example

Suppose GPU utilization is only:

```text
30%
```

Do NOT immediately conclude:

```text
GPU is broken
```

Possible reasons:

```text
CPU bottleneck
↓
Small batch
↓
Kernel launch overhead
↓
Memory-bound workload
↓
Synchronization
↓
Data transfer
↓
Poor memory access
↓
Warp divergence
↓
Low arithmetic intensity
↓
Insufficient parallelism
```

Production GPU optimization starts with profiling rather than guessing.

---

# 2.34 Tools You Will Use Later

Important NVIDIA tools:

| Tool             | Purpose                     |
| ---------------- | --------------------------- |
| `nvidia-smi`     | GPU monitoring              |
| Nsight Systems   | System-level timeline       |
| Nsight Compute   | Kernel-level profiling      |
| CUDA events      | Kernel timing               |
| PyTorch Profiler | PyTorch execution profiling |
| DCGM             | GPU monitoring/telemetry    |

Typical workflow:

```text
Application
    ↓
Measure
    ↓
Profile
    ↓
Find bottleneck
    ↓
Optimize
    ↓
Benchmark
    ↓
Compare
```

Never optimize based only on intuition.

---

# 2.35 Production CUDA Checklist

Before considering a CUDA kernel production-ready:

```text
□ Correct indexing
□ Bounds checking
□ CUDA error handling
□ Memory leak checks
□ Race-condition checks
□ Correct synchronization
□ Coalesced memory access
□ Reasonable register usage
□ Reasonable shared-memory usage
□ Sufficient parallelism
□ Benchmarking
□ Profiling
□ Numerical correctness
□ Deterministic behavior where required
□ Tested on target GPU architectures
```

---

# 2.36 Common CUDA Interview Questions

### Fundamentals

**Q: What is a CUDA kernel?**

A function executed by many GPU threads.

---

**Q: What is the difference between host and device?**

```text
Host   → CPU
Device → GPU
```

---

**Q: What is a block?**

A group of CUDA threads that can cooperate using shared memory and block-level synchronization.

---

**Q: What is a warp?**

A group of 32 threads executed together using NVIDIA's SIMT execution model.

---

**Q: What is shared memory?**

Fast, programmer-managed memory accessible by threads within a block.

---

**Q: Why is coalesced memory access important?**

It allows memory requests from neighboring threads to be serviced efficiently, improving effective memory bandwidth.

---

**Q: What is occupancy?**

The ratio of active warps on an SM to the maximum supported active warps.

Higher occupancy can help hide latency, but does not automatically mean higher performance.

---

**Q: Why can too many registers hurt performance?**

High register usage can limit the number of simultaneously active warps/blocks on an SM.

---

**Q: What causes warp divergence?**

Threads in the same warp following different control-flow paths.

---

**Q: Why use shared memory?**

To reduce repeated global-memory accesses and increase data reuse.

---

**Q: Why use CUDA streams?**

To organize asynchronous work and potentially overlap independent computation and memory transfers.

---

# 2.37 The Complete CUDA Mental Model

Remember this architecture:

```text
                    CPU
                     │
                     │
              CUDA API / Runtime
                     │
                     ▼
                  Kernel
                     │
                     ▼
                   Grid
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       Block 0                Block 1
          │                     │
       ┌──┴──┐               ┌──┴──┐
       ▼     ▼               ▼     ▼
     Warp   Warp           Warp   Warp
       │     │               │     │
       ▼     ▼               ▼     ▼
    Threads Threads       Threads Threads
       │
       ▼
      SM
       │
 ┌─────┼───────────────┐
 │     │               │
 ▼     ▼               ▼
Registers Shared      CUDA /
          Memory      Tensor Cores
 │     │
 └─────┴───────────────┐
                       ▼
                  L1 / L2 Cache
                       │
                       ▼
                  Global Memory
                       │
                       ▼
                    HBM/GDDR
```

The programming model is:

```text
CPU
 ↓
Launch Kernel
 ↓
Grid
 ↓
Blocks
 ↓
Threads
 ↓
SM
 ↓
Warp Scheduling
 ↓
CUDA/Tensor Cores
 ↓
Registers / Shared Memory
 ↓
Cache
 ↓
Global Memory
```

---

# 2.38 The Most Important CUDA Principles

If you remember only these principles:

### Principle 1 — Think in parallel

Instead of:

```text
process item 1
process item 2
process item 3
```

think:

```text
Thread 1 → item 1
Thread 2 → item 2
Thread 3 → item 3
...
```

---

### Principle 2 — Data movement matters

Moving data:

```text
HBM
 ↓
L2
 ↓
L1
 ↓
Shared Memory
 ↓
Registers
```

has a cost.

Good GPU code minimizes unnecessary movement.

---

### Principle 3 — Reuse data

Prefer:

```text
Load once
 ↓
Reuse many times
```

rather than:

```text
Load repeatedly
```

---

### Principle 4 — Keep the GPU busy

You want:

```text
Enough parallel work
+
Efficient scheduling
+
Good memory access
```

---

### Principle 5 — Profile before optimizing

Never assume:

```text
GPU utilization = performance
```

Measure:

```text
Latency
Throughput
Memory bandwidth
SM utilization
Kernel duration
Occupancy
Warp efficiency
Memory transactions
```

---

# Phase 02 Final Summary

You should now understand:

```text
CUDA
 │
 ├── Kernel
 │
 ├── Thread
 │
 ├── Block
 │
 ├── Grid
 │
 ├── Warp
 │
 ├── Thread indexing
 │
 ├── Host memory
 │
 ├── Device memory
 │
 ├── cudaMalloc
 │
 ├── cudaMemcpy
 │
 ├── Synchronization
 │
 ├── Shared memory
 │
 ├── Streams
 │
 ├── Events
 │
 ├── Error handling
 │
 ├── Atomics
 │
 └── Performance optimization
```

And the core execution flow:

```text
             CPU
              │
              │ kernel launch
              ▼
            GPU
              │
            Grid
              │
           Blocks
              │
            Warps
              │
           Threads
              │
             SM
              │
      ┌───────┴────────┐
      ▼                ▼
 CUDA Cores       Tensor Cores
      │                │
      └───────┬────────┘
              ▼
          Registers
              │
       Shared Memory
              │
          L1 / L2
              │
              ▼
        Global Memory
```

## The Staff-Level Mental Model

When you see any GPU workload, ask:

```text
1. What is the parallel unit?

2. How are threads mapped to data?

3. How are blocks mapped to SMs?

4. What memory does each thread access?

5. Are accesses coalesced?

6. Can data be reused?

7. Is the kernel compute-bound or memory-bound?

8. How much register/shared-memory pressure exists?

9. Is synchronization necessary?

10. Can kernels be fused?

11. Can computation and transfers overlap?

12. What does the profiler show?
```

This mental model will carry directly into:

```text
CUDA
 ↓
PyTorch CUDA
 ↓
Triton
 ↓
Tensor Cores
 ↓
FlashAttention
 ↓
vLLM
 ↓
SGLang
 ↓
TensorRT-LLM
 ↓
Production LLM inference
```

# Phase 02 Status

**Phase 02 — CUDA Programming: COMPLETE**

Next:

# Phase 03 — CUDA Memory

We will go much deeper into:

```text
Global Memory
Shared Memory
Registers
L1 Cache
L2 Cache
Constant Memory
Memory Coalescing
Memory Transactions
Bank Conflicts
Pinned Memory
Unified Memory
Memory Bandwidth
Memory Latency
Memory Optimization
```

This phase is especially important because **a large portion of real GPU performance optimization is ultimately a data-movement problem.**
