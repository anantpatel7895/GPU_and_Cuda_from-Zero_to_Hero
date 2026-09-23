
# CUDA & GPU Engineering — Zero to Hero

> A production-oriented learning roadmap for mastering GPU architecture, CUDA programming, GPU optimization, multi-GPU systems, and LLM inference infrastructure.

---

## 🎯 Goal

The goal of this repository is to progress from **GPU fundamentals to production-grade GPU engineering**.

This is not intended to be a CUDA syntax-only course.

By the end of this roadmap, the objective is to be able to:

* Understand modern NVIDIA GPU architecture
* Understand how CUDA executes workloads
* Write CUDA kernels
* Understand GPU memory hierarchy
* Optimize GPU memory access
* Understand warps, blocks, grids, and SMs
* Profile GPU workloads
* Identify compute-bound vs memory-bound workloads
* Optimize CUDA kernels
* Understand Tensor Cores
* Work with FP32, FP16, BF16, TF32, FP8, INT8 and INT4
* Write custom PyTorch CUDA operations
* Write GPU kernels using Triton
* Understand FlashAttention and fused kernels
* Understand KV Cache and LLM inference bottlenecks
* Understand vLLM/SGLang GPU execution concepts
* Understand CUDA streams and asynchronous execution
* Understand CUDA Graphs
* Understand NCCL
* Understand multi-GPU communication
* Understand Data, Tensor, Pipeline and Expert Parallelism
* Deploy GPU workloads using Docker and Kubernetes
* Monitor GPU workloads
* Optimize GPU utilization, latency, throughput and cost

The final goal is:

> **Be able to look at a production AI workload, understand what the GPU is doing, identify the bottleneck, and make an informed engineering optimization.**

---

# 🧠 Learning Philosophy

This roadmap follows:

```text
Understand
    ↓
Implement
    ↓
Test
    ↓
Benchmark
    ↓
Profile
    ↓
Optimize
    ↓
Apply to LLMs
    ↓
Deploy to Production
```

We will avoid jumping directly into frameworks such as vLLM or TensorRT-LLM without understanding the GPU concepts underneath them.

---

# 🗺️ Roadmap

```text
Phase 00
GPU Fundamentals
        ↓
Phase 01
GPU Architecture
        ↓
Phase 02
CUDA Programming
        ↓
Phase 03
CUDA Memory
        ↓
Phase 04
CUDA Parallelism
        ↓
Phase 05
CUDA Performance Engineering
        ↓
Phase 06
Advanced CUDA
        ↓
Phase 07
PyTorch + CUDA
        ↓
Phase 08
Triton
        ↓
Phase 09
Tensor Cores
        ↓
Phase 10
Multi-GPU
        ↓
Phase 11
LLM GPU Optimization
        ↓
Phase 12
Production GPU Infrastructure
```

---

# 📚 Phase 00 — GPU Fundamentals

## Objective

Build a strong mental model of GPUs before writing CUDA.

### Lessons

* [ ] 0.1 — What is a GPU?
* [ ] 0.2 — CPU vs GPU
* [ ] 0.3 — Why GPUs are good for AI
* [ ] 0.4 — Latency vs Throughput
* [ ] 0.5 — FLOPS
* [ ] 0.6 — Memory Bandwidth
* [ ] 0.7 — Compute-bound vs Memory-bound
* [ ] 0.8 — Arithmetic Intensity
* [ ] 0.9 — SIMD vs SIMT
* [ ] 0.10 — GPU Memory Basics
* [ ] 0.11 — NVIDIA GPU Overview
* [ ] 0.12 — GPU Execution Lifecycle
* [ ] 0.13 — GPU Performance Mental Model

### Key concepts

```text
CPU
 ├── Few powerful cores
 ├── Large caches
 ├── Low latency
 └── Complex control logic

GPU
 ├── Many parallel execution units
 ├── High memory bandwidth
 ├── High throughput
 └── Designed for massively parallel workloads
```

---

# 📚 Phase 01 — GPU Architecture

## Objective

Understand what physically happens inside an NVIDIA GPU.

### Lessons

* [ ] 1.1 — NVIDIA GPU Architecture
* [ ] 1.2 — Streaming Multiprocessor (SM)
* [ ] 1.3 — CUDA Cores
* [ ] 1.4 — Tensor Cores
* [ ] 1.5 — Warp
* [ ] 1.6 — Thread
* [ ] 1.7 — Thread Block
* [ ] 1.8 — Grid
* [ ] 1.9 — Warp Scheduler
* [ ] 1.10 — Registers
* [ ] 1.11 — Shared Memory
* [ ] 1.12 — L1 Cache
* [ ] 1.13 — L2 Cache
* [ ] 1.14 — Global Memory
* [ ] 1.15 — HBM/GDDR
* [ ] 1.16 — PCIe
* [ ] 1.17 — NVLink
* [ ] 1.18 — GPU Execution Model

### Mental model

```text
GPU
│
├── SM 0
│   ├── CUDA Cores
│   ├── Tensor Cores
│   ├── Registers
│   ├── Shared Memory
│   └── L1 Cache
│
├── SM 1
│   └── ...
│
├── SM 2
│   └── ...
│
└── ...
        │
        ▼
       L2
        │
        ▼
   Global Memory
```

---

# 📚 Phase 02 — CUDA Programming

## Objective

Write CUDA kernels and understand CUDA's programming model.

### Lessons

* [ ] 2.1 — CUDA Overview
* [ ] 2.2 — CUDA Runtime
* [ ] 2.3 — CUDA Toolkit
* [ ] 2.4 — CUDA Compilation
* [ ] 2.5 — CUDA Kernel
* [ ] 2.6 — `__global__`
* [ ] 2.7 — `__device__`
* [ ] 2.8 — `__host__`
* [ ] 2.9 — Kernel Launch
* [ ] 2.10 — Thread Indexing
* [ ] 2.11 — Block Indexing
* [ ] 2.12 — Grid Indexing
* [ ] 2.13 — 1D Grid
* [ ] 2.14 — 2D Grid
* [ ] 2.15 — 3D Grid
* [ ] 2.16 — CUDA Error Handling

### Fundamental indexing

```cpp
int idx =
    blockIdx.x * blockDim.x
    + threadIdx.x;
```

### First execution model

```text
Grid
│
├── Block 0
│   ├── Thread 0
│   ├── Thread 1
│   ├── Thread 2
│   └── ...
│
├── Block 1
│   ├── Thread 0
│   ├── Thread 1
│   └── ...
│
└── Block N
```

---

# 📚 Phase 03 — CUDA Memory

## Objective

Understand GPU memory hierarchy and efficient data movement.

### Lessons

* [ ] 3.1 — GPU Memory Hierarchy
* [ ] 3.2 — Registers
* [ ] 3.3 — Local Memory
* [ ] 3.4 — Shared Memory
* [ ] 3.5 — Global Memory
* [ ] 3.6 — Constant Memory
* [ ] 3.7 — Texture Memory
* [ ] 3.8 — L1 Cache
* [ ] 3.9 — L2 Cache
* [ ] 3.10 — Memory Coalescing
* [ ] 3.11 — Memory Alignment
* [ ] 3.12 — Bank Conflicts
* [ ] 3.13 — Host-to-Device Transfer
* [ ] 3.14 — Device-to-Host Transfer
* [ ] 3.15 — Pinned Memory
* [ ] 3.16 — Unified Memory

### Memory hierarchy

```text
Fast
 │
 ▼
Registers
 │
 ▼
Shared Memory
 │
 ▼
L1 Cache
 │
 ▼
L2 Cache
 │
 ▼
Global Memory
 │
 ▼
CPU Memory
 │
 ▼
Slow
```

---

# 📚 Phase 04 — CUDA Parallelism

## Objective

Understand how thousands of GPU threads execute efficiently.

### Lessons

* [ ] 4.1 — Thread Hierarchy
* [ ] 4.2 — Warp Execution
* [ ] 4.3 — SIMT
* [ ] 4.4 — Warp Scheduling
* [ ] 4.5 — Warp Divergence
* [ ] 4.6 — Branch Divergence
* [ ] 4.7 — Thread Synchronization
* [ ] 4.8 — `__syncthreads()`
* [ ] 4.9 — Atomic Operations
* [ ] 4.10 — Race Conditions
* [ ] 4.11 — Warp-Level Primitives
* [ ] 4.12 — Warp Shuffle
* [ ] 4.13 — Parallel Reduction
* [ ] 4.14 — Prefix Sum / Scan

### Warp

```text
Block
│
├── Warp 0
│   ├── Thread 0
│   ├── Thread 1
│   ├── ...
│   └── Thread 31
│
├── Warp 1
│
└── Warp N
```

---

# 📚 Phase 05 — CUDA Performance Engineering

## Objective

Move from "working CUDA" to **fast CUDA**.

### Lessons

* [ ] 5.1 — GPU Performance Model
* [ ] 5.2 — Compute-bound Workloads
* [ ] 5.3 — Memory-bound Workloads
* [ ] 5.4 — Arithmetic Intensity
* [ ] 5.5 — Roofline Model
* [ ] 5.6 — Occupancy
* [ ] 5.7 — Register Pressure
* [ ] 5.8 — Shared Memory Usage
* [ ] 5.9 — Warp Efficiency
* [ ] 5.10 — Memory Throughput
* [ ] 5.11 — Instruction Throughput
* [ ] 5.12 — Kernel Launch Overhead
* [ ] 5.13 — Kernel Fusion
* [ ] 5.14 — Persistent Kernels
* [ ] 5.15 — Performance Benchmarking

### Important principle

> High occupancy does not automatically mean high performance.

A kernel can have high occupancy and still be slow because it is:

* memory-bandwidth bound
* instruction bound
* synchronization bound
* limited by register pressure
* limited by memory latency
* suffering from poor memory access patterns

---

# 📚 Phase 06 — Advanced CUDA

## Objective

Understand asynchronous and production-oriented CUDA execution.

### Lessons

* [ ] 6.1 — CUDA Streams
* [ ] 6.2 — Default Stream
* [ ] 6.3 — Multiple Streams
* [ ] 6.4 — Asynchronous Execution
* [ ] 6.5 — CUDA Events
* [ ] 6.6 — Async Memory Copy
* [ ] 6.7 — Pinned Memory
* [ ] 6.8 — Overlapping Compute and Transfer
* [ ] 6.9 — CUDA Graphs
* [ ] 6.10 — CUDA Graph Capture
* [ ] 6.11 — CUDA Graph Replay
* [ ] 6.12 — CUDA Memory Pools
* [ ] 6.13 — Error Handling
* [ ] 6.14 — Debugging CUDA Applications

### Concurrent execution

```text
Stream 0:
[Kernel A]      [Kernel B]

Stream 1:
      [Memcpy]       [Kernel C]

Stream 2:
[Kernel D]
```

---

# 📚 Phase 07 — PyTorch + CUDA

## Objective

Understand how deep-learning frameworks interact with CUDA.

### Lessons

* [ ] 7.1 — PyTorch CUDA Architecture
* [ ] 7.2 — ATen
* [ ] 7.3 — CUDA Dispatch
* [ ] 7.4 — Tensor Device
* [ ] 7.5 — CUDA Memory Allocation
* [ ] 7.6 — PyTorch CUDA Streams
* [ ] 7.7 — PyTorch Synchronization
* [ ] 7.8 — CUDA Graphs in PyTorch
* [ ] 7.9 — Custom CUDA Extensions
* [ ] 7.10 — Custom PyTorch Operators
* [ ] 7.11 — Benchmarking PyTorch GPU Code
* [ ] 7.12 — Profiling PyTorch

### Execution path

```text
Python
  │
  ▼
PyTorch
  │
  ▼
ATen
  │
  ▼
CUDA Runtime
  │
  ▼
CUDA Kernel
  │
  ▼
GPU
```

---

# 📚 Phase 08 — Triton

## Objective

Learn modern GPU kernel development for AI workloads.

### Lessons

* [ ] 8.1 — Why Triton?
* [ ] 8.2 — Triton Programming Model
* [ ] 8.3 — Triton Program IDs
* [ ] 8.4 — Blocks
* [ ] 8.5 — Memory Operations
* [ ] 8.6 — Masking
* [ ] 8.7 — Reductions
* [ ] 8.8 — Autotuning
* [ ] 8.9 — Kernel Fusion
* [ ] 8.10 — Triton MatMul
* [ ] 8.11 — Triton Softmax
* [ ] 8.12 — Triton LayerNorm
* [ ] 8.13 — Triton Attention
* [ ] 8.14 — CUDA vs Triton

### Comparison

```text
PyTorch
   │
   ▼
High-level GPU operations


Triton
   │
   ▼
Python-based GPU kernels


CUDA
   │
   ▼
Low-level GPU programming
```

---

# 📚 Phase 09 — Tensor Cores & Numerical Formats

## Objective

Understand hardware acceleration for deep-learning workloads.

### Lessons

* [ ] 9.1 — Tensor Core Architecture
* [ ] 9.2 — Matrix Multiply Accumulate
* [ ] 9.3 — FP32
* [ ] 9.4 — FP16
* [ ] 9.5 — BF16
* [ ] 9.6 — TF32
* [ ] 9.7 — FP8
* [ ] 9.8 — INT8
* [ ] 9.9 — INT4
* [ ] 9.10 — Mixed Precision
* [ ] 9.11 — Automatic Mixed Precision
* [ ] 9.12 — Quantization
* [ ] 9.13 — Weight-only Quantization
* [ ] 9.14 — Activation Quantization
* [ ] 9.15 — GPTQ
* [ ] 9.16 — AWQ
* [ ] 9.17 — SmoothQuant

### Numerical formats

```text
FP32
 │
 ├── High precision
 └── High memory usage

FP16 / BF16
 │
 ├── Lower memory
 └── High throughput

FP8
 │
 ├── Very high throughput
 └── Lower numerical precision

INT8 / INT4
 │
 ├── Very low memory
 └── Quantized inference
```

---

# 📚 Phase 10 — Multi-GPU

## Objective

Understand communication and parallelism across multiple GPUs.

### Lessons

* [ ] 10.1 — Why Multi-GPU?
* [ ] 10.2 — PCIe
* [ ] 10.3 — NVLink
* [ ] 10.4 — NVSwitch
* [ ] 10.5 — GPU-to-GPU Communication
* [ ] 10.6 — NCCL
* [ ] 10.7 — AllReduce
* [ ] 10.8 — AllGather
* [ ] 10.9 — ReduceScatter
* [ ] 10.10 — Broadcast
* [ ] 10.11 — AllToAll
* [ ] 10.12 — Communication Bottlenecks

### Example

```text
GPU 0 ─────┐
GPU 1 ─────┤
GPU 2 ─────┼──► NCCL
GPU 3 ─────┘
```

---

# 📚 Phase 11 — Parallelism for AI/LLMs

## Objective

Understand how large models are distributed across GPUs.

### Lessons

* [ ] 11.1 — Data Parallelism
* [ ] 11.2 — Tensor Parallelism
* [ ] 11.3 — Pipeline Parallelism
* [ ] 11.4 — Sequence Parallelism
* [ ] 11.5 — Expert Parallelism
* [ ] 11.6 — Hybrid Parallelism
* [ ] 11.7 — Communication Overhead
* [ ] 11.8 — GPU Memory Scaling

### Data Parallelism

```text
GPU 0 → Batch 0
GPU 1 → Batch 1
GPU 2 → Batch 2
GPU 3 → Batch 3
```

### Tensor Parallelism

```text
             Model
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
     GPU 0   GPU 1   GPU 2
```

### Pipeline Parallelism

```text
GPU 0 → Layers 0-10
          ↓
GPU 1 → Layers 11-20
          ↓
GPU 2 → Layers 21-30
```

---

# 📚 Phase 12 — LLM GPU Optimization

## Objective

Understand how GPU engineering applies directly to LLM inference.

### Lessons

* [ ] 12.1 — Transformer GPU Workload
* [ ] 12.2 — Matrix Multiplication
* [ ] 12.3 — Attention GPU Execution
* [ ] 12.4 — Prefill
* [ ] 12.5 — Decode
* [ ] 12.6 — KV Cache
* [ ] 12.7 — KV Cache Memory Calculation
* [ ] 12.8 — MHA
* [ ] 12.9 — MQA
* [ ] 12.10 — GQA
* [ ] 12.11 — Memory Bandwidth During Decode
* [ ] 12.12 — FlashAttention
* [ ] 12.13 — PagedAttention
* [ ] 12.14 — Continuous Batching
* [ ] 12.15 — Prefix Caching
* [ ] 12.16 — Speculative Decoding
* [ ] 12.17 — Quantized Inference
* [ ] 12.18 — Kernel Fusion
* [ ] 12.19 — CUDA Graphs for LLM Serving

### LLM execution

```text
Prompt
  │
  ▼
Tokenizer
  │
  ▼
Prefill
  │
  ├── Q
  ├── K
  ├── V
  │
  ▼
KV Cache
  │
  ▼
Decode
  │
  ▼
Next Token
  │
  ▼
KV Cache Update
  │
  ▼
Next Token
```

---

# 📚 Phase 13 — LLM Inference Engines

> This phase connects GPU fundamentals with production GenAI infrastructure.

### Technologies

* [ ] vLLM
* [ ] SGLang
* [ ] TensorRT-LLM
* [ ] llama.cpp
* [ ] Hugging Face TGI
* [ ] Triton Inference Server

### Topics

* [ ] 13.1 — Inference Engine Architecture
* [ ] 13.2 — Scheduler
* [ ] 13.3 — Continuous Batching
* [ ] 13.4 — KV Cache Management
* [ ] 13.5 — PagedAttention
* [ ] 13.6 — Prefix Caching
* [ ] 13.7 — Speculative Decoding
* [ ] 13.8 — Tensor Parallelism
* [ ] 13.9 — Quantization
* [ ] 13.10 — GPU Memory Management
* [ ] 13.11 — Throughput Optimization
* [ ] 13.12 — Latency Optimization

---

# 📚 Phase 14 — Production GPU Infrastructure

## Objective

Deploy and operate GPU workloads in production.

### Lessons

* [ ] 14.1 — NVIDIA Drivers
* [ ] 14.2 — CUDA Toolkit
* [ ] 14.3 — NVIDIA Container Toolkit
* [ ] 14.4 — Docker GPU Runtime
* [ ] 14.5 — Kubernetes GPU Scheduling
* [ ] 14.6 — NVIDIA Device Plugin
* [ ] 14.7 — GPU Resource Requests
* [ ] 14.8 — GPU Limits
* [ ] 14.9 — MIG
* [ ] 14.10 — MPS
* [ ] 14.11 — GPU Isolation
* [ ] 14.12 — DCGM
* [ ] 14.13 — Prometheus
* [ ] 14.14 — Grafana
* [ ] 14.15 — GPU Monitoring
* [ ] 14.16 — GPU Autoscaling
* [ ] 14.17 — Cost Optimization

### Production architecture

```text
                    Client
                      │
                      ▼
                Load Balancer
                      │
                      ▼
                 API Gateway
                      │
                      ▼
                  FastAPI
                      │
                      ▼
              Inference Engine
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
       GPU Node                GPU Node
          │                       │
      ┌───┴───┐               ┌───┴───┐
      │       │               │       │
    GPU 0   GPU 1           GPU 2   GPU 3
      │       │               │       │
      └───────┴───────────────┴───────┘
                      │
                    NCCL
```

---

# 🧪 Projects

The roadmap is project-driven.

---

## Project 01 — GPU Information CLI

Build a CLI that reports:

```text
GPU Name
GPU Memory
Compute Capability
SM Count
CUDA Version
Driver Version
Memory Bandwidth
Temperature
Power Usage
Utilization
```

---

## Project 02 — CUDA Vector Operations

Implement:

* [ ] Vector Addition
* [ ] Vector Subtraction
* [ ] Vector Multiplication
* [ ] Vector Scaling

Compare:

```text
CPU
CUDA
PyTorch
```

Measure:

```text
Latency
Throughput
Speedup
```

---

## Project 03 — CUDA Matrix Multiplication

Implement progressively:

```text
CPU
 ↓
Naive CUDA
 ↓
Tiled CUDA
 ↓
Shared Memory
 ↓
Register Optimization
 ↓
Tensor Core
```

Measure:

* latency
* throughput
* GFLOPS
* memory bandwidth
* speedup

---

## Project 04 — Parallel Reduction

Implement:

```text
sum()
mean()
max()
min()
```

Versions:

```text
Naive
 ↓
Shared Memory
 ↓
Warp Reduction
 ↓
Optimized
```

---

## Project 05 — CUDA Softmax

Implement:

```text
Softmax
```

Then optimize using:

```text
Global Memory
 ↓
Shared Memory
 ↓
Warp-level reduction
```

---

## Project 06 — CUDA LayerNorm

Implement LayerNorm from scratch.

Compare against:

```python
torch.nn.LayerNorm
```

Measure correctness and performance.

---

## Project 07 — CUDA Attention

Implement:

```text
QKᵀ
 ↓
Scale
 ↓
Softmax
 ↓
× V
```

Then optimize.

---

## Project 08 — FlashAttention-style Implementation

Study:

```text
Naive Attention
        ↓
Tiled Attention
        ↓
IO-aware Attention
        ↓
FlashAttention concepts
```

Focus on:

* HBM traffic
* SRAM/shared memory
* tiling
* numerical stability
* memory efficiency

---

## Project 09 — Custom PyTorch CUDA Operator

Create a custom operator:

```python
torch.ops.my_ops.custom_operation(...)
```

Compare:

```text
PyTorch implementation
CUDA implementation
Triton implementation
```

---

## Project 10 — Triton Kernel Suite

Implement:

* [ ] Vector Add
* [ ] Softmax
* [ ] LayerNorm
* [ ] MatMul
* [ ] Attention

Benchmark:

```text
PyTorch
CUDA
Triton
```

---

## Project 11 — KV Cache Simulator

Build an LLM KV-cache simulator.

Calculate:

```text
KV Cache Memory
=
2 × Layers
× KV Heads
× Head Dimension
× Sequence Length
× Bytes per Element
× Batch Size
```

Experiment with:

```text
MHA
MQA
GQA
FP16
BF16
FP8
```

---

## Project 12 — Multi-GPU Communication

Implement experiments using NCCL:

```text
AllReduce
AllGather
ReduceScatter
Broadcast
```

Measure:

```text
Latency
Bandwidth
Scaling
```

---

## Project 13 — Multi-GPU LLM Inference

Deploy an open-source LLM using:

```text
Tensor Parallelism
+
NCCL
+
Multiple GPUs
```

Measure:

```text
TTFT
TPOT
Tokens/sec
GPU utilization
GPU memory
Communication overhead
```

---

## Project 14 — Production GPU Inference Platform

Final project:

```text
Client
  │
  ▼
Load Balancer
  │
  ▼
FastAPI
  │
  ▼
vLLM / SGLang
  │
  ▼
GPU Cluster
  │
  ├── GPU 0
  ├── GPU 1
  ├── GPU 2
  └── GPU 3
  │
  ▼
NCCL
```

Add:

```text
Docker
Kubernetes
Prometheus
Grafana
DCGM
Load Testing
Monitoring
Autoscaling
```

---

# 📊 Benchmarking

Every performance-oriented project should include benchmarks.

We will measure:

```text
Latency
Throughput
GPU Utilization
Memory Utilization
Memory Bandwidth
Compute Utilization
Kernel Duration
CPU Overhead
PCIe Transfer
Inter-GPU Communication
```

For LLMs:

```text
TTFT
Time Per Output Token
Tokens/sec
Request/sec
Batch Size
Context Length
KV Cache Usage
GPU Memory
```

---

# 🔬 Profiling

Primary profiling tools:

```text
Nsight Systems
Nsight Compute
NVIDIA SMI
DCGM
PyTorch Profiler
```

### Nsight Systems

Used for:

```text
System-level timeline
CPU ↔ GPU interaction
Kernel execution
Memory transfers
Streams
Synchronization
```

### Nsight Compute

Used for:

```text
Kernel-level analysis
Occupancy
Memory throughput
Warp efficiency
Instruction throughput
Cache behavior
```

---

# 🧠 Core Performance Mental Model

For every GPU workload ask:

```text
1. Is it compute-bound?

2. Is it memory-bandwidth bound?

3. Is it memory-latency bound?

4. Is kernel launch overhead significant?

5. Is synchronization limiting performance?

6. Is memory access coalesced?

7. Is occupancy sufficient?

8. Is register pressure too high?

9. Is shared memory being used efficiently?

10. Can kernels be fused?

11. Can computation overlap with communication?

12. Can multiple requests be batched?
```

---

# 🚀 LLM Performance Mental Model

For LLM inference:

```text
                    LLM Request
                         │
                         ▼
                      Prefill
                         │
                         │
                  Compute Heavy
                         │
                         ▼
                    KV Cache
                         │
                         ▼
                       Decode
                         │
                         │
                Memory Bandwidth
                     Sensitive
                         │
                         ▼
                    Next Token
```

Optimization opportunities:

```text
Model
 │
 ├── Quantization
 │
 ├── Kernel Fusion
 │
 ├── FlashAttention
 │
 ├── KV Cache Optimization
 │
 ├── Continuous Batching
 │
 ├── Prefix Caching
 │
 ├── CUDA Graphs
 │
 ├── Tensor Parallelism
 │
 └── Speculative Decoding
```

---

# 🏗️ Repository Structure

```text
cuda-gpu-engineering/
│
├── README.md
│
├── pyproject.toml
├── CMakeLists.txt
│
├── docs/
│   ├── gpu_architecture/
│   ├── cuda/
│   ├── memory/
│   ├── performance/
│   ├── multi_gpu/
│   └── llm/
│
├── phase_00_gpu_fundamentals/
│
├── phase_01_gpu_architecture/
│
├── phase_02_cuda_programming/
│
├── phase_03_cuda_memory/
│
├── phase_04_cuda_parallelism/
│
├── phase_05_cuda_performance/
│
├── phase_06_advanced_cuda/
│
├── phase_07_pytorch_cuda/
│
├── phase_08_triton/
│
├── phase_09_tensor_cores/
│
├── phase_10_multi_gpu/
│
├── phase_11_llm_gpu/
│
├── phase_12_production/
│
├── projects/
│   ├── project_01_gpu_info/
│   ├── project_02_vector_operations/
│   ├── project_03_matrix_multiplication/
│   ├── project_04_reduction/
│   ├── project_05_softmax/
│   ├── project_06_layernorm/
│   ├── project_07_attention/
│   ├── project_08_flash_attention/
│   ├── project_09_pytorch_cuda/
│   ├── project_10_triton/
│   ├── project_11_kv_cache/
│   ├── project_12_nccl/
│   ├── project_13_multi_gpu_llm/
│   └── project_14_production_inference/
│
├── benchmarks/
│
├── profiling/
│
└── tests/
```

---

# 🧪 Testing Strategy

Each implementation should have:

```text
Unit Tests
    ↓
Correctness Tests
    ↓
Numerical Comparison
    ↓
Benchmark
    ↓
Profiling
```

For numerical kernels we should compare against a trusted implementation.

Example:

```python
expected = torch.softmax(x, dim=-1)

actual = custom_softmax(x)

torch.testing.assert_close(
    actual,
    expected
)
```

Performance tests should be separated from correctness tests.

---

# 📈 Progress Tracking

## Phase Progress

| Phase | Topic                         | Status |
| ----- | ----------------------------- | ------ |
| 00    | GPU Fundamentals              | ⬜      |
| 01    | GPU Architecture              | ⬜      |
| 02    | CUDA Programming              | ⬜      |
| 03    | CUDA Memory                   | ⬜      |
| 04    | CUDA Parallelism              | ⬜      |
| 05    | CUDA Performance              | ⬜      |
| 06    | Advanced CUDA                 | ⬜      |
| 07    | PyTorch + CUDA                | ⬜      |
| 08    | Triton                        | ⬜      |
| 09    | Tensor Cores                  | ⬜      |
| 10    | Multi-GPU                     | ⬜      |
| 11    | LLM GPU Optimization          | ⬜      |
| 12    | Production GPU Infrastructure | ⬜      |

Legend:

```text
⬜ Not Started
🟡 In Progress
🟢 Completed
```

---

# 🎯 Final Competency

At the end of this roadmap, the target architecture knowledge is:

```text
                    GPU Hardware
                         │
                         ▼
                 GPU Architecture
                         │
                         ▼
                    CUDA Model
                         │
                         ▼
                 Memory Hierarchy
                         │
                         ▼
                   CUDA Kernels
                         │
                         ▼
                Performance Tuning
                         │
                         ▼
                   CUDA Profiling
                         │
                         ▼
                 PyTorch / Triton
                         │
                         ▼
                    Tensor Cores
                         │
                         ▼
                  LLM Kernels
                         │
                         ▼
               KV Cache / Attention
                         │
                         ▼
                 Inference Engines
                         │
                         ▼
                 Multi-GPU / NCCL
                         │
                         ▼
                    Kubernetes
                         │
                         ▼
              Production GPU Systems
```

---

# 🧑‍💻 Engineering Principles

Throughout this roadmap:

### 1. Correctness before optimization

```text
Correct
  ↓
Benchmark
  ↓
Profile
  ↓
Optimize
```

Never optimize a kernel without verifying correctness.

---

### 2. Measure instead of guessing

Do not assume:

> "This should be faster."

Measure:

```text
Before
  ↓
Optimization
  ↓
After
```

---

### 3. Understand the bottleneck first

```text
No bottleneck identified
        ↓
Don't optimize yet
```

---

### 4. Understand hardware implications

Every optimization should answer:

```text
What happens to:

Memory traffic?
Compute?
Registers?
Shared memory?
Occupancy?
Synchronization?
Communication?
```

---

### 5. Optimize the system, not just the kernel

Production performance is:

```text
Application
    +
Framework
    +
Kernel
    +
Memory
    +
GPU
    +
Communication
    +
Scheduling
```

---

# 🎤 Interview Preparation

Every major topic will include interview questions.

Examples:

### CUDA

* What is a CUDA kernel?
* What is a warp?
* Why does NVIDIA use 32-thread warps?
* What is occupancy?
* What causes warp divergence?
* What is memory coalescing?
* Shared memory vs global memory?
* What is a race condition?
* What does `__syncthreads()` do?

### Performance

* How do you determine whether a kernel is compute-bound?
* What is arithmetic intensity?
* What is the roofline model?
* Why doesn't higher occupancy always mean higher performance?
* How do you profile a CUDA kernel?
* How do you reduce GPU memory traffic?

### LLM

* Why is decode memory-bandwidth bound?
* What is KV Cache?
* Why does GQA reduce memory?
* Why does FlashAttention improve performance?
* What is PagedAttention?
* What is continuous batching?
* How does tensor parallelism work?

### Distributed GPU

* What is NCCL?
* What is AllReduce?
* NVLink vs PCIe?
* Tensor parallelism vs pipeline parallelism?
* What happens when inter-GPU communication becomes the bottleneck?

---

# 🏁 End Goal

The final capability we are targeting is:

> **Given a production LLM workload, independently reason from application → framework → CUDA kernel → GPU architecture → memory → communication → infrastructure, identify bottlenecks, profile the system, optimize it, and validate the improvement with measurable benchmarks.**

This roadmap intentionally connects **low-level GPU engineering** with the **modern GenAI stack**:

```text
CUDA
 ↓
PyTorch
 ↓
Triton
 ↓
Tensor Cores
 ↓
FlashAttention
 ↓
KV Cache
 ↓
vLLM / SGLang / TensorRT-LLM
 ↓
NCCL
 ↓
Multi-GPU
 ↓
Kubernetes
 ↓
Production LLM Serving
```

---

## Current Status

**Current Phase:** Phase 00 — GPU Fundamentals

**Current Lesson:** 0.1 — What is a GPU?

**Next:** Build the GPU mental model from the hardware level upward.
