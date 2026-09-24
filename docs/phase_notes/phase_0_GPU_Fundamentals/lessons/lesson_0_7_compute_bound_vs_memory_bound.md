# Lesson 0.7 — Compute-Bound vs Memory-Bound

## 1. Core Idea

The key question in GPU performance is:

> **What is limiting performance — computation or data movement?**

```text
GPU Workload
     │
     ├── Compute-bound
     │      ↓
     │   Computation is the bottleneck
     │
     └── Memory-bound
            ↓
         Data movement is the bottleneck
```

---

## 2. Compute-Bound

A workload is **compute-bound** when the GPU spends most of its time performing calculations.

```text
Memory
  ↓
Load
  ↓
Heavy Computation
  ↓
Store
```

Example:

```text
Large Matrix Multiplication
```

The GPU performs a huge number of multiply-add operations.

---

## 3. Memory-Bound

A workload is **memory-bound** when data movement is the primary bottleneck.

```text
Memory
  ↓
Load
  ↓
Small Computation
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

---

## 4. Simple Comparison

### Element-wise operation

```python
y = x + 1
```

For each element:

```text
Read → Add → Write
```

Very little computation compared with data movement.

→ Often memory-bound.

### Matrix multiplication

```text
C = A × B
```

Large number of multiply-add operations.

→ Often compute-intensive.

---

## 5. Arithmetic Intensity

The key metric is:

$$
Arithmetic\ Intensity =
\frac{FLOPs}{Bytes\ Transferred}
$$

Example:

```text
10,000 FLOPs
1,000 Bytes

AI = 10 FLOPs/Byte
```

### Low Arithmetic Intensity

```text
Low FLOPs
+
High memory traffic
        ↓
Memory-bound tendency
```

### High Arithmetic Intensity

```text
High FLOPs
+
Less memory traffic
        ↓
Compute-bound tendency
```

---

## 6. FLOPS vs Memory Bandwidth

Suppose a GPU has:

```text
Compute = 100 TFLOPS
Memory Bandwidth = 1 TB/s
```

If the workload cannot get data from memory fast enough:

```text
Memory
  ↓
Data arrives slowly
  ↓
Compute units wait
  ↓
Performance decreases
```

Adding more compute capacity will not necessarily solve the problem.

---

## 7. Roofline Model

The Roofline Model connects:

```text
Compute Throughput
        +
Memory Bandwidth
        +
Arithmetic Intensity
```

Conceptually:

```text
Performance
   │
   │             Compute Limit
   │          ───────────────────
   │        /
   │      /
   │    /
   │  /
   │ /
   └──────────────────────────────►
       Arithmetic Intensity
```

The workload is limited by either:

```text
Memory bandwidth
        OR
Compute throughput
```

---

## 8. LLM Example

### Prefill

```text
Prompt
  ↓
Large Matrix Operations
  ↓
GPU
```

Prefill can often be relatively compute-intensive.

### Decode

```text
Generate Token
      ↓
Access Weights / KV Cache
      ↓
Compute
      ↓
Next Token
```

Decode can be strongly affected by:

* Memory bandwidth
* KV-cache access
* Batch size
* Kernel efficiency

The exact bottleneck depends on the workload and hardware.

---

## 9. Optimization

### If Compute-Bound

Focus on:

* Tensor Cores
* Mixed precision
* Better kernels
* Kernel fusion
* Parallelism
* GPU utilization

### If Memory-Bound

Focus on:

* Memory access patterns
* Coalesced access
* Data reuse
* Cache utilization
* Shared memory
* Kernel fusion
* Reducing memory traffic
* Quantization

---

## 10. Production Debugging Mental Model

When a GPU application is slow:

```text
Is it Compute-bound?
        │
        ├── YES
        │    ↓
        │ Optimize computation
        │
        └── NO
             ↓
        Is it Memory-bound?
             │
             ├── YES
             │    ↓
             │ Optimize data movement
             │
             └── NO
                  ↓
             Check:
             • Kernel launch
             • Synchronization
             • Communication
             • CPU bottlenecks
```

---

## 11. Key Takeaway

> **Compute-bound = computation is the bottleneck.**

> **Memory-bound = data movement is the bottleneck.**

```text
FLOPS
  +
Memory Bandwidth
  +
Arithmetic Intensity
      ↓
Bottleneck
      ↓
Actual GPU Performance
```
