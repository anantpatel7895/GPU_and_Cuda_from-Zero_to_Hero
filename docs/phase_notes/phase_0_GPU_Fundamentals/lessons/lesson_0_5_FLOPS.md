# Lesson 0.5 — FLOPS

## 1. Definition

**FLOP** = Floating-Point Operation
**FLOPS** = Floating-Point Operations Per Second

```text
1 GFLOPS = 10⁹ FLOPS
1 TFLOPS = 10¹² FLOPS
1 PFLOPS = 10¹⁵ FLOPS
```

---

## 2. Why FLOPS Matter

Deep learning performs huge numbers of:

* Matrix multiplications
* Multiply-accumulate operations
* Tensor operations

Therefore, GPUs need high floating-point compute capability.

---

## 3. Matrix Multiplication

For:

```text
A = M × K
B = K × N
```

Approximate computation:

$$
FLOPs \approx 2MKN
$$

The `2` comes from:

```text
Multiplication + Addition = 2 FLOPs
```

---

## 4. FLOPS and Precision

Peak FLOPS depend on numerical precision.

Common formats:

```text
FP32
FP16
BF16
TF32
FP8
```

Generally:

```text
Lower Precision
      ↓
Less memory per value
      ↓
Higher potential throughput
      ↓
Higher FLOPS
```

Modern NVIDIA GPUs can achieve much higher AI throughput at lower precision using Tensor Cores.

> Higher FLOPS at lower precision does not automatically mean better accuracy.

---

## 5. Tensor Cores

Tensor Cores are specialized NVIDIA hardware for high-throughput matrix operations.

```text
Matrix Multiplication
        ↓
Tensor Cores
        ↓
High Compute Throughput
```

They are especially important for modern deep learning and LLM workloads.

---

## 6. FLOPS vs Memory Bandwidth

**FLOPS** measures compute capability.

**Memory Bandwidth** measures how quickly data can move between memory and compute.

```text
GPU
│
├── Compute
│    └── FLOPS
│
└── Memory
     └── GB/s
```

A workload can have high theoretical FLOPS but still be slow if memory cannot supply data fast enough.

```text
Memory
  ↓
Data arrives slowly
  ↓
Compute units wait
  ↓
Actual performance ↓
```

---

## 7. FLOPS and Arithmetic Intensity

Arithmetic Intensity measures computation relative to memory movement:

$$
Arithmetic\ Intensity =
\frac{FLOPs}{Bytes\ Transferred}
$$

Example:

```text
1,000 FLOPs
100 Bytes transferred

Arithmetic Intensity = 10 FLOPs/Byte
```

### High Arithmetic Intensity

```text
More computation
+
Less memory traffic
        ↓
Potentially Compute-bound
```

### Low Arithmetic Intensity

```text
Less computation
+
More memory traffic
        ↓
Potentially Memory-bound
```

---

## 8. Peak FLOPS ≠ Real Performance

A GPU may advertise:

```text
100 TFLOPS
```

but an application may achieve much less because of:

* Memory bandwidth
* Memory access pattern
* Kernel efficiency
* Insufficient parallelism
* Synchronization
* Communication

---

## 9. Key Takeaway

```text
FLOPS
   +
Memory Bandwidth
   +
Arithmetic Intensity
   +
Kernel Efficiency
   =
Actual GPU Performance
```

> **FLOPS tells us the compute capacity of the GPU, not the actual performance of our application.**
