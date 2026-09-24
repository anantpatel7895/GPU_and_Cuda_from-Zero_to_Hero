# Lesson 0.8 — Arithmetic Intensity

## 1. Definition

> **Arithmetic Intensity = How much computation is performed for every byte of data moved.**

$$
Arithmetic\ Intensity =
\frac{FLOPs}{Bytes\ Transferred}
$$

Unit:

```text
FLOPs / Byte
```

---

## 2. Example

```text
FLOPs = 1,000,000
Bytes Transferred = 100,000
```

$$
AI = \frac{1,000,000}{100,000}
= 10\ FLOPs/Byte
$$

Meaning:

```text
10 FLOPs
for every
1 Byte transferred
```

---

## 3. Why It Matters

GPU performance has two major limits:

```text
GPU
├── Compute Throughput
│      └── TFLOPS
│
└── Memory Bandwidth
       └── TB/s
```

Arithmetic intensity helps identify the likely bottleneck.

```text
Low AI
 ↓
Memory-bound tendency

High AI
 ↓
Compute-bound tendency
```

---

## 4. Simple Example

### Element-wise operation

```python
y = x + 1
```

```text
Read → Add → Write
```

Very little computation compared with memory movement.

```text
Low AI
  ↓
Likely memory-bound
```

### Matrix multiplication

```text
C = A × B
```

Large matrix multiplication performs many calculations while reusing data.

```text
High AI
  ↓
Potentially compute-bound
```

---

## 5. Machine Balance Point

Suppose:

```text
GPU Compute = 100 TFLOPS
Memory Bandwidth = 1 TB/s
```

Then:

$$
Machine\ Balance =
\frac{100\ TFLOPS}{1\ TB/s}
$$

$$
= 100\ FLOPs/Byte
$$

Simplified interpretation:

```text
AI < 100 FLOPs/Byte
        ↓
Memory-bound tendency

AI > 100 FLOPs/Byte
        ↓
Compute-bound tendency
```

This is the basic idea behind the **Roofline Model**.

---

## 6. Roofline Model

```text
Performance
   │
   │              Compute Limit
   │           ───────────────────
   │         /
   │       /
   │     /
   │   /
   │ /
   └──────────────────────────────►
       Arithmetic Intensity

       Memory-bound → Compute-bound
```

The workload is limited by either:

```text
Memory Bandwidth
       OR
Compute Throughput
```

---

## 7. LLM Connection

Transformers perform many matrix operations:

```text
X × WQ
X × WK
X × WV
Q × Kᵀ
Attention × V
X × W1
X × W2
```

These can have high arithmetic intensity, especially with sufficiently large workloads.

However, LLM decode can become more memory-sensitive because of:

```text
Model Weights
+
KV Cache
+
Memory Access
```

Therefore:

> **LLM performance depends on workload shape, not just theoretical FLOPS.**

---

## 8. Increasing Arithmetic Intensity

A common strategy is to perform more computation per loaded byte.

Techniques include:

* Data reuse
* Tiling
* Shared memory
* Cache reuse
* Kernel fusion
* Efficient matrix operations

Instead of:

```text
Load → Compute → Store
Load → Compute → Store
Load → Compute → Store
```

Try to reuse data:

```text
Load
 ↓
Compute
 ↓
Compute
 ↓
Compute
 ↓
Store
```

This reduces memory traffic relative to computation.

---

## 9. Key Takeaway

$$
\boxed{
Arithmetic\ Intensity =
\frac{FLOPs}{Bytes}
}
$$

```text
Low AI
 ↓
Memory-bound tendency

High AI
 ↓
Compute-bound tendency
```

### Mental Model

```text
FLOPS
   +
Memory Bandwidth
   +
Arithmetic Intensity
        ↓
Performance Limit
```

### Interview Question

**Why is arithmetic intensity important?**

> It helps determine whether a workload is more likely to be limited by GPU compute throughput or memory bandwidth, allowing us to choose the appropriate optimization strategy.

