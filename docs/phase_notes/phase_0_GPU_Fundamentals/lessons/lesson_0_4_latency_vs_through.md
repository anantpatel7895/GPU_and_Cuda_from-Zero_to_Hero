# Phase 00 — GPU Fundamentals

# Lesson 0.4 — Latency vs Throughput

## 1. Core Definition

### Latency

> **Latency = How long one request takes to complete.**

Example:

```text
Request → Processing → Response

Latency = 200 ms
```

---

### Throughput

> **Throughput = How much work the system completes per unit of time.**

Example:

```text
100 requests
      ↓
   1 second

Throughput = 100 requests/sec
```

For LLMs, throughput is often measured as:

```text
tokens/sec
```

---

# 2. Simple Example

Suppose an LLM processes 3 requests:

```text
Request A → 2 sec
Request B → 2 sec
Request C → 2 sec
```

Sequential execution:

```text
A ──2s──► B ──2s──► C ──2s──►

Total = 6 seconds
```

Throughput:

$$
\frac{3\ requests}{6\ sec}
= 0.5\ requests/sec
$$

---

# 3. Latency vs Throughput

| Metric         | Meaning                     |          Example |
| -------------- | --------------------------- | ---------------: |
| Latency        | Time for one request        |           200 ms |
| Throughput     | Work per unit time          |      100 req/sec |
| LLM throughput | Tokens processed per second | 1,000 tokens/sec |

---

# 4. Batch Size and Throughput

Increasing batch size can provide more parallel work:

```text
Batch Size ↑
     ↓
More parallel work
     ↓
GPU utilization ↑
     ↓
Throughput ↑
```

Example:

```text
Batch = 1
Throughput ≈ 100 tokens/sec

Batch = 8
Throughput ≈ 700 tokens/sec

Batch = 16
Throughput ≈ 1,100 tokens/sec
```

> These numbers are illustrative; actual performance depends on model, GPU, sequence length, precision, and implementation.

However:

```text
Batch Size ↑
     ↓
Memory usage ↑
     ↓
Queueing/waiting may ↑
     ↓
Latency may ↑
```

---

# 5. LLM Latency Metrics

LLM inference has several important latency metrics.

## TTFT — Time To First Token

Time from request arrival until the first generated token.

```text
Request
   │
   ├──────────────► First Token
   │
   └── TTFT ──────┘
```

Example:

```text
TTFT = 300 ms
```

Important for interactive applications.

---

## Inter-Token Latency

Time between generated tokens.

```text
Token 1 ──50ms──► Token 2 ──50ms──► Token 3
```

Example:

```text
Inter-token latency = 50 ms
```

Approximately:

$$
Tokens/sec = \frac{1}{0.05}
= 20\ tokens/sec
$$

---

## End-to-End Latency

Total time from request arrival until the final token.

```text
Request
   │
   ▼
First Token
   │
   ▼
Token 2
   │
   ▼
Token 3
   │
   ▼
Final Token
```

Example:

```text
Total latency = 4.5 seconds
```

---

# 6. LLM Throughput Example

Suppose a server generates:

```text
10,000 output tokens
```

in:

```text
10 seconds
```

Then:

$$
Throughput =
\frac{10,000}{10}
=
1,000\ tokens/sec
$$

So:

```text
Throughput = 1,000 tokens/sec
```

---

# 7. Prefill vs Decode

LLM inference:

```text
Prompt
  ↓
Prefill
  ↓
First Token
  ↓
Decode
  ↓
Token → Token → Token
```

Generally:

```text
TTFT
 ↓
Strongly influenced by Prefill

Generation speed
 ↓
Strongly influenced by Decode
```

The exact bottleneck depends on:

* Model architecture
* GPU
* Batch size
* Context length
* KV cache
* Precision
* Kernel implementation

---

# 8. Latency Percentiles

Production systems should not monitor only average latency.

Common metrics:

```text
P50
P95
P99
```

Example:

```text
P50 = 200 ms
P95 = 500 ms
P99 = 1,200 ms
```

Meaning approximately:

```text
50% of requests ≤ 200 ms
95% of requests ≤ 500 ms
99% of requests ≤ 1,200 ms
```

P95/P99 are important for understanding **tail latency**.

---

# 9. Workload Comparison

| Workload              | Important Metrics                 |
| --------------------- | --------------------------------- |
| Chatbot               | TTFT, inter-token latency         |
| Real-time API         | P95/P99 latency                   |
| Batch inference       | Throughput                        |
| Offline summarization | Tokens/sec                        |
| LLM serving           | Latency + throughput              |
| Large-scale serving   | Latency + throughput + cost/token |

---

# 10. Important Trade-off

```text
                 Performance
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
       Latency              Throughput
          │                     │
     Fast response          More work/sec
          │                     │
          └──────────┬──────────┘
                     ▼
              GPU Efficiency
```

A system can have:

```text
High throughput
+
High latency
```

For example, aggressive batching may process many tokens efficiently while increasing queueing/wait time for individual requests.

---

# 11. Key Takeaways

1. **Latency** = time taken by a request.
2. **Throughput** = amount of work completed per unit time.
3. Larger batches can improve throughput.
4. Larger batches can increase memory usage and latency.
5. **TTFT** is important for interactive LLM applications.
6. **Inter-token latency** affects how quickly generated text appears.
7. **Tokens/sec** is a common LLM throughput metric.
8. **P95/P99** are important production latency metrics.
9. Prefill primarily affects TTFT.
10. Decode strongly affects token-generation speed.
11. High GPU utilization does not automatically mean low latency.
12. Production LLM systems must balance **latency, throughput, GPU utilization, memory, and cost**.

---

## Interview Question

### Can a system have high throughput but high latency?

**Yes.**

For example:

```text
Large batches
    ↓
High GPU utilization
    ↓
High throughput

BUT

Requests wait for batching
    ↓
Higher individual latency
```

Therefore:

> **Throughput optimization and latency optimization are related but different engineering problems.**
