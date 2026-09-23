## 1. GPU execution hierarchy

For now, remember this hierarchy:

```text
GPU
 │
 ├── SM
 │    │
 │    ├── Warp
 │    │     │
 │    │     └── Threads
 │    │
 │    ├── Registers
 │    ├── Shared Memory
 │    └── Execution Units
 │
 ├── L2 Cache
 │
 └── Global Memory
```

Don't worry about understanding every component yet.

We'll build each layer one by one.

---

## 2. What happens when we execute GPU code?

Eventually, a flow like:

```python
x = torch.randn(4096, 4096, device="cuda")
```

will make much more sense.

Conceptually:

```text
Python
	│
	▼
PyTorch
	│
	▼
CUDA Runtime
	│
	▼
GPU Driver
	│
	▼
GPU
	│
	├── Allocate GPU memory
	│
	└── Execute kernels
```

And:

```python
y = x @ weight
```

eventually results in GPU kernels executing matrix operations.

Later we'll go deep into exactly how PyTorch reaches CUDA.

---

## 3. Important correction: GPU != CUDA

These are different things.

### GPU

Hardware.

Examples:

```text
NVIDIA A10
NVIDIA A100
NVIDIA H100
NVIDIA H200
NVIDIA B200
```

### CUDA

NVIDIA's GPU computing platform and programming ecosystem.

```text
CUDA
 │
 ├── CUDA Runtime
 ├── CUDA Libraries
 ├── CUDA Compiler
 ├── CUDA APIs
 ├── CUDA Toolkit
 └── CUDA Programming Model
```

Therefore:

```text
GPU = Hardware

CUDA = Software/programming ecosystem for NVIDIA GPUs
```

This distinction is fundamental.
