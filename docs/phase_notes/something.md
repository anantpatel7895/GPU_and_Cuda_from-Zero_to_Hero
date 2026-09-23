# GPU ≠ CUDA

### GPU is a Hardware component


- NVIDIA A10
- NVIDIA A100
- NVIDIA H100
- NVIDIA H200
- NVIDIA B200

### NVIDIA's GPU computing platform and programming ecosystem.

```
CUDA
 │
 ├── CUDA Runtime
 ├── CUDA Libraries
 ├── CUDA Compiler
 ├── CUDA APIs
 ├── CUDA Toolkit
 └── CUDA Programming Model
```

```
GPU = Hardware

CUDA = Software/programming ecosystem for NVIDIA GPUs
```

### GPU vs CUDA vs PyTorch

```
┌──────────────────────────────┐
│          PyTorch             │
│   High-level ML framework    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│            CUDA              │
│ Runtime / Libraries / APIs   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│          NVIDIA GPU          │
│       Hardware / SMs         │
└──────────────────────────────┘
```

or 

```
┌──────────────────────────────┐
│          PyTorch             │
│   High-level ML framework    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│          triton              │
│   High-level ML framework    │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│            CUDA              │
│ Runtime / Libraries / APIs   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│          NVIDIA GPU          │
│       Hardware / SMs         │
└──────────────────────────────┘
```


