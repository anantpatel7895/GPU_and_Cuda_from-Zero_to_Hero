## 14. GPU vs CUDA vs PyTorch

Keep these three layers separate:

```text
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

Later we'll add:

```text
PyTorch
	 ↓
Triton
	 ↓
CUDA
	 ↓
GPU
```

And:

```text
vLLM
	 ↓
PyTorch / custom kernels / Triton / CUDA
	 ↓
GPU
```
