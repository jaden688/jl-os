# JL OS Crazy Linux: Multi-Microkernel AI Inference Engine OS

![JL OS System](https://img.shields.bg/badge/Kernel-Multi--Microkernel-cyan)
![Inference Engine](https://img.shields.bg/badge/Inference-Inline--Native-emerald)
![IPC](https://img.shields.bg/badge/IPC-Zero--Copy--Ring--Buffer-purple)

**JL OS** is an experimental OS architecture built to execute real-time quantized LLM and Tensor inference directly across decoupled multi-microkernel cores.

---

## ⚡ Architecture Breakdown

JL OS splits low-level OS operations and neural inference into specialized microkernel layers:

1. **`MK-1` Tensor Matrix Engine (`mk1_tensor`)**
   - Direct hardware GEMM matrix multiplication & FP16/INT4 tensor dispatch.
2. **`MK-2` KV-Cache & Attention Scheduler (`mk2_attn`)**
   - High-speed Paged Attention allocation, KV-Cache indexing, and token context memory.
3. **`MK-3` Synapse Zero-Copy Router (`mk3_synapse`)**
   - Zero-copy shared memory IPC ring-buffer with sub-microsecond latency.

---

## 🚀 Quickstart & Usage

### 1. Run native C Microkernel Engine
```bash
gcc -O3 -o jl_kernel.exe jl_kernel.c
.\jl_kernel.exe "Synthesize multi-microkernel tensor pipeline"
```

### 2. Launch Cybernetic HUD UI Supervisor
```bash
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser.
