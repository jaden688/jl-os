import React, { useState, useEffect, useRef } from 'react';
import { 
  Cpu, 
  Zap, 
  Activity, 
  Terminal, 
  Layers, 
  ShieldAlert, 
  Play, 
  RotateCcw, 
  Server, 
  Radio,
  Sliders,
  Database,
  CheckCircle2,
  HardDrive
} from 'lucide-react';

const VOCAB_TOKENS = [
  "SYNAPSE", "KERNEL_OK", "TENSOR", "QUANT_Q4", "AO_MATRIX", "VECTOR_X", "FLOP_BURST",
  "INFERENCE", "LATENCY_SUB_NS", "ATTENTION_V2", "NEURAL_NET", "HYPERPLANE", "CORE_MK1", 
  "CORE_MK2", "CORE_MK3", "PAGED_ATTN", "KV_CACHE_HIT", "TOKEN_GEN", "DISPATCH_OK", 
  "ZERO_COPY_RING", "STREAM_FAST", "OVERCLOCK_MAX", "NANO_SYNC", "RING_BUS_ACTIVE", 
  "EMBEDDING_128D", "SOFTMAX_FUSED", "WEIGHT_INT4", "TERAFLOPS", "FRANKEN_AO_OS", 
  "HYPER_PIPELINE", "SYS_READY", "NODE_HEALTHY", "L1_CACHE_DIRECT", "MULTITHREAD_RUN"
];

export default function App() {
  // Microkernel states
  const [kernels, setKernels] = useState([
    {
      id: 'MK1',
      name: 'MK-1 Tensor Matrix Engine',
      type: 'GEMM & Vector Compute',
      threads: 8,
      load: 34,
      memory: 4096,
      ops: 142090,
      status: 'ONLINE',
      color: 'from-cyan-500 to-blue-600',
      textColor: 'text-cyan-400',
      borderColor: 'border-cyan-500/40',
      glow: 'shadow-cyan-500/20'
    },
    {
      id: 'MK2',
      name: 'MK-2 KV-Cache & Attention Scheduler',
      type: 'Paged Attention & Context',
      threads: 4,
      load: 48,
      memory: 8192,
      ops: 98402,
      status: 'ONLINE',
      color: 'from-purple-500 to-pink-600',
      textColor: 'text-purple-400',
      borderColor: 'border-purple-500/40',
      glow: 'shadow-purple-500/20'
    },
    {
      id: 'MK3',
      name: 'MK-3 Synapse Zero-Copy Router',
      type: 'Ring-Buffer IPC & Neural Bus',
      threads: 6,
      load: 22,
      memory: 2048,
      ops: 312040,
      status: 'ONLINE',
      color: 'from-emerald-400 to-green-600',
      textColor: 'text-emerald-400',
      borderColor: 'border-emerald-500/40',
      glow: 'shadow-emerald-500/20'
    }
  ]);

  // IPC Ring Messages
  const [ipcLogs, setIpcLogs] = useState([]);
  const [prompt, setPrompt] = useState('Synthesize deep multi-kernel matrix inline inference pipeline');
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedTokens, setGeneratedTokens] = useState([]);
  const [overclock, setOverclock] = useState(false);

  // System Specs & Metrics
  const [metrics, setMetrics] = useState({
    tps: 184.2,
    latency: 0.312,
    ipcRate: 14200,
    totalTokens: 89400
  });

  const terminalEndRef = useRef(null);

  // Periodic Kernel Load fluctuate simulation
  useEffect(() => {
    const interval = setInterval(() => {
      setKernels(prev => prev.map(k => {
        const baseLoad = overclock ? 80 : (isGenerating ? 65 : 25);
        const loadVariation = Math.floor(Math.random() * 25) - 12;
        const newLoad = Math.min(99, Math.max(10, baseLoad + loadVariation));
        return {
          ...k,
          load: newLoad,
          ops: k.ops + Math.floor(Math.random() * (overclock ? 1200 : 300))
        };
      }));

      setMetrics(prev => ({
        ...prev,
        tps: overclock ? (280 + Math.random() * 40) : (160 + Math.random() * 30),
        latency: overclock ? (0.12 + Math.random() * 0.08) : (0.28 + Math.random() * 0.12),
        ipcRate: prev.ipcRate + Math.floor(Math.random() * 50)
      }));
    }, 800);

    return () => clearInterval(interval);
  }, [isGenerating, overclock]);

  // Run Inference Stream Simulation
  const handleStartInference = () => {
    if (!prompt.trim()) return;
    setIsGenerating(true);
    setGeneratedTokens([]);
    
    let count = 0;
    const maxTokens = overclock ? 32 : 20;

    const tokenInterval = setInterval(() => {
      const randomToken = VOCAB_TOKENS[Math.floor(Math.random() * VOCAB_TOKENS.length)];
      
      setGeneratedTokens(prev => [...prev, randomToken]);
      
      // Inject IPC Routing log
      const senderIdx = count % 3;
      const receiverIdx = (count + 1) % 3;
      const sender = kernels[senderIdx].id;
      const receiver = kernels[receiverIdx].id;

      const newIPC = {
        id: Date.now() + Math.random(),
        time: new Date().toLocaleTimeString() + '.' + Math.floor(Math.random() * 900 + 100),
        sender,
        receiver,
        opcode: senderIdx === 0 ? 'EXEC_GEMM' : (senderIdx === 1 ? 'KV_CACHE_LOOKUP' : 'TOKEN_ROUTE'),
        payload: randomToken,
        latencyUs: (0.15 + Math.random() * 0.25).toFixed(3)
      };

      setIpcLogs(prev => [newIPC, ...prev.slice(0, 49)]);

      setMetrics(m => ({ ...m, totalTokens: m.totalTokens + 1 }));

      count++;
      if (count >= maxTokens) {
        clearInterval(tokenInterval);
        setIsGenerating(false);
      }
    }, overclock ? 100 : 180);
  };

  return (
    <div className="min-h-screen bg-cyber-grid scanline relative text-slate-200 font-sans p-4 md:p-6 flex flex-col gap-6">
      
      {/* HEADER / OS BAR */}
      <header className="franken-panel rounded-xl p-4 flex flex-col md:flex-row justify-between items-start md:items-center border-l-4 border-l-cyan-400 gap-4">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-cyan-500/10 border border-cyan-500/30 rounded-lg animate-pulse-fast">
            <Cpu className="w-8 h-8 text-cyan-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="font-display text-2xl font-extrabold tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 via-emerald-400 to-purple-500">
                FRANKEN-AO OS
              </h1>
              <span className="px-2 py-0.5 text-xs font-mono font-bold bg-cyan-950 text-cyan-400 border border-cyan-500/40 rounded">
                v1.0.0-INFERENCE-KERNEL
              </span>
            </div>
            <p className="text-xs text-slate-400 font-mono mt-0.5">
              Multi-Microkernel Native AI Core • Zero-Copy Ring-Buffer IPC • Low-Latency Tensor OS
            </p>
          </div>
        </div>

        {/* TOP STATUS BAR */}
        <div className="flex flex-wrap items-center gap-4 text-xs font-mono">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-900/80 rounded-md border border-slate-700">
            <Activity className="w-4 h-4 text-emerald-400 animate-spin" style={{ animationDuration: '3s' }} />
            <span className="text-slate-400">INFERENCE TPS:</span>
            <span className="text-emerald-400 font-bold">{metrics.tps.toFixed(1)}</span>
          </div>

          <div className="flex items-center gap-2 px-3 py-1.5 bg-slate-900/80 rounded-md border border-slate-700">
            <Zap className="w-4 h-4 text-cyan-400" />
            <span className="text-slate-400">IPC LATENCY:</span>
            <span className="text-cyan-400 font-bold">{metrics.latency.toFixed(3)} μs</span>
          </div>

          <button 
            onClick={() => setOverclock(!overclock)}
            className={`flex items-center gap-2 px-4 py-1.5 rounded-md font-bold transition-all duration-300 border ${
              overclock 
                ? 'bg-gradient-to-r from-pink-600 to-purple-600 text-white border-pink-400 shadow-lg shadow-pink-500/40 animate-pulse' 
                : 'bg-slate-900 text-slate-300 border-slate-700 hover:border-slate-500'
            }`}
          >
            <Sliders className="w-4 h-4" />
            {overclock ? 'OVERCLOCK ACTIVE (3.8 GHz)' : 'ENABLE OVERCLOCK'}
          </button>
        </div>
      </header>

      {/* THREE MICROKERNELS ARCHITECTURE GRID */}
      <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {kernels.map((k) => (
          <div 
            key={k.id} 
            className={`franken-panel rounded-xl p-5 border ${k.borderColor} shadow-lg ${k.glow} relative overflow-hidden flex flex-col justify-between`}
          >
            <div className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl from-white/5 to-transparent pointer-events-none" />

            <div>
              {/* Header */}
              <div className="flex justify-between items-start mb-3">
                <div className="flex items-center gap-2">
                  <span className={`font-mono text-sm font-extrabold px-2 py-0.5 rounded bg-slate-900 ${k.textColor} border border-current`}>
                    {k.id}
                  </span>
                  <h2 className="font-mono text-sm font-bold text-slate-100">{k.name}</h2>
                </div>
                <span className="flex items-center gap-1 text-[10px] font-mono font-bold text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-500/30">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping" />
                  {k.status}
                </span>
              </div>

              <p className="text-xs font-mono text-slate-400 mb-4">{k.type}</p>

              {/* Load Bar */}
              <div className="space-y-1 mb-4">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-slate-400">Microkernel Workload</span>
                  <span className={`font-bold ${k.load > 85 ? 'text-pink-400' : k.textColor}`}>{k.load}%</span>
                </div>
                <div className="w-full h-2.5 bg-slate-900 rounded-full overflow-hidden border border-slate-800 p-0.5">
                  <div 
                    className={`h-full rounded-full bg-gradient-to-r ${k.color} transition-all duration-300`}
                    style={{ width: `${k.load}%` }}
                  />
                </div>
              </div>

              {/* Core Details */}
              <div className="grid grid-cols-2 gap-2 text-xs font-mono mb-4">
                <div className="bg-slate-900/60 p-2 rounded border border-slate-800">
                  <div className="text-slate-500 text-[10px]">CPU THREADS</div>
                  <div className="text-slate-200 font-bold flex items-center gap-1 mt-0.5">
                    <Layers className="w-3 h-3 text-cyan-400" />
                    {k.threads} Cores
                  </div>
                </div>

                <div className="bg-slate-900/60 p-2 rounded border border-slate-800">
                  <div className="text-slate-500 text-[10px]">ALLOCATED MEM</div>
                  <div className="text-slate-200 font-bold flex items-center gap-1 mt-0.5">
                    <Database className="w-3 h-3 text-purple-400" />
                    {k.memory} KB
                  </div>
                </div>
              </div>
            </div>

            {/* Total Kernel Ops */}
            <div className="pt-3 border-t border-slate-800/80 flex justify-between items-center text-xs font-mono">
              <span className="text-slate-500">Kernel Dispatches</span>
              <span className="text-slate-300 font-bold">{k.ops.toLocaleString()} ops</span>
            </div>
          </div>
        ))}
      </section>

      {/* MAIN INFERENCE & IPC INTERACTION SECTION */}
      <section className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1">
        
        {/* LEFT: INFERENCE RUNTIME CONTROLLER */}
        <div className="lg:col-span-7 franken-panel rounded-xl p-5 border border-cyan-500/30 flex flex-col justify-between gap-4">
          <div>
            <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Terminal className="w-5 h-5 text-cyan-400" />
                <h3 className="font-mono text-base font-bold text-slate-100">
                  Inline Neural Inference Terminal
                </h3>
              </div>
              <span className="text-xs font-mono text-slate-400 flex items-center gap-1">
                <Radio className="w-3.5 h-3.5 text-cyan-400 animate-pulse" /> Direct Kernel Stream
              </span>
            </div>

            {/* PROMPT INPUT */}
            <div className="space-y-2 mb-4">
              <label className="text-xs font-mono text-slate-400">INPUT INFERENCE PROMPT</label>
              <div className="flex gap-2">
                <input 
                  type="text" 
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  disabled={isGenerating}
                  placeholder="Type a neural instruction prompt..."
                  className="flex-1 bg-slate-900/90 border border-slate-700 focus:border-cyan-400 rounded-lg px-4 py-2.5 text-sm font-mono text-slate-100 outline-none transition-colors"
                />
                <button
                  onClick={handleStartInference}
                  disabled={isGenerating}
                  className={`px-5 py-2.5 rounded-lg font-mono font-bold text-sm flex items-center gap-2 transition-all ${
                    isGenerating
                      ? 'bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700'
                      : 'bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-black shadow-lg shadow-cyan-500/20'
                  }`}
                >
                  {isGenerating ? (
                    <>
                      <RotateCcw className="w-4 h-4 animate-spin" /> RUNNING...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4 fill-current" /> EXECUTE INFERENCE
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* TOKEN OUTPUT WINDOW */}
            <div className="space-y-2">
              <div className="flex justify-between text-xs font-mono text-slate-400">
                <span>INLINE TOKEN STREAM OUTPUT</span>
                <span>TOKENS GENERATED: <strong className="text-cyan-400">{generatedTokens.length}</strong></span>
              </div>
              <div className="min-h-[160px] max-h-[220px] bg-slate-950 p-4 rounded-lg border border-slate-800 font-mono text-sm overflow-y-auto flex flex-wrap gap-2 items-start">
                {generatedTokens.length === 0 ? (
                  <span className="text-slate-600 italic">Click 'EXECUTE INFERENCE' to trigger multi-microkernel forward pass...</span>
                ) : (
                  generatedTokens.map((tok, idx) => (
                    <span 
                      key={idx} 
                      className="px-2 py-1 bg-cyan-950/80 border border-cyan-500/40 text-cyan-300 rounded text-xs font-bold animate-pulse-fast"
                    >
                      {tok}
                    </span>
                  ))
                )}
              </div>
            </div>
          </div>

          {/* SUMMARY INFERENCE FOOTER */}
          <div className="grid grid-cols-3 gap-3 pt-3 border-t border-slate-800 text-xs font-mono">
            <div className="bg-slate-900/50 p-2 rounded">
              <span className="text-slate-500 block text-[10px]">TOTAL TOKENS</span>
              <span className="text-cyan-400 font-bold">{metrics.totalTokens.toLocaleString()}</span>
            </div>
            <div className="bg-slate-900/50 p-2 rounded">
              <span className="text-slate-500 block text-[10px]">QUANT MODEL</span>
              <span className="text-purple-400 font-bold">Franken-Q4_K_M</span>
            </div>
            <div className="bg-slate-900/50 p-2 rounded">
              <span className="text-slate-500 block text-[10px]">HARDWARE ACCEL</span>
              <span className="text-emerald-400 font-bold">AVX-512 / Tensor</span>
            </div>
          </div>
        </div>

        {/* RIGHT: ZERO-COPY IPC RING BUS MONITOR */}
        <div className="lg:col-span-5 franken-panel rounded-xl p-5 border border-purple-500/30 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-3 border-b border-slate-800 pb-3">
              <div className="flex items-center gap-2">
                <Server className="w-5 h-5 text-purple-400" />
                <h3 className="font-mono text-base font-bold text-slate-100">
                  Zero-Copy IPC Ring Bus Logs
                </h3>
              </div>
              <span className="text-[10px] font-mono bg-purple-950 text-purple-300 px-2 py-0.5 rounded border border-purple-500/40">
                RING SIZE: 256 BUF
              </span>
            </div>

            {/* IPC LOG LIST */}
            <div className="h-[280px] bg-slate-950 rounded-lg p-3 border border-slate-800 font-mono text-xs overflow-y-auto space-y-2">
              {ipcLogs.length === 0 ? (
                <div className="text-slate-600 italic text-center py-10">
                  Waiting for IPC ring buffer kernel traffic...
                </div>
              ) : (
                ipcLogs.map((log) => (
                  <div key={log.id} className="p-2 bg-slate-900/80 rounded border border-slate-800/80 flex flex-col gap-1">
                    <div className="flex justify-between items-center text-[10px]">
                      <span className="text-slate-500">{log.time}</span>
                      <span className="text-emerald-400 font-bold">{log.latencyUs} μs</span>
                    </div>
                    <div className="flex items-center gap-2 text-xs">
                      <span className="text-cyan-400 font-bold">{log.sender}</span>
                      <span className="text-slate-600">➔</span>
                      <span className="text-purple-400 font-bold">{log.receiver}</span>
                      <span className="ml-auto text-amber-400 font-bold">{log.opcode}</span>
                    </div>
                    <div className="text-[11px] text-slate-400 truncate">
                      Payload Token: <span className="text-slate-200 font-bold">{log.payload}</span>
                    </div>
                  </div>
                ))
              )}
              <div ref={terminalEndRef} />
            </div>
          </div>

          <div className="pt-3 border-t border-slate-800 flex justify-between items-center text-xs font-mono">
            <span className="text-slate-500">Shared Memory Ring Bus:</span>
            <span className="text-emerald-400 font-bold flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" /> 0.00% Packet Loss
            </span>
          </div>
        </div>

      </section>

      {/* FOOTER OS BAR */}
      <footer className="franken-panel rounded-xl px-4 py-3 flex flex-col md:flex-row justify-between items-center text-xs font-mono text-slate-500 gap-2">
        <div>
          FRANKEN-AO LINUX HYBRID OS • REGISTERED NODES: 3 KERNELS • ZERO-COPY RING IPC ENABLED
        </div>
        <div className="flex gap-4 text-slate-400">
          <span>HOST: x86_64-franken-linux</span>
          <span>BUILD: STABLE-DEBUG</span>
        </div>
      </footer>

    </div>
  );
}
