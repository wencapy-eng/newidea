import React, { useEffect, useMemo, useRef, useState } from 'react';
import { SYNC_CORR_THRESHOLD } from '../calib/frontend';
import type {
  AlgoVersion,
  ErrorHistoryPoint,
  LMTracePoint,
  SyncDiagnostics,
} from '../calib/types';
import {
  computeErrorChartBounds,
  errorHistoryFromLmTrace,
  normalizeErrorValue,
  summarizeErrorHistory,
} from '../utils/errorHistory';

interface DiagnosticCanvasesProps {
  algoVersion?: AlgoVersion;
  width: number;
  height: number;
  rgb: Uint8ClampedArray | null;
  depth: Float32Array | null;
  mask: Uint8Array | null;
  v3PosteriorOverlay?: Uint8Array | null;
  syncDiagnostics: SyncDiagnostics | null;
  lmTrace: LMTracePoint[];
  errorHistory?: ErrorHistoryPoint[];
  isRunning?: boolean;
}

/** Authentic Google Turbo colormap polynomial approximation for x in [0, 1] -> [r, g, b] in 0..255 */
function turboColormap(x: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, x));
  const r = Math.max(
    0,
    Math.min(
      1,
      0.13572138 +
        t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))))
    )
  );
  const g = Math.max(
    0,
    Math.min(
      1,
      0.09140261 +
        t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))))
    )
  );
  const b = Math.max(
    0,
    Math.min(
      1,
      0.1066733 +
        t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))))
    )
  );
  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

export const DiagnosticCanvases: React.FC<DiagnosticCanvasesProps> = ({
  algoVersion = 'v2',
  width,
  height,
  rgb,
  depth,
  mask,
  v3PosteriorOverlay,
  syncDiagnostics,
  lmTrace,
  errorHistory = [],
  isRunning = false,
}) => {
  const rgbCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const depthCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const syncCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const historyCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [useLogScale, setUseLogScale] = useState<boolean>(false);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);

  // Merge live errorHistory with fallback conversion from lmTrace if needed
  const effectiveHistory = useMemo<ErrorHistoryPoint[]>(() => {
    if (errorHistory.length > 0) return errorHistory;
    if (lmTrace.length > 0) {
      return errorHistoryFromLmTrace(lmTrace, algoVersion === 'v3' ? 'em_lm' : 'lm');
    }
    return [];
  }, [errorHistory, lmTrace, algoVersion]);

  const historySummary = useMemo(
    () => summarizeErrorHistory(effectiveHistory),
    [effectiveHistory]
  );

  // Reset hover index when history shrinks or resets
  useEffect(() => {
    if (hoverIdx !== null && hoverIdx >= effectiveHistory.length) {
      setHoverIdx(null);
    }
  }, [effectiveHistory.length, hoverIdx]);

  // 1. Virtual RGB canvas
  useEffect(() => {
    const canvas = rgbCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (!rgb || rgb.length !== width * height * 4) {
      ctx.fillStyle = '#090C12';
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#64748b';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText('等待虚拟 D435i RGB 帧...', 14, height / 2);
      return;
    }

    const imgData = new ImageData(new Uint8ClampedArray(rgb), width, height);
    ctx.putImageData(imgData, 0, 0);
  }, [rgb, width, height]);

  // 2. Turbo pseudo-color depth map + cyan active sub-chain mask overlay
  useEffect(() => {
    const canvas = depthCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    if (!depth || depth.length !== width * height) {
      ctx.fillStyle = '#090C12';
      ctx.fillRect(0, 0, width, height);
      ctx.fillStyle = '#64748b';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText('等待 24-bit 线性深度图...', 14, height / 2);
      return;
    }

    const out = new Uint8ClampedArray(width * height * 4);
    const minD = 0.55;
    const maxD = 2.65;

    for (let i = 0; i < width * height; i++) {
      const z = depth[i];
      const o = i * 4;
      if (z <= 0.08) {
        // Depth hole / invalid
        out[o] = 10;
        out[o + 1] = 14;
        out[o + 2] = 22;
        out[o + 3] = 255;
      } else {
        const norm = (z - minD) / (maxD - minD);
        const [r, g, b] = turboColormap(1 - Math.max(0, Math.min(1, norm)));
        const v3Cls =
          v3PosteriorOverlay && v3PosteriorOverlay.length === width * height
            ? v3PosteriorOverlay[i]
            : 0;

        if (v3Cls === 1) {
          // Class I: Nominal Rigid Manifold (Cyan)
          out[o] = Math.round(r * 0.35 + 6 * 0.65);
          out[o + 1] = Math.round(g * 0.35 + 225 * 0.65);
          out[o + 2] = Math.round(b * 0.35 + 245 * 0.65);
          out[o + 3] = 255;
        } else if (v3Cls === 2) {
          // Class II: Unmodeled Rigid Attachment (Amber)
          out[o] = Math.round(r * 0.25 + 245 * 0.75);
          out[o + 1] = Math.round(g * 0.25 + 166 * 0.75);
          out[o + 2] = Math.round(b * 0.25 + 35 * 0.75);
          out[o + 3] = 255;
        } else if (v3Cls === 3) {
          // Class III: Coupled Non-Rigid / Outlier (Magenta)
          out[o] = Math.round(r * 0.22 + 236 * 0.78);
          out[o + 1] = Math.round(g * 0.22 + 72 * 0.78);
          out[o + 2] = Math.round(b * 0.22 + 153 * 0.78);
          out[o + 3] = 255;
        } else if (v3Cls === 4) {
          // 2D Depth-Conditioned Front Occlusion Band M_occ_2D (Crimson tint)
          out[o] = Math.round(r * 0.45 + 225 * 0.55);
          out[o + 1] = Math.round(g * 0.45 + 29 * 0.55);
          out[o + 2] = Math.round(b * 0.45 + 72 * 0.55);
          out[o + 3] = 255;
        } else if (!v3PosteriorOverlay && mask && mask[i] === 1) {
          // V2 Cyan highlight overlay for active downstream sub-chain mask
          out[o] = Math.round(r * 0.35 + 6 * 0.65);
          out[o + 1] = Math.round(g * 0.35 + 225 * 0.65);
          out[o + 2] = Math.round(b * 0.35 + 245 * 0.65);
          out[o + 3] = 255;
        } else {
          out[o] = r;
          out[o + 1] = g;
          out[o + 2] = b;
          out[o + 3] = 255;
        }
      }
    }

    const imgData = new ImageData(out, width, height);
    ctx.putImageData(imgData, 0, 0);
  }, [depth, mask, v3PosteriorOverlay, width, height]);

  // 3. Timestamp synchronization cross-correlation waveform canvas
  useEffect(() => {
    const canvas = syncCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.fillStyle = '#090C12';
    ctx.fillRect(0, 0, W, H);

    const padL = 32;
    const padR = 12;
    const padT = 16;
    const padB = 22;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    // Grid lines
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(W - padR, y);
      ctx.stroke();
    }

    ctx.fillStyle = '#64748b';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText('1.0', 4, padT + 4);
    ctx.fillText('0.5', 4, padT + plotH * 0.5 + 3);
    ctx.fillText('0.0', 4, padT + plotH + 3);

    if (!syncDiagnostics || syncDiagnostics.encoderSpeed.length < 2) {
      ctx.fillStyle = '#64748b';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText(
        algoVersion === 'v3'
          ? '运行标定后显示多碎片因果同步曲线 (ρ≥0.40)'
          : '运行标定后显示同步相关性曲线 (ρ 阈值 0.42)',
        padL + 10,
        padT + plotH * 0.5
      );
      return;
    }

    const T = syncDiagnostics.encoderSpeed.length;
    const xOf = (t: number) => padL + (t / (T - 1)) * plotW;
    const yOf = (v: number) => padT + (1 - Math.max(0, Math.min(1, v))) * plotH;

    // Frame index ticks
    ctx.fillStyle = '#64748b';
    ctx.fillText('F0', padL - 4, H - 6);
    ctx.fillText(`F${Math.floor(T / 2)}`, xOf(Math.floor(T / 2)) - 8, H - 6);
    ctx.fillText(`F${T - 1}`, W - padR - 18, H - 6);

    // Draw rejected distractor components (red solid)
    const rejected = syncDiagnostics.components.filter((c) => !c.accepted).slice(0, 2);
    for (const comp of rejected) {
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 1.6;
      ctx.setLineDash([]);
      ctx.beginPath();
      comp.activity.forEach((v, idx) => {
        const x = xOf(idx);
        const y = yOf(v);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // Draw accepted robot components (emerald green solid)
    const accepted = syncDiagnostics.components.filter((c) => c.accepted).slice(0, 2);
    for (const comp of accepted) {
      ctx.strokeStyle = '#10b981';
      ctx.lineWidth = 2.0;
      ctx.setLineDash([]);
      ctx.beginPath();
      comp.activity.forEach((v, idx) => {
        const x = xOf(idx);
        const y = yOf(v);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // Draw encoder |qdot| profile (blue dashed)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.8;
    ctx.setLineDash([5, 3]);
    ctx.beginPath();
    syncDiagnostics.encoderSpeed.forEach((v, idx) => {
      const x = xOf(idx);
      const y = yOf(v);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }, [syncDiagnostics, algoVersion]);

  // 4. Real-Time Trans/Rot Error History dual-axis line chart (closed-form steps + live LM / EM-LM iterations)
  useEffect(() => {
    const canvas = historyCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.fillStyle = '#090C12';
    ctx.fillRect(0, 0, W, H);

    const padL = 40;
    const padR = 38;
    const padT = 18;
    const padB = 22;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    // Horizontal grid lines
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = padT + (i / 4) * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(W - padR, y);
      ctx.stroke();
    }

    if (effectiveHistory.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '11px "JetBrains Mono", monospace';
      ctx.fillText(
        isRunning
          ? '正在采集冷启动探索步，等待首次外参求解...'
          : '运行单次标定后实时追踪平移/旋转误差收敛轨迹',
        padL + 4,
        padT + plotH * 0.52
      );
      return;
    }

    const bounds = computeErrorChartBounds(effectiveHistory, useLogScale);
    const nPts = effectiveHistory.length;
    const xOf = (idx: number) =>
      nPts === 1 ? padL + plotW * 0.5 : padL + (idx / (nPts - 1)) * plotW;
    const yTrans = (mm: number) =>
      padT +
      (1 - normalizeErrorValue(mm, bounds.minTransMm, bounds.maxTransMm, bounds.logScale)) *
        plotH;
    const yRot = (deg: number) =>
      padT +
      (1 - normalizeErrorValue(deg, bounds.minRotDeg, bounds.maxRotDeg, bounds.logScale)) *
        plotH;

    // Left Y-axis labels (Translation mm, cyan)
    const fmtMm = (v: number) => (v >= 100 ? `${v.toFixed(0)}` : v >= 10 ? `${v.toFixed(0)}` : `${v.toFixed(1)}`);
    const midTrans = bounds.logScale
      ? Math.sqrt(bounds.minTransMm * bounds.maxTransMm)
      : (bounds.minTransMm + bounds.maxTransMm) * 0.5;
    ctx.fillStyle = '#22d3ee';
    ctx.font = '9.5px "JetBrains Mono", monospace';
    ctx.fillText(`${fmtMm(bounds.maxTransMm)}mm`, 2, padT + 4);
    ctx.fillText(`${fmtMm(midTrans)}mm`, 2, padT + plotH * 0.5 + 3);
    ctx.fillText(`${bounds.logScale ? '0.1' : '0'}mm`, 4, padT + plotH + 3);

    // Right Y-axis labels (Rotation deg, amber)
    const midRot = bounds.logScale
      ? Math.sqrt(bounds.minRotDeg * bounds.maxRotDeg)
      : (bounds.minRotDeg + bounds.maxRotDeg) * 0.5;
    ctx.fillStyle = '#fbbf24';
    ctx.fillText(`${bounds.maxRotDeg.toFixed(1)}°`, W - padR + 4, padT + 4);
    ctx.fillText(`${midRot.toFixed(1)}°`, W - padR + 4, padT + plotH * 0.5 + 3);
    ctx.fillText(`${bounds.logScale ? '0.01' : '0'}°`, W - padR + 4, padT + plotH + 3);

    // Phase transition vertical dashed divider (Closed-form exploration -> Manifold LM / Annealed EM-LM)
    if (historySummary.refinementStartIdx !== null && nPts >= 2) {
      const xSplit = xOf(historySummary.refinementStartIdx);
      ctx.save();
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.45)';
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(xSplit, padT);
      ctx.lineTo(xSplit, padT + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#94a3b8';
      ctx.font = '9px "JetBrains Mono", monospace';
      if (xSplit - padL > 28) {
        ctx.fillText('闭式解', padL + 3, padT - 5);
      }
      ctx.fillText(
        algoVersion === 'v3' ? '退火 EM-LM' : '流形 LM',
        Math.min(W - padR - 46, xSplit + 4),
        padT - 5
      );
      ctx.restore();
    }

    // X-axis step/iteration ticks
    ctx.fillStyle = '#64748b';
    ctx.font = '9.5px "JetBrains Mono", monospace';
    ctx.fillText(effectiveHistory[0].label, padL - 2, H - 6);
    if (nPts >= 3) {
      const midIdx = Math.floor((nPts - 1) / 2);
      ctx.fillText(effectiveHistory[midIdx].label, xOf(midIdx) - 10, H - 6);
    }
    if (nPts >= 2) {
      const lastLabel = effectiveHistory[nPts - 1].label;
      ctx.fillText(lastLabel, W - padR - Math.max(24, lastLabel.length * 6), H - 6);
    }

    // Subtle translucent area fill under Translation error curve
    if (nPts >= 2) {
      const grad = ctx.createLinearGradient(0, padT, 0, padT + plotH);
      grad.addColorStop(0, 'rgba(6, 182, 212, 0.22)');
      grad.addColorStop(1, 'rgba(6, 182, 212, 0.01)');
      ctx.fillStyle = grad;
      ctx.beginPath();
      effectiveHistory.forEach((p, idx) => {
        const x = xOf(idx);
        const y = yTrans(p.transErrMm);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.lineTo(xOf(nPts - 1), padT + plotH);
      ctx.lineTo(xOf(0), padT + plotH);
      ctx.closePath();
      ctx.fill();
    }

    // Translation error line (cyan solid)
    if (nPts >= 2) {
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 2;
      ctx.beginPath();
      effectiveHistory.forEach((p, idx) => {
        const x = xOf(idx);
        const y = yTrans(p.transErrMm);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Rotation error line (amber solid)
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      effectiveHistory.forEach((p, idx) => {
        const x = xOf(idx);
        const y = yRot(p.rotErrDeg);
        if (idx === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    // Closed-form / init step milestone markers
    for (let i = 0; i < nPts; i++) {
      const p = effectiveHistory[i];
      if (p.phase === 'init' || p.phase === 'closed_form' || nPts <= 10) {
        const x = xOf(i);
        ctx.fillStyle = '#090C12';
        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(x, yTrans(p.transErrMm), 2.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        ctx.strokeStyle = '#fbbf24';
        ctx.beginPath();
        ctx.arc(x, yRot(p.rotErrDeg), 2.1, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    // Latest point highlighted tip markers
    const last = effectiveHistory[nPts - 1];
    const xLast = xOf(nPts - 1);
    ctx.fillStyle = '#22d3ee';
    ctx.beginPath();
    ctx.arc(xLast, yTrans(last.transErrMm), 3.3, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#fbbf24';
    ctx.beginPath();
    ctx.arc(xLast, yRot(last.rotErrDeg), 3.0, 0, Math.PI * 2);
    ctx.fill();

    // Interactive hover crosshair & active nodes
    if (hoverIdx !== null && hoverIdx >= 0 && hoverIdx < nPts) {
      const hp = effectiveHistory[hoverIdx];
      const hx = xOf(hoverIdx);
      ctx.save();
      ctx.strokeStyle = 'rgba(226, 232, 240, 0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(hx, padT);
      ctx.lineTo(hx, padT + plotH);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = '#06b6d4';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.arc(hx, yTrans(hp.transErrMm), 3.8, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.strokeStyle = '#f59e0b';
      ctx.beginPath();
      ctx.arc(hx, yRot(hp.rotErrDeg), 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }
  }, [effectiveHistory, historySummary.refinementStartIdx, useLogScale, hoverIdx, algoVersion, isRunning]);

  const handleHistoryMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = historyCanvasRef.current;
    if (!canvas || effectiveHistory.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0) return;
    const padL = 40;
    const padR = 38;
    const plotW = canvas.width - padL - padR;
    const relX = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const norm = Math.max(0, Math.min(1, (relX - padL) / Math.max(1, plotW)));
    const idx =
      effectiveHistory.length === 1
        ? 0
        : Math.round(norm * (effectiveHistory.length - 1));
    setHoverIdx(idx);
  };

  const bestRobotComp = syncDiagnostics?.components.find((c) => c.accepted) ?? null;
  const topDistractorComp = syncDiagnostics?.components.find((c) => !c.accepted) ?? null;
  const activePoint =
    hoverIdx !== null && effectiveHistory[hoverIdx]
      ? effectiveHistory[hoverIdx]
      : historySummary.latest;

  return (
    <aside
      aria-label="传感器与算法诊断视图"
      className="flex flex-col gap-2.5 bg-[#10151F] border border-slate-800 rounded-md p-3 overflow-y-auto"
    >
      {/* 1. Virtual RGB */}
      <section>
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-[12px] font-semibold text-slate-200">虚拟 RealSense D435i RGB</h2>
          <span className="text-[11px] font-mono text-slate-400">
            {width}×{height} · FOV 58°
          </span>
        </div>
        <div className="bg-[#090C12] border border-slate-800 rounded overflow-hidden">
          <canvas
            ref={rgbCanvasRef}
            width={width}
            height={height}
            data-testid="canvas-rgb"
            className="w-full h-auto block aspect-[4/3] max-h-[186px] object-contain"
          />
        </div>
      </section>

      {/* 2. Turbo Depth + Active Sub-chain / V3 EM Posterior Overlay */}
      <section>
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-[12px] font-semibold text-slate-200">
            {algoVersion === 'v3' && v3PosteriorOverlay
              ? 'Turbo 深度 + V3 EM 后验分类图'
              : 'Turbo 深度 + 活跃子链掩码'}
          </h2>
          <span className="text-[11px] font-mono text-cyan-300">
            {algoVersion === 'v3' && v3PosteriorOverlay ? 'I / II / III / M_occ' : '青色 = 活跃前景'}
          </span>
        </div>
        <div className="bg-[#090C12] border border-slate-800 rounded overflow-hidden">
          <canvas
            ref={depthCanvasRef}
            width={width}
            height={height}
            data-testid="canvas-depth"
            className="w-full h-auto block aspect-[4/3] max-h-[186px] object-contain"
          />
        </div>
        {algoVersion === 'v3' && v3PosteriorOverlay && (
          <div className="flex flex-wrap items-center gap-x-2.5 gap-y-0.5 text-[10px] font-mono text-slate-400 mt-1">
            <span className="text-cyan-300">■ I 名义刚体</span>
            <span className="text-amber-300">■ II 非标刚体</span>
            <span className="text-pink-400">■ III 柔性/离群</span>
            <span className="text-rose-400">■ M_occ 遮挡带</span>
          </div>
        )}
      </section>

      {/* 3. Timestamp Sync Cross-Correlation Waveform */}
      <section>
        <div className="flex items-center justify-between mb-1">
          <h2 className="text-[12px] font-semibold text-slate-200">
            {algoVersion === 'v3' ? '多碎片因果同步聚合波形' : '时间戳同步相关性波形'}
          </h2>
          <span className="text-[10.5px] font-mono text-slate-400">
            阈值 ρ≥{algoVersion === 'v3' ? '0.40' : SYNC_CORR_THRESHOLD}
          </span>
        </div>
        <div className="bg-[#090C12] border border-slate-800 rounded overflow-hidden">
          <canvas
            ref={syncCanvasRef}
            width={320}
            height={114}
            data-testid="canvas-sync"
            className="w-full h-auto block"
          />
        </div>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[10.5px] font-mono text-slate-400 mt-1">
          <span className="text-sky-400">--- 编码器 |q̇|</span>
          <span className="text-emerald-400">
            — 机械臂 ρ={bestRobotComp ? bestRobotComp.rho.toFixed(2) : '—'}
          </span>
          <span className="text-rose-400">
            — 干扰物 ρ={topDistractorComp ? topDistractorComp.rho.toFixed(2) : '—'}
          </span>
        </div>
      </section>

      {/* 4. Real-Time Trans/Rot Error History Convergence Chart */}
      <section data-testid="error-history-panel">
        <div className="flex items-center justify-between gap-1.5 mb-1">
          <div className="flex items-center gap-1.5 min-w-0">
            <h2 className="text-[12px] font-semibold text-slate-200 whitespace-nowrap">
              Trans/Rot Error History
            </h2>
            {isRunning && effectiveHistory.length > 0 && (
              <span className="text-[9.5px] font-mono text-cyan-300 px-1 py-0 bg-cyan-500/15 border border-cyan-500/40 rounded-[2px] shrink-0">
                LIVE
              </span>
            )}
          </div>
          <div className="flex items-center gap-1.5 shrink-0">
            <span
              data-testid="error-history-readout"
              className="text-[10px] font-mono text-slate-300 whitespace-nowrap"
            >
              {activePoint
                ? `${activePoint.label} ${activePoint.transErrMm.toFixed(1)}mm/${activePoint.rotErrDeg.toFixed(2)}°`
                : '等待迭代'}
            </span>
            <button
              type="button"
              data-testid="error-history-scale-toggle"
              aria-label="切换误差折线图线性或对数纵轴"
              onClick={() => setUseLogScale((v) => !v)}
              className="px-1.5 py-0.5 text-[9.5px] font-mono rounded-[2px] bg-[#090C12] border border-slate-700 text-slate-300 hover:text-cyan-200 hover:border-cyan-500/60 transition-colors"
            >
              {useLogScale ? 'Log' : 'Lin'}
            </button>
          </div>
        </div>
        <div className="bg-[#090C12] border border-slate-800 rounded overflow-hidden">
          <canvas
            ref={historyCanvasRef}
            width={320}
            height={116}
            data-testid="canvas-error-history"
            data-points={effectiveHistory.length}
            data-scale={useLogScale ? 'log' : 'lin'}
            onMouseMove={handleHistoryMouseMove}
            onMouseLeave={() => setHoverIdx(null)}
            className="w-full h-auto block cursor-crosshair"
          />
        </div>
        <div className="flex items-center justify-between text-[10px] font-mono text-slate-400 mt-0.5">
          <span className="text-cyan-400">— 平移误差 (mm)</span>
          <span className="text-slate-500">
            {effectiveHistory.length > 0 ? `${effectiveHistory.length} pts` : '实时轨迹'}
          </span>
          <span className="text-amber-400">— 旋转误差 (deg)</span>
        </div>
      </section>
    </aside>
  );
};
