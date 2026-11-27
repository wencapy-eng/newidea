import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { AblationRow } from '../calib/types';
import {
  buildMonteCarloHistogram,
  type HistogramMetric,
} from '../utils/mcHistogram';

export interface SingleRunErrorSample {
  transErrMm: number;
  rotErrDeg: number;
}

interface MonteCarloHistogramProps {
  ablationRows: AblationRow[];
  singleRunSamples?: SingleRunErrorSample[];
  isRunningBenchmark?: boolean;
}

export const MonteCarloHistogram: React.FC<MonteCarloHistogramProps> = ({
  ablationRows,
  singleRunSamples = [],
  isRunningBenchmark = false,
}) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [metric, setMetric] = useState<HistogramMetric>('trans');
  const [selectedRowId, setSelectedRowId] = useState<string>('first');
  const [hoverBinIdx, setHoverBinIdx] = useState<number | null>(null);

  // Reset selected row if current selection disappeared (e.g. robot/version switch)
  useEffect(() => {
    if (
      selectedRowId !== 'first' &&
      selectedRowId !== 'all' &&
      !ablationRows.some((r) => r.id === selectedRowId)
    ) {
      setSelectedRowId('first');
    }
  }, [ablationRows, selectedRowId]);

  const { samples, sourceTag } = useMemo(() => {
    if (ablationRows.length > 0) {
      if (selectedRowId === 'all') {
        const merged: number[] = [];
        for (const r of ablationRows) {
          const arr = metric === 'trans' ? r.trialTransMms : r.trialRotDegs;
          for (let i = 0; i < arr.length; i++) merged.push(arr[i]);
        }
        return { samples: merged, sourceTag: `全部 ${ablationRows.length} 组` };
      }
      const targetRow =
        selectedRowId === 'first'
          ? ablationRows[0]
          : ablationRows.find((r) => r.id === selectedRowId) ?? ablationRows[0];
      const arr = metric === 'trans' ? targetRow.trialTransMms : targetRow.trialRotDegs;
      const shortName = targetRow.name.replace(/^\d+\.\s*/, '').split('(')[0].trim();
      return { samples: arr, sourceTag: shortName };
    }

    if (singleRunSamples.length > 0) {
      return {
        samples: singleRunSamples.map((s) =>
          metric === 'trans' ? s.transErrMm : s.rotErrDeg
        ),
        sourceTag: '单次累计',
      };
    }

    return { samples: [], sourceTag: '' };
  }, [ablationRows, selectedRowId, singleRunSamples, metric]);

  const hist = useMemo(
    () => buildMonteCarloHistogram(samples, metric),
    [samples, metric]
  );

  useEffect(() => {
    if (hoverBinIdx !== null && hoverBinIdx >= hist.bins.length) {
      setHoverBinIdx(null);
    }
  }, [hist.bins.length, hoverBinIdx]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    ctx.fillStyle = '#090C12';
    ctx.fillRect(0, 0, W, H);

    const padL = 26;
    const padR = 10;
    const padT = 24;
    const padB = 22;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    // Horizontal grid lines
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const y = padT + (i / 3) * plotH;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(W - padR, y);
      ctx.stroke();
    }

    if (hist.samples.length === 0 || hist.bins.length === 0) {
      ctx.fillStyle = '#64748b';
      ctx.font = '10px "JetBrains Mono", monospace';
      const msg1 = isRunningBenchmark
        ? '正在统计 Monte Carlo 试验...'
        : '运行 6 组消融或多次单次标定';
      const msg2 = isRunningBenchmark
        ? '完成后自动绘制误差直方图'
        : '以生成最终位姿误差分布';
      ctx.fillText(msg1, padL + 8, padT + plotH * 0.44);
      ctx.fillText(msg2, padL + 8, padT + plotH * 0.64);
      return;
    }

    const maxY = Math.max(1, hist.maxCount);
    const xOfVal = (v: number) =>
      padL +
      Math.max(0, Math.min(1, (v - hist.domainMin) / Math.max(1e-6, hist.domainMax - hist.domainMin))) *
        plotW;

    // Y-axis count ticks
    ctx.fillStyle = '#64748b';
    ctx.font = '9.5px "JetBrains Mono", monospace';
    ctx.fillText(`${maxY}`, 6, padT + 4);
    ctx.fillText('0', 10, padT + plotH + 3);

    // Draw histogram bars
    const binW = plotW / hist.bins.length;
    const gap = Math.max(2, Math.min(4, binW * 0.14));

    for (let i = 0; i < hist.bins.length; i++) {
      const bin = hist.bins[i];
      const bx = padL + i * binW + gap * 0.5;
      const bw = Math.max(3, binW - gap);
      const bh = bin.count > 0 ? Math.max(3, (bin.count / maxY) * plotH) : 0;
      const by = padT + plotH - bh;
      const isHover = hoverBinIdx === i;

      if (bin.count > 0) {
        let fill = metric === 'trans' ? 'rgba(6, 182, 212, 0.72)' : 'rgba(245, 158, 11, 0.72)';
        let topStroke = metric === 'trans' ? '#22d3ee' : '#fbbf24';
        if (bin.status === 'fail') {
          fill = 'rgba(244, 63, 94, 0.76)';
          topStroke = '#fb7185';
        } else if (bin.status === 'border') {
          fill = 'rgba(251, 146, 60, 0.76)';
          topStroke = '#fdba74';
        }
        if (isHover) {
          fill = bin.status === 'fail'
            ? 'rgba(251, 113, 133, 0.95)'
            : metric === 'trans'
              ? 'rgba(34, 211, 238, 0.95)'
              : 'rgba(251, 191, 36, 0.95)';
        }

        ctx.fillStyle = fill;
        ctx.fillRect(bx, by, bw, bh);

        ctx.fillStyle = topStroke;
        ctx.fillRect(bx, by, bw, 1.5);

        // Bar count label above bar
        ctx.fillStyle = isHover ? '#ffffff' : '#cbd5e1';
        ctx.font = '9px "JetBrains Mono", monospace';
        const txt = `${bin.count}`;
        const tx = bx + bw * 0.5 - txt.length * 2.6;
        ctx.fillText(txt, tx, Math.max(padT - 2, by - 3));
      } else if (isHover) {
        ctx.fillStyle = 'rgba(148, 163, 184, 0.14)';
        ctx.fillRect(bx, padT, bw, plotH);
      }
    }

    // Vertical dashed Median (P50) marker (emerald)
    const xMed = xOfVal(hist.median);
    ctx.save();
    ctx.strokeStyle = '#34d399';
    ctx.lineWidth = 1.4;
    ctx.setLineDash([3, 2]);
    ctx.beginPath();
    ctx.moveTo(xMed, padT);
    ctx.lineTo(xMed, padT + plotH);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = '#34d399';
    ctx.font = '8.5px "JetBrains Mono", monospace';
    ctx.fillText('P50', Math.min(W - padR - 18, Math.max(padL, xMed - 8)), 10);
    ctx.restore();

    // Vertical dashed Pass Threshold marker if inside plotted domain
    if (hist.threshold > hist.domainMin && hist.threshold <= hist.domainMax) {
      const xThr = xOfVal(hist.threshold);
      ctx.save();
      ctx.strokeStyle = 'rgba(244, 63, 94, 0.78)';
      ctx.lineWidth = 1.2;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(xThr, padT);
      ctx.lineTo(xThr, padT + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#fb7185';
      ctx.font = '8.5px "JetBrains Mono", monospace';
      const thrLbl = metric === 'trans' ? '20mm' : '2°';
      ctx.fillText(thrLbl, Math.min(W - padR - 22, Math.max(padL, xThr - 10)), 10);
      ctx.restore();
    }

    // X-axis domain ticks
    const fmtTick = (v: number) =>
      metric === 'trans'
        ? v >= 10
          ? `${v.toFixed(0)}`
          : `${v.toFixed(1)}`
        : `${v.toFixed(1)}`;
    const midDom = (hist.domainMin + hist.domainMax) * 0.5;
    ctx.fillStyle = '#64748b';
    ctx.font = '9.5px "JetBrains Mono", monospace';
    ctx.fillText('0', padL - 2, H - 6);
    ctx.fillText(fmtTick(midDom), padL + plotW * 0.5 - 8, H - 6);
    const maxLbl = `${fmtTick(hist.domainMax)}${hist.unit}`;
    ctx.fillText(maxLbl, W - padR - maxLbl.length * 5.6, H - 6);
  }, [hist, metric, hoverBinIdx, isRunningBenchmark]);

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || hist.bins.length === 0) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0) return;
    const padL = 26;
    const padR = 10;
    const plotW = canvas.width - padL - padR;
    const relX = ((e.clientX - rect.left) / rect.width) * canvas.width;
    const norm = (relX - padL) / Math.max(1, plotW);
    if (norm < 0 || norm > 1) {
      setHoverBinIdx(null);
      return;
    }
    const idx = Math.min(hist.bins.length - 1, Math.floor(norm * hist.bins.length));
    setHoverBinIdx(idx);
  };

  const hoveredBin =
    hoverBinIdx !== null && hist.bins[hoverBinIdx] ? hist.bins[hoverBinIdx] : null;

  return (
    <section data-testid="mc-histogram-panel" className="min-w-0">
      <div className="flex items-center justify-between gap-1 mb-1">
        <div className="flex items-center gap-1 min-w-0">
          <h2 className="text-[11.5px] font-semibold text-slate-200 whitespace-nowrap">
            MC Dist.
          </h2>
          {hist.samples.length > 0 && (
            <span
              data-testid="mc-hist-n-badge"
              className="text-[9.5px] font-mono text-emerald-300 px-1 py-0 bg-emerald-500/15 border border-emerald-500/35 rounded-[2px] shrink-0"
            >
              N={hist.samples.length}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1 shrink-0">
          {ablationRows.length > 1 && (
            <select
              aria-label="选择消融条件分布"
              data-testid="mc-hist-condition-select"
              value={selectedRowId}
              onChange={(e) => setSelectedRowId(e.target.value)}
              className="max-w-[68px] bg-[#090C12] border border-slate-700 rounded-[2px] px-1 py-0.5 text-[9.5px] font-mono text-slate-300 focus:outline-none focus:border-cyan-500/60"
            >
              {ablationRows.map((row, idx) => (
                <option key={row.id} value={idx === 0 ? 'first' : row.id}>
                  #{idx + 1} {row.name.replace(/^\d+\.\s*/, '').slice(0, 10)}
                </option>
              ))}
              <option value="all">全部组</option>
            </select>
          )}
          <button
            type="button"
            data-testid="mc-hist-metric-toggle"
            aria-label="切换直方图平移或旋转误差指标"
            onClick={() => setMetric((m) => (m === 'trans' ? 'rot' : 'trans'))}
            className="px-1.5 py-0.5 text-[9.5px] font-mono rounded-[2px] bg-[#090C12] border border-slate-700 text-slate-300 hover:text-cyan-200 hover:border-cyan-500/60 transition-colors"
          >
            {metric === 'trans' ? 'mm' : 'deg'}
          </button>
        </div>
      </div>

      <div className="bg-[#090C12] border border-slate-800 rounded overflow-hidden">
        <canvas
          ref={canvasRef}
          width={230}
          height={116}
          data-testid="canvas-mc-histogram"
          data-samples={hist.samples.length}
          data-metric={metric}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setHoverBinIdx(null)}
          className="w-full h-auto block cursor-crosshair"
        />
      </div>

      <div
        data-testid="mc-hist-summary"
        className="flex items-center justify-between gap-1 text-[9.5px] font-mono text-slate-400 mt-0.5 truncate"
      >
        {hoveredBin ? (
          <span className="text-slate-200 truncate">
            [{hoveredBin.min.toFixed(1)}–{hoveredBin.max.toFixed(1)}
            {hist.unit}]: {hoveredBin.count}/{hist.samples.length} (
            {(hoveredBin.fraction * 100).toFixed(0)}%)
          </span>
        ) : hist.samples.length > 0 ? (
          <>
            <span className="text-emerald-400 truncate">
              P50 {hist.median.toFixed(metric === 'trans' ? 1 : 2)}
              {hist.unit}
            </span>
            <span className="text-slate-400 truncate" title={sourceTag}>
              {sourceTag}
            </span>
            <span
              className={
                hist.successRate >= 80
                  ? 'text-cyan-300 shrink-0'
                  : hist.successRate >= 50
                    ? 'text-amber-300 shrink-0'
                    : 'text-rose-400 shrink-0'
              }
            >
              {hist.successRate.toFixed(0)}%合格
            </span>
          </>
        ) : (
          <>
            <span className="text-emerald-400">┆ P50 中位</span>
            <span className="text-rose-400">┆ 合格阈值</span>
          </>
        )}
      </div>
    </section>
  );
};
