import type { ErrorHistoryPhase, ErrorHistoryPoint, LMTracePoint } from '../calib/types';

export interface ErrorChartBounds {
  minTransMm: number;
  maxTransMm: number;
  minRotDeg: number;
  maxRotDeg: number;
  logScale: boolean;
}

export interface ErrorHistorySummary {
  count: number;
  initial: ErrorHistoryPoint | null;
  latest: ErrorHistoryPoint | null;
  bestTransMm: number;
  bestRotDeg: number;
  /** First index where refinement ('lm' | 'em_lm') begins after earlier closed-form / init points */
  refinementStartIdx: number | null;
}

/**
 * Append a new real error history measurement, sanitizing non-finite values
 * and assigning a sequential monotonic step index.
 */
export function appendErrorHistoryPoint(
  history: readonly ErrorHistoryPoint[],
  entry: Omit<ErrorHistoryPoint, 'step'>
): ErrorHistoryPoint[] {
  const transErrMm =
    Number.isFinite(entry.transErrMm) && entry.transErrMm >= 0 ? entry.transErrMm : 0;
  const rotErrDeg =
    Number.isFinite(entry.rotErrDeg) && entry.rotErrDeg >= 0 ? entry.rotErrDeg : 0;
  return [
    ...history,
    {
      ...entry,
      step: history.length,
      transErrMm,
      rotErrDeg,
    },
  ];
}

/**
 * Convert an existing LMTracePoint[] into ErrorHistoryPoint[] fallback
 * (used e.g. when viewing completed ablation trial traces).
 */
export function errorHistoryFromLmTrace(
  lmTrace: readonly LMTracePoint[],
  phase: Extract<ErrorHistoryPhase, 'lm' | 'em_lm'>
): ErrorHistoryPoint[] {
  const prefix = phase === 'em_lm' ? 'EM' : 'LM';
  return lmTrace.map((pt, idx) => ({
    step: idx,
    label: `${prefix}${pt.iter}`,
    phase,
    transErrMm: Number.isFinite(pt.transErrMm) ? Math.max(0, pt.transErrMm) : 0,
    rotErrDeg: Number.isFinite(pt.rotErrDeg) ? Math.max(0, pt.rotErrDeg) : 0,
    stage: pt.stage,
  }));
}

/**
 * Compute clean dual-Y-axis chart bounds for Translation (mm) and Rotation (deg).
 * Supports both linear scale and log10 scale (clamped at 0.1 mm / 0.01 deg floor).
 */
export function computeErrorChartBounds(
  history: readonly ErrorHistoryPoint[],
  logScale: boolean
): ErrorChartBounds {
  if (history.length === 0) {
    return {
      minTransMm: logScale ? 0.1 : 0,
      maxTransMm: 50,
      minRotDeg: logScale ? 0.01 : 0,
      maxRotDeg: 5,
      logScale,
    };
  }

  let peakTrans = 0;
  let peakRot = 0;
  for (let i = 0; i < history.length; i++) {
    if (history[i].transErrMm > peakTrans) peakTrans = history[i].transErrMm;
    if (history[i].rotErrDeg > peakRot) peakRot = history[i].rotErrDeg;
  }

  if (logScale) {
    const minTransMm = 0.1;
    const minRotDeg = 0.01;
    const maxTransMm = Math.max(10, peakTrans * 1.25);
    const maxRotDeg = Math.max(1.0, peakRot * 1.25);
    return { minTransMm, maxTransMm, minRotDeg, maxRotDeg, logScale: true };
  }

  return {
    minTransMm: 0,
    maxTransMm: Math.max(10, peakTrans * 1.12),
    minRotDeg: 0,
    maxRotDeg: Math.max(1.0, peakRot * 1.12),
    logScale: false,
  };
}

/**
 * Map a scalar error value into normalized [0, 1] vertical chart ratio (0 = bottom, 1 = top).
 */
export function normalizeErrorValue(
  value: number,
  minVal: number,
  maxVal: number,
  logScale: boolean
): number {
  const safeVal = Number.isFinite(value) ? value : minVal;
  if (logScale) {
    const lo = Math.log10(Math.max(1e-4, minVal));
    const hi = Math.log10(Math.max(minVal * 1.1, maxVal));
    const clamped = Math.max(Math.pow(10, lo), Math.min(Math.pow(10, hi), safeVal));
    const ratio = (Math.log10(clamped) - lo) / Math.max(1e-6, hi - lo);
    return Math.max(0, Math.min(1, ratio));
  }
  const span = Math.max(1e-6, maxVal - minVal);
  return Math.max(0, Math.min(1, (safeVal - minVal) / span));
}

/**
 * Summarize an error history trajectory for live badges and phase-divider markers.
 */
export function summarizeErrorHistory(
  history: readonly ErrorHistoryPoint[]
): ErrorHistorySummary {
  if (history.length === 0) {
    return {
      count: 0,
      initial: null,
      latest: null,
      bestTransMm: 0,
      bestRotDeg: 0,
      refinementStartIdx: null,
    };
  }

  let bestTransMm = Infinity;
  let bestRotDeg = Infinity;
  let refinementStartIdx: number | null = null;

  for (let i = 0; i < history.length; i++) {
    const pt = history[i];
    if (pt.transErrMm < bestTransMm) bestTransMm = pt.transErrMm;
    if (pt.rotErrDeg < bestRotDeg) bestRotDeg = pt.rotErrDeg;
    if (
      refinementStartIdx === null &&
      i > 0 &&
      (pt.phase === 'lm' || pt.phase === 'em_lm') &&
      (history[i - 1].phase === 'init' || history[i - 1].phase === 'closed_form')
    ) {
      refinementStartIdx = i;
    }
  }

  return {
    count: history.length,
    initial: history[0],
    latest: history[history.length - 1],
    bestTransMm,
    bestRotDeg,
    refinementStartIdx,
  };
}
