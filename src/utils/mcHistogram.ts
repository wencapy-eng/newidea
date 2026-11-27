export type HistogramMetric = 'trans' | 'rot';

export interface HistogramBin {
  index: number;
  min: number;
  max: number;
  count: number;
  /** Fraction of total samples in [0, 1] */
  fraction: number;
  /** 'pass' = strictly below threshold, 'border' = straddles threshold, 'fail' = at/above threshold */
  status: 'pass' | 'border' | 'fail';
}

export interface MonteCarloHistogramData {
  metric: HistogramMetric;
  unit: 'mm' | '°';
  threshold: number;
  samples: number[];
  bins: HistogramBin[];
  domainMin: number;
  domainMax: number;
  maxCount: number;
  mean: number;
  std: number;
  median: number;
  p95: number;
  min: number;
  max: number;
  /** Percentage (0..100) of samples strictly below convergence threshold */
  successRate: number;
}

export const MC_PASS_THRESHOLD_MM = 20.0;
export const MC_PASS_THRESHOLD_DEG = 2.0;

function computePercentile(sorted: readonly number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const clamped = Math.max(0, Math.min(1, p));
  const idx = clamped * (sorted.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const frac = idx - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

/**
 * Pick a clean human-readable upper bound for histogram x-axis ticks.
 */
function niceCeil(val: number, minSpan: number): number {
  const target = Math.max(val, minSpan);
  const exp = Math.floor(Math.log10(target));
  const base = Math.pow(10, exp);
  const norm = target / base;
  let nice = 10;
  if (norm <= 1.0) nice = 1.0;
  else if (norm <= 1.2) nice = 1.2;
  else if (norm <= 1.5) nice = 1.5;
  else if (norm <= 2.0) nice = 2.0;
  else if (norm <= 2.5) nice = 2.5;
  else if (norm <= 3.0) nice = 3.0;
  else if (norm <= 4.0) nice = 4.0;
  else if (norm <= 5.0) nice = 5.0;
  else if (norm <= 6.0) nice = 6.0;
  else if (norm <= 8.0) nice = 8.0;
  return nice * base;
}

/**
 * Build a quantitative Monte Carlo final pose error histogram with exact descriptive
 * statistics (mean, std, median, P95, and pass rate) from real calibration trial samples.
 */
export function buildMonteCarloHistogram(
  rawSamples: readonly number[],
  metric: HistogramMetric,
  desiredBins?: number
): MonteCarloHistogramData {
  const unit: 'mm' | '°' = metric === 'trans' ? 'mm' : '°';
  const threshold = metric === 'trans' ? MC_PASS_THRESHOLD_MM : MC_PASS_THRESHOLD_DEG;

  const samples = rawSamples.filter((v) => Number.isFinite(v) && v >= 0);
  if (samples.length === 0) {
    return {
      metric,
      unit,
      threshold,
      samples: [],
      bins: [],
      domainMin: 0,
      domainMax: metric === 'trans' ? 20 : 2,
      maxCount: 0,
      mean: 0,
      std: 0,
      median: 0,
      p95: 0,
      min: 0,
      max: 0,
      successRate: 0,
    };
  }

  const sorted = [...samples].sort((a, b) => a - b);
  const n = sorted.length;
  const mean = sorted.reduce((s, v) => s + v, 0) / n;
  const variance = sorted.reduce((s, v) => s + (v - mean) * (v - mean), 0) / n;
  const std = Math.sqrt(variance);
  const median = computePercentile(sorted, 0.5);
  const p95 = computePercentile(sorted, 0.95);
  const minVal = sorted[0];
  const maxVal = sorted[n - 1];
  const passCount = sorted.filter((v) => v < threshold).length;
  const successRate = (passCount / n) * 100;

  const binCount = Math.max(
    4,
    Math.min(10, desiredBins ?? Math.max(5, Math.min(8, Math.ceil(Math.sqrt(n)) + 3)))
  );

  const domainMin = 0;
  const minSpan = metric === 'trans' ? 4.0 : 0.4;
  const domainMax = niceCeil(maxVal * 1.15, minSpan);
  const binWidth = (domainMax - domainMin) / binCount;

  const counts = new Array<number>(binCount).fill(0);
  for (let i = 0; i < n; i++) {
    const rawIdx = Math.floor((sorted[i] - domainMin) / binWidth);
    const bIdx = Math.max(0, Math.min(binCount - 1, rawIdx));
    counts[bIdx]++;
  }

  const bins: HistogramBin[] = counts.map((count, index) => {
    const bMin = domainMin + index * binWidth;
    const bMax = domainMin + (index + 1) * binWidth;
    let status: 'pass' | 'border' | 'fail' = 'pass';
    if (bMin >= threshold) status = 'fail';
    else if (bMax > threshold) status = 'border';
    return {
      index,
      min: bMin,
      max: bMax,
      count,
      fraction: count / n,
      status,
    };
  });

  let maxCount = 0;
  for (let i = 0; i < bins.length; i++) {
    if (bins[i].count > maxCount) maxCount = bins[i].count;
  }

  return {
    metric,
    unit,
    threshold,
    samples: sorted,
    bins,
    domainMin,
    domainMax,
    maxCount,
    mean,
    std,
    median,
    p95,
    min: minVal,
    max: maxVal,
    successRate,
  };
}
