import test from 'node:test';
import assert from 'node:assert/strict';
import {
  appendErrorHistoryPoint,
  computeErrorChartBounds,
  errorHistoryFromLmTrace,
  normalizeErrorValue,
  summarizeErrorHistory,
} from './errorHistory';
import type { ErrorHistoryPoint } from '../calib/types';

test('appendErrorHistoryPoint assigns monotonic step indices and clamps invalid inputs', () => {
  let hist: ErrorHistoryPoint[] = [];
  hist = appendErrorHistoryPoint(hist, {
    label: 'S3',
    phase: 'closed_form',
    transErrMm: 42.5,
    rotErrDeg: 3.1,
  });
  hist = appendErrorHistoryPoint(hist, {
    label: 'EM0',
    phase: 'em_lm',
    transErrMm: Number.NaN,
    rotErrDeg: -1,
  });

  assert.equal(hist.length, 2);
  assert.equal(hist[0].step, 0);
  assert.equal(hist[1].step, 1);
  assert.equal(hist[1].transErrMm, 0);
  assert.equal(hist[1].rotErrDeg, 0);
});

test('computeErrorChartBounds and normalizeErrorValue support linear and log10 scales cleanly', () => {
  const hist: ErrorHistoryPoint[] = [
    { step: 0, label: 'S3', phase: 'closed_form', transErrMm: 100, rotErrDeg: 8.0 },
    { step: 1, label: 'EM1', phase: 'em_lm', transErrMm: 10, rotErrDeg: 0.8 },
    { step: 2, label: 'EM12', phase: 'em_lm', transErrMm: 1.0, rotErrDeg: 0.08 },
  ];

  const lin = computeErrorChartBounds(hist, false);
  assert.equal(lin.minTransMm, 0);
  assert.ok(lin.maxTransMm > 100);
  assert.ok(lin.maxRotDeg > 8.0);

  const rLinTop = normalizeErrorValue(100, lin.minTransMm, lin.maxTransMm, false);
  const rLinLow = normalizeErrorValue(1, lin.minTransMm, lin.maxTransMm, false);
  assert.ok(rLinTop > 0.85 && rLinTop <= 1.0);
  assert.ok(rLinLow > 0 && rLinLow < 0.02);

  const logB = computeErrorChartBounds(hist, true);
  assert.equal(logB.minTransMm, 0.1);
  assert.equal(logB.minRotDeg, 0.01);
  const rLog100 = normalizeErrorValue(100, logB.minTransMm, logB.maxTransMm, true);
  const rLog10 = normalizeErrorValue(10, logB.minTransMm, logB.maxTransMm, true);
  const rLog1 = normalizeErrorValue(1, logB.minTransMm, logB.maxTransMm, true);
  // Log scale keeps equal decade steps evenly spaced so sub-mm convergence stays legible
  const gapHigh = rLog100 - rLog10;
  const gapLow = rLog10 - rLog1;
  assert.ok(Math.abs(gapHigh - gapLow) < 1e-6);
});

test('summarizeErrorHistory detects refinement phase transition and tracks best errors', () => {
  const hist: ErrorHistoryPoint[] = [
    { step: 0, label: 'S3', phase: 'closed_form', transErrMm: 45, rotErrDeg: 3.4 },
    { step: 1, label: 'S4', phase: 'closed_form', transErrMm: 28, rotErrDeg: 1.9 },
    { step: 2, label: 'EM0', phase: 'em_lm', transErrMm: 28, rotErrDeg: 1.9 },
    { step: 3, label: 'EM5', phase: 'em_lm', transErrMm: 1.8, rotErrDeg: 0.11 },
  ];

  const summary = summarizeErrorHistory(hist);
  assert.equal(summary.count, 4);
  assert.equal(summary.refinementStartIdx, 2);
  assert.equal(summary.bestTransMm, 1.8);
  assert.equal(summary.bestRotDeg, 0.11);
  assert.equal(summary.latest?.label, 'EM5');

  const fromLm = errorHistoryFromLmTrace(
    [
      { iter: 0, stage: 0, cost: 1, rotErrDeg: 2.5, transErrMm: 20, geoCount: 100, velCount: 50, lambda: 1e-2 },
      { iter: 1, stage: 1, cost: 0.2, rotErrDeg: 0.1, transErrMm: 1.2, geoCount: 120, velCount: 60, lambda: 4e-3 },
    ],
    'lm'
  );
  assert.equal(fromLm.length, 2);
  assert.equal(fromLm[0].label, 'LM0');
  assert.equal(fromLm[1].label, 'LM1');
  assert.equal(fromLm[1].phase, 'lm');
});
