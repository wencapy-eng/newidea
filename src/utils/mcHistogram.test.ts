import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildMonteCarloHistogram,
  MC_PASS_THRESHOLD_DEG,
  MC_PASS_THRESHOLD_MM,
} from './mcHistogram';

test('buildMonteCarloHistogram handles empty and non-finite inputs cleanly', () => {
  const empty = buildMonteCarloHistogram([Number.NaN, -5, Infinity], 'trans');
  assert.equal(empty.samples.length, 0);
  assert.equal(empty.bins.length, 0);
  assert.equal(empty.maxCount, 0);
  assert.equal(empty.threshold, MC_PASS_THRESHOLD_MM);
  assert.equal(empty.unit, 'mm');
});

test('buildMonteCarloHistogram preserves exact sample count, statistics, and pass rate for translation errors', () => {
  const samples = [0.8, 1.2, 1.5, 2.1, 2.4, 3.0, 24.5];
  const hist = buildMonteCarloHistogram(samples, 'trans', 6);

  assert.equal(hist.samples.length, 7);
  assert.equal(hist.bins.length, 6);
  const sumCounts = hist.bins.reduce((acc, b) => acc + b.count, 0);
  assert.equal(sumCounts, 7);

  // Median of 7 sorted values is index 3 => 2.1
  assert.ok(Math.abs(hist.median - 2.1) < 1e-9);
  // 6 of 7 are < 20 mm
  assert.ok(Math.abs(hist.successRate - (6 / 7) * 100) < 1e-6);
  assert.ok(hist.p95 > 15 && hist.p95 <= 24.5);
  // At least one bin must be marked fail or border since max=24.5 > 20mm
  assert.ok(hist.bins.some((b) => b.status === 'fail' || b.status === 'border'));
});

test('buildMonteCarloHistogram handles identical single/repeated rotation samples without division by zero', () => {
  const hist = buildMonteCarloHistogram([0.08, 0.08, 0.08], 'rot', 5);
  assert.equal(hist.unit, '°');
  assert.equal(hist.threshold, MC_PASS_THRESHOLD_DEG);
  assert.equal(hist.std, 0);
  assert.equal(hist.median, 0.08);
  assert.equal(hist.successRate, 100);
  assert.equal(hist.bins.reduce((s, b) => s + b.count, 0), 3);
  assert.ok(hist.domainMax >= 0.4);
  assert.ok(hist.bins.every((b) => b.status === 'pass'));
});
