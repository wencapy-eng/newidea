import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computeFunctionProfile,
  exportSessionJson,
  exportSessionJsonl,
  filterLogEntries,
  parseImportedLogContent,
  summarizeForLog,
  TraceLogger,
} from './traceLogger';

test('summarizeForLog compacts typed arrays and redacts secret keys', () => {
  const depth = new Float32Array([0, 1.2, 1.5, 2.0]);
  const summary = summarizeForLog({
    depth,
    apiKey: 'ghp_secret_value',
    authorization: 'Bearer abc123',
    pose: { t: [0.12345678, -0.5, 1.25] },
  }) as Record<string, unknown>;

  assert.equal(summary.apiKey, '[REDACTED]');
  assert.equal(summary.authorization, '[REDACTED]');
  const depthSum = summary.depth as Record<string, unknown>;
  assert.equal(depthSum.__typedArray, 'Float32Array');
  assert.equal(depthSum.length, 4);
  assert.equal(depthSum.nonZeroSampled, 3);
  assert.equal(depthSum.max, 2);
});

test('TraceLogger captures hierarchical sync/async calls, inputs, outputs, and errors', async () => {
  const logger = new TraceLogger({ persist: false });
  const session = logger.startSession({
    title: 'Unit Test Session',
    kind: 'single',
    algoVersion: 'v3',
    robotId: 'panda',
  });

  const outer = await logger.traceAsync(
    'pipelineV3',
    'runSingleCalibrationV3',
    { seed: 42 },
    async () => {
      const inner = logger.traceCall('screw', 'solveClosedForm', { count: 3 }, () => ({
        transMm: 4.5,
      }));
      logger.log('metric', 'solver', 'final', 'converged', { output: { errMm: 1.2 } });
      return { ok: true, stage1: inner };
    }
  );

  assert.equal(outer.ok, true);
  assert.equal(session.entries.length, 3);
  assert.equal(session.entries[0].depth, 0);
  assert.equal(session.entries[1].depth, 1);
  assert.equal(session.entries[1].parentId, session.entries[0].id);
  assert.deepEqual(session.entries[1].input, { count: 3 });
  assert.deepEqual(session.entries[1].output, { transMm: 4.5 });

  assert.throws(() => {
    logger.traceCall('solver', 'failFn', { x: 1 }, () => {
      throw new Error('singular matrix');
    });
  }, /singular matrix/);
  assert.equal(session.entries[3].level, 'error');
  assert.equal(session.entries[3].error, 'singular matrix');

  const profile = computeFunctionProfile(session.entries);
  assert.equal(profile.length, 3);
  const failedRow = profile.find((r) => r.fn === 'failFn');
  assert.equal(failedRow?.errors, 1);

  const filtered = filterLogEntries(session.entries, { level: 'call', query: 'solveClosedForm' });
  assert.equal(filtered.length, 1);
});

test('exportSessionJson and exportSessionJsonl round-trip through parseImportedLogContent', () => {
  const logger = new TraceLogger({ persist: false });
  const session = logger.startSession({
    title: 'RoundTrip Run',
    kind: 'single',
    algoVersion: 'v3',
    robotId: 'ur5e',
    configSnapshot: { numSteps: 6 },
  });
  logger.traceCall('frontend', 'segment', { frames: 16 }, () => ({ foregroundPx: 1420 }));
  logger.endSession({ finalTransMm: 1.85, finalRotDeg: 0.08, success: true });

  const jsonText = exportSessionJson(session);
  const importedJson = parseImportedLogContent(jsonText, 'run.json');
  assert.equal(importedJson.kind, 'imported');
  assert.equal(importedJson.entries.length, 1);
  assert.equal(importedJson.entries[0].fn, 'segment');
  assert.equal(importedJson.summary?.finalTransMm, 1.85);

  const jsonlText = exportSessionJsonl(session);
  const importedJsonl = parseImportedLogContent(jsonlText, 'run.jsonl');
  assert.equal(importedJsonl.entries.length, 1);
  assert.equal(importedJsonl.robotId, 'ur5e');
  assert.deepEqual(importedJsonl.entries[0].output, { foregroundPx: 1420 });
});
