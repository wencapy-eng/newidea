import test from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_DELTA_PARAMS, solveDeltaFK } from './deltaKinematics';
import { solveParallelHandEyeClosedForm } from '../calib/screwStage1';
import { computePoseError, lookAtCV, mat3MulVec, normalize3, transformPoint, type Vec3 } from '../math/se3';
import type { ScrewObservation } from '../calib/types';

test('Delta parallel FK preserves exact closed-loop parallelogram rod length across workspace', () => {
  const p = DEFAULT_DELTA_PARAMS;
  const configs: Array<[number, number, number]> = [
    [0.22, 0.22, 0.22],
    [-0.10, 0.55, 0.15],
    [0.48, -0.05, 0.42],
    [0.10, 0.45, -0.08],
    [0.58, 0.58, 0.58],
  ];

  for (const [q0, q1, q2] of configs) {
    const fk = solveDeltaFK(q0, q1, q2, p);
    assert.ok(fk !== null, `FK should succeed at (${q0}, ${q1}, ${q2})`);
    assert.equal(fk.rods.length, 6);

    // Every single one of the 6 twin parallelogram rods must match forearmLen to < 1e-9 m
    for (let r = 0; r < fk.rods.length; r++) {
      const rod = fk.rods[r];
      assert.ok(
        Math.abs(rod.length - p.forearmLen) < 1e-9,
        `Rod #${r} length mismatch: got ${rod.length}, expected ${p.forearmLen}`
      );
      const qNorm = Math.hypot(rod.quat[0], rod.quat[1], rod.quat[2], rod.quat[3]);
      assert.ok(Math.abs(qNorm - 1.0) < 1e-9, `Rod #${r} quaternion not unit: ${qNorm}`);
    }

    // Platform must lie strictly below overhead base plate
    assert.ok(fk.platformPos[2] < p.baseZ - 0.15 && fk.platformPos[2] > 0.12);
  }
});

test('Parallel Stage 1 closed-form solver recovers exact R_cb and accurate t_cb from 3 Delta sweeps', () => {
  const T_gt = lookAtCV([1.28, 0.36, 0.86], [0.02, 0.0, 0.44], 0.06);
  const p = DEFAULT_DELTA_PARAMS;

  const sweeps: Array<{ qStart: [number, number, number]; qEnd: [number, number, number] }> = [
    { qStart: [-0.08, 0.20, 0.20], qEnd: [0.56, 0.20, 0.20] },
    { qStart: [0.20, -0.08, 0.20], qEnd: [0.20, 0.56, 0.20] },
    { qStart: [0.20, 0.20, -0.08], qEnd: [0.20, 0.20, 0.56] },
  ];

  const obs: ScrewObservation[] = sweeps.map((sw, idx) => {
    const pS = solveDeltaFK(sw.qStart[0], sw.qStart[1], sw.qStart[2], p)!.platformPos;
    const pE = solveDeltaFK(sw.qEnd[0], sw.qEnd[1], sw.qEnd[2], p)!.platformPos;
    const pM: Vec3 = [0.5 * (pS[0] + pE[0]), 0.5 * (pS[1] + pE[1]), 0.5 * (pS[2] + pE[2])];
    const deltaPBase: Vec3 = [pE[0] - pS[0], pE[1] - pS[1], pE[2] - pS[2]];
    const deltaPCam = mat3MulVec(T_gt.R, deltaPBase);
    const centroidBase: Vec3 = [pM[0], pM[1], pM[2] - 0.022];
    const cCamExact = transformPoint(T_gt, centroidBase);
    // Simulate front-surface shift of -0.038m along viewing ray (matches solver compensation)
    const ray = normalize3(cCamExact);
    const centroidCam: Vec3 = [
      cCamExact[0] - 0.038 * ray[0],
      cCamExact[1] - 0.038 * ray[1],
      cCamExact[2] - 0.038 * ray[2],
    ];
    return {
      action: {
        id: `test_m${idx}`,
        label: `M${idx}`,
        jointIndex: idx,
        qBase: [0.2, 0.2, 0.2, 0],
        qStart: -0.08,
        qEnd: 0.56,
      },
      kind: 'translation',
      xiBase: [0, 0, 0, 0, 0, 0],
      xiCam: [0, 0, 0, 0, 0, 0],
      deltaQ: 0.64,
      thetaObs: Math.hypot(deltaPCam[0], deltaPCam[1], deltaPCam[2]),
      pitch: 0,
      icpRms: 0.002,
      weight: 1.0,
      dTCam: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: deltaPCam },
      deltaPBase,
      deltaPCam,
      centroidBase,
      centroidCam,
    };
  });

  const T_est = solveParallelHandEyeClosedForm(obs);
  const err = computePoseError(T_est, T_gt);
  assert.ok(err.rotDeg < 1e-8, `Parallel Stage 1 rotation error too large: ${err.rotDeg}`);
  assert.ok(err.transMm < 1e-6, `Parallel Stage 1 translation error too large: ${err.transMm}`);
});
