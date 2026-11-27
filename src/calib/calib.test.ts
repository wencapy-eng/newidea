import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyAdjoint,
  composeSE3,
  computePoseError,
  cross3,
  dot3,
  expSE3,
  lookAtCV,
  mat3Det,
  mat3Mul,
  mat3MulVec,
  mat3Transpose,
  normalize3,
  SeededRNG,
  SpatialHashGrid3D,
  transformPoint,
  type PointNormalSample,
  type Vec3,
  type Vec6,
} from '../math/se3';
import { computeMaxDepthEnvelope, computeMaxLaggedPearson, SYNC_CORR_THRESHOLD } from './frontend';
import {
  evaluateStaticGeoResidual,
  evaluateVelocityResidual,
  runStage2ManifoldLM,
  type DynamicVelObs,
  type StaticGeoFrameObs,
} from './manifoldStage2';
import { generateInitialHandEyePose } from './pipeline';
import { solveScrewHandEyeClosedForm } from './screwStage1';

test('Stage 1 Wahba SVD + Plucker WLS recovers exact T_cb from 3 orthogonal screw axes', () => {
  const T_gt = lookAtCV([1.25, 0.48, 0.86], [0.1, 0.0, 0.42], 0.09);

  const axesBase: Array<{ omega: Vec3; anchor: Vec3 }> = [
    { omega: [0, 0, 1], anchor: [0, 0, 0.333] },
    { omega: [0, 1, 0], anchor: [0, 0, 0.333] },
    { omega: [1, 0, 0], anchor: [0.0825, -0.316, 0.55] },
  ];

  const observations = axesBase.map(({ omega, anchor }) => {
    const w = normalize3(omega);
    const v = cross3(anchor, w); // -w x r == r x w
    const xiBase: Vec6 = [w[0], w[1], w[2], v[0], v[1], v[2]];
    const xiCam = applyAdjoint(T_gt, xiBase);
    return { xiBase, xiCam, weight: 1.0 };
  });

  const T_est = solveScrewHandEyeClosedForm(observations);
  const err = computePoseError(T_est, T_gt);
  assert.ok(err.rotDeg < 1e-8, `Stage 1 rotation error too large: ${err.rotDeg}`);
  assert.ok(err.transMm < 1e-7, `Stage 1 translation error too large: ${err.transMm}`);
});

test('Stage 2 analytical Jacobians J_geo and J_vel match central finite differences to <1e-6', () => {
  const T = lookAtCV([1.3, -0.4, 0.9], [0.05, 0.05, 0.45], -0.05);
  const eps = 1e-6;

  // 1. Static point-to-plane Jacobian check (at tangent-plane point where s^c == q^c)
  const sBase: Vec3 = [0.22, -0.14, 0.58];
  const nBase: Vec3 = normalize3([0.6, -0.3, 0.74]);
  const sCam = transformPoint(T, sBase);
  const nCam = mat3MulVec(T.R, nBase);
  // Shift qCam along nCam so (sCam - qCam) || nCam -> (sCam - qCam) x nCam == 0
  const qCam: Vec3 = [sCam[0] - 0.015 * nCam[0], sCam[1] - 0.015 * nCam[1], sCam[2] - 0.015 * nCam[2]];

  const { J: Jgeo } = evaluateStaticGeoResidual(T, sBase, qCam, nCam);
  for (let k = 0; k < 6; k++) {
    const dPlus: Vec6 = [0, 0, 0, 0, 0, 0];
    const dMinus: Vec6 = [0, 0, 0, 0, 0, 0];
    dPlus[k] = eps;
    dMinus[k] = -eps;
    const ePlus = evaluateStaticGeoResidual(composeSE3(expSE3(dPlus), T), sBase, qCam, nCam).e;
    const eMinus = evaluateStaticGeoResidual(composeSE3(expSE3(dMinus), T), sBase, qCam, nCam).e;
    const numJ = (ePlus - eMinus) / (2 * eps);
    assert.ok(
      Math.abs(Jgeo[k] - numJ) < 1e-6,
      `J_geo[${k}] mismatch: analytic=${Jgeo[k]}, numeric=${numJ}`
    );
  }

  // 2. Dynamic normal velocity Jacobian check + positive sign physical check
  const omegaB: Vec3 = normalize3([0.2, 0.9, 0.35]);
  const anchorB: Vec3 = [0.08, -0.12, 0.45];
  const vB = cross3(anchorB, omegaB);
  const qdot = 1.15;
  const xiBaseDot: Vec6 = [
    omegaB[0] * qdot,
    omegaB[1] * qdot,
    omegaB[2] * qdot,
    vB[0] * qdot,
    vB[1] * qdot,
    vB[2] * qdot,
  ];
  // Compute exact ray-surface intersection depth rate dD/dt for ground truth T
  const xiCamTrue = applyAdjoint(T, xiBaseDot);
  const wC: Vec3 = [xiCamTrue[0], xiCamTrue[1], xiCamTrue[2]];
  const vC: Vec3 = [xiCamTrue[3], xiCamTrue[4], xiCamTrue[5]];
  const vTrue = cross3(wC, qCam);
  const vPredTrue: Vec3 = [vTrue[0] + vC[0], vTrue[1] + vC[1], vTrue[2] + vC[2]];
  // Make sure normal faces camera (n^T q < 0)
  const nCamFace: Vec3 = dot3(nCam, qCam) > 0 ? [-nCam[0], -nCam[1], -nCam[2]] : nCam;
  const zC = qCam[2];
  // Exact ray intersection derivative: dD/dt = (n^T v_true) * zC / (n^T qC)
  const dDdt = (dot3(nCamFace, vPredTrue) * zC) / dot3(nCamFace, qCam);
  // Positive-sign observation formula from prompt:
  const vNObs = +(dDdt) * (dot3(nCamFace, qCam) / zC);

  const velObs: DynamicVelObs = {
    qCam,
    nCam: nCamFace,
    zCam: zC,
    vNObs,
    xiBaseDot,
    w: 1.0,
  };

  // At true T, residual MUST be zero (proves positive sign is exact!)
  const { e: eAtGT } = evaluateVelocityResidual(T, velObs);
  assert.ok(Math.abs(eAtGT) < 1e-12, `Velocity residual at GT must be 0, got ${eAtGT}`);

  // Perturb T away from GT and check J_vel vs finite difference
  const T_pert = composeSE3(expSE3([0.04, -0.03, 0.05, 0.02, -0.03, 0.015]), T);
  const { J: Jvel } = evaluateVelocityResidual(T_pert, velObs);
  for (let k = 0; k < 6; k++) {
    const dPlus: Vec6 = [0, 0, 0, 0, 0, 0];
    const dMinus: Vec6 = [0, 0, 0, 0, 0, 0];
    dPlus[k] = eps;
    dMinus[k] = -eps;
    const ePlus = evaluateVelocityResidual(composeSE3(expSE3(dPlus), T_pert), velObs).e;
    const eMinus = evaluateVelocityResidual(composeSE3(expSE3(dMinus), T_pert), velObs).e;
    const numJ = (ePlus - eMinus) / (2 * eps);
    assert.ok(
      Math.abs(Jvel[k] - numJ) < 1e-6,
      `J_vel[${k}] mismatch: analytic=${Jvel[k]}, numeric=${numJ}`
    );
  }
});

test('Stage 2 Manifold LM converges to sub-millimeter accuracy from perturbed Stage 1 pose', () => {
  const T_gt = lookAtCV([1.28, 0.35, 0.88], [0.1, 0.0, 0.45], 0.04);
  // Synthetic 3D CAD surface points with diverse normals
  const cadSamples: PointNormalSample[] = [];
  const faces: Array<{ n: Vec3; c: Vec3; u: Vec3; v: Vec3 }> = [
    { n: [1, 0, 0], c: [0.2, 0.0, 0.45], u: [0, 1, 0], v: [0, 0, 1] },
    { n: [0, 1, 0], c: [0.05, 0.18, 0.55], u: [1, 0, 0], v: [0, 0, 1] },
    { n: [0, 0, 1], c: [0.0, -0.1, 0.72], u: [1, 0, 0], v: [0, 1, 0] },
    { n: normalize3([1, 1, 1]), c: [0.15, 0.12, 0.35], u: normalize3([1, -1, 0]), v: normalize3([1, 1, -2]) },
  ];
  for (const f of faces) {
    for (let iu = -3; iu <= 3; iu++) {
      for (let iv = -3; iv <= 3; iv++) {
        const s = 0.025;
        const p: Vec3 = [
          f.c[0] + iu * s * f.u[0] + iv * s * f.v[0],
          f.c[1] + iu * s * f.u[1] + iv * s * f.v[1],
          f.c[2] + iu * s * f.u[2] + iv * s * f.v[2],
        ];
        cadSamples.push({ p, n: f.n });
      }
    }
  }

  const grid = new SpatialHashGrid3D<PointNormalSample>(0.04);
  grid.insertAll(cadSamples);

  const staticPts = cadSamples.map((s, idx) => {
    const qCam = transformPoint(T_gt, s.p);
    const nCam = mat3MulVec(T_gt.R, s.n);
    const nFace: Vec3 = dot3(nCam, qCam) > 0 ? [-nCam[0], -nCam[1], -nCam[2]] : nCam;
    return { p: qCam, n: nFace, w: 1.0, u: idx, v: 0, z: qCam[2] };
  });

  const staticObs: StaticGeoFrameObs[] = [{ qEncoder: [0, 0, 0, 0, 0, 0, 0], points: staticPts, modelGrid: grid }];
  // Perturb by ~4 deg and ~35 mm
  const T_init = composeSE3(expSE3([0.04, -0.05, 0.03, 0.02, -0.025, 0.015]), T_gt);

  const res = runStage2ManifoldLM(T_init, staticObs, [], {
    useVelocityField: false,
    T_cb_gt_for_log: T_gt,
  });
  const finalErr = computePoseError(res.T_est, T_gt);
  assert.ok(finalErr.transMm < 0.2, `Expected <0.2mm, got ${finalErr.transMm}mm`);
  assert.ok(finalErr.rotDeg < 0.02, `Expected <0.02deg, got ${finalErr.rotDeg}deg`);
});

test('Timestamp-sync lagged Pearson correlation accepts synchronous robot and rejects async distractor', () => {
  // 16-frame S-curve encoder speed profile (3 pre-pause + 10 sweep + 3 post-pause)
  const encSpeed = Array.from({ length: 16 }, (_, t) => {
    if (t < 3 || t > 12) return 0;
    const tau = (t - 2) / 11;
    return Math.sin(Math.PI * tau);
  });

  // Robot component delayed by 1 frame (simulating ~25-50ms async delay)
  const robotActivity = Array.from({ length: 16 }, (_, t) => {
    const src = Math.max(0, Math.min(15, t - 1));
    return encSpeed[src] + 0.03 * Math.sin(t * 1.7);
  });

  // Asynchronous sinusoidal distractor oscillating independently across pause & sweep frames
  const distractorActivity = Array.from({ length: 16 }, (_, t) =>
    Math.abs(Math.sin(2 * Math.PI * 0.92 * t * 0.065 + 1.1))
  );

  const robotCorr = computeMaxLaggedPearson(robotActivity, encSpeed, 2);
  const distCorr = computeMaxLaggedPearson(distractorActivity, encSpeed, 2);

  assert.ok(
    robotCorr.rho >= SYNC_CORR_THRESHOLD && robotCorr.rho > 0.9,
    `Robot sync rho should be high, got ${robotCorr.rho}`
  );
  assert.ok(
    distCorr.rho < SYNC_CORR_THRESHOLD,
    `Async distractor rho should be below threshold ${SYNC_CORR_THRESHOLD}, got ${distCorr.rho}`
  );
});

test('Stage 1 consensus triplet + Huber IRLS rejects corrupted outlier screw sweep', () => {
  const T_gt = lookAtCV([1.22, 0.42, 0.85], [0.08, 0.0, 0.40], 0.06);
  const axesBase: Array<{ omega: Vec3; anchor: Vec3 }> = [
    { omega: [0, 0, 1], anchor: [0, 0, 0.25] },
    { omega: [0, 1, 0], anchor: [0.05, 0, 0.35] },
    { omega: [1, 0, 0], anchor: [0.08, -0.22, 0.52] },
    { omega: normalize3([0.5, 0.86, 0.1]), anchor: [0.15, 0.12, 0.48] },
    { omega: normalize3([-0.4, 0.3, 0.86]), anchor: [-0.08, 0.14, 0.38] },
  ];

  const observations = axesBase.map(({ omega, anchor }) => {
    const w = normalize3(omega);
    const v = cross3(anchor, w);
    const xiBase: Vec6 = [w[0], w[1], w[2], v[0], v[1], v[2]];
    const xiCam = applyAdjoint(T_gt, xiBase);
    return { xiBase, xiCam, weight: 1.0 };
  });

  // Inject 1 severely corrupted metallic-glint outlier sweep (~45 deg off & wrong Plucker moment)
  observations.push({
    xiBase: [0, 1, 0, -0.4, 0, 0.1],
    xiCam: [0.7071, 0.7071, 0, 0.45, -0.35, 0.6],
    weight: 0.9,
  });

  const T_est = solveScrewHandEyeClosedForm(observations);
  const err = computePoseError(T_est, T_gt);
  assert.ok(err.rotDeg < 0.8, `Expected robust Stage 1 rot < 0.8 deg, got ${err.rotDeg}`);
  assert.ok(err.transMm < 12.0, `Expected robust Stage 1 trans < 12 mm, got ${err.transMm}`);
});

test('Robust K-th order computeMaxDepthEnvelope resists 2.5% positive stereo spikes across 96 frames', () => {
  const W = 20;
  const H = 10;
  const nPix = W * H;
  const rng = new SeededRNG(42);
  const trueBg = 2.0;

  const frames = Array.from({ length: 96 }, (_, t) => {
    const depth = new Float32Array(nPix);
    for (let i = 0; i < nPix; i++) {
      let z = trueBg + rng.gaussian(0, 0.004);
      // 2.5% stereo mismatch outlier spikes (+-6..20 cm)
      if (rng.next() < 0.025) {
        z += (rng.next() < 0.5 ? -1 : 1) * rng.uniform(0.06, 0.20);
      }
      depth[i] = z;
    }
    return {
      width: W,
      height: H,
      rgb: new Uint8ClampedArray(nPix * 4),
      depth,
      timestamp: t * 0.065,
      qTrue: [0],
      qEncoder: [0],
      qdotEncoder: [0],
      activeJoint: 0,
    };
  });

  const bgEnv = computeMaxDepthEnvelope(frames, W, H);
  let corruptedCount = 0;
  for (let i = 0; i < nPix; i++) {
    if (Math.abs(bgEnv[i] - trueBg) > 0.025) corruptedCount++;
  }
  assert.ok(
    corruptedCount <= 2,
    `Expected <= 2 corrupted pixels out of ${nPix}, got ${corruptedCount}`
  );
});

test('Initial hand-eye extrinsic modes: zero (I4), controlled_random (exact Δθ/Δt), and full_random (Haar SO3)', () => {
  const T_gt = lookAtCV([1.25, 0.42, 0.88], [0.08, 0.0, 0.42], 0.05);
  const rng = new SeededRNG(20261126);

  // 1. Zero initialization: R = I_3, t = [0, 0, 0]
  const T_zero = generateInitialHandEyePose('zero', T_gt, rng);
  assert.deepEqual(T_zero.R, [1, 0, 0, 0, 1, 0, 0, 0, 1]);
  assert.deepEqual(T_zero.t, [0, 0, 0]);
  const errZero = computePoseError(T_zero, T_gt);
  assert.ok(errZero.rotDeg > 60 && errZero.transMm > 800, 'Zero init should have large realistic error vs GT');

  // 2. Controlled random initialization: exact Δθ and Δt on SO(3) x R^3
  const T_ctrl = generateInitialHandEyePose('controlled_random', T_gt, rng, {
    rotErrDeg: 22.5,
    transErrMm: 165,
  });
  const errCtrl = computePoseError(T_ctrl, T_gt);
  assert.ok(Math.abs(errCtrl.rotDeg - 22.5) < 1e-7, `Expected 22.5 deg, got ${errCtrl.rotDeg}`);
  assert.ok(Math.abs(errCtrl.transMm - 165) < 1e-7, `Expected 165 mm, got ${errCtrl.transMm}`);
  assert.ok(Math.abs(mat3Det(T_ctrl.R) - 1.0) < 1e-12, 'Controlled random R must have det=+1');

  // 3. Full random initialization: valid orthonormal SO(3) (det=+1, R^T R = I) independent of GT
  const T_full = generateInitialHandEyePose('full_random', T_gt, rng);
  assert.ok(Math.abs(mat3Det(T_full.R) - 1.0) < 1e-10, 'Full random R must have det=+1');
  const RtR = mat3Mul(mat3Transpose(T_full.R), T_full.R);
  const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  for (let i = 0; i < 9; i++) {
    assert.ok(Math.abs(RtR[i] - I[i]) < 1e-10, `Full random R not orthogonal at [${i}]`);
  }
  assert.ok(T_full.t[2] >= 0.25 && T_full.t[2] <= 2.5, 'Full random z should be in [0.25, 2.5]');
});
