import assert from 'node:assert/strict';
import test from 'node:test';
import {
  applyAdjoint,
  composeSE3,
  computePoseError,
  cross3,
  dot3,
  expSE3,
  lookAtCV,
  mat3MulVec,
  normalize3,
  SeededRNG,
  SpatialHashGrid3D,
  transformPoint,
  type PointNormalSample,
  type Vec3,
  type Vec6,
} from '../../math/se3';
import type { CameraIntrinsics, WeightedPoint } from '../types';
import {
  computeAnnealedEmPosterior,
  runStage4AnnealedEmLM,
  type V3DynamicFrameGrid,
  type V3JointDynamicObs,
  type V3StaticFrameObs,
} from './annealedEmLmStage4';
import {
  computeAxisIndependenceSigmaMin,
  runGncWelschPointToPlaneICP,
  V3_PROBE_MIN_SIGMA_SIN,
} from './blindProbeStage2';
import {
  buildDepthConditionedOcc2DBand,
  isRejectedByOcc2D,
  OnlineDepthEnvelope,
  StaticVoxelExclusion3D,
} from './dualDomainFrontend';

const TEST_K: CameraIntrinsics = {
  width: 80,
  height: 60,
  fx: 72,
  fy: 72,
  cx: 40,
  cy: 30,
};

test('V3 Stage 1: 2D M_occ_2D rejects front-pillar flying pixels while preserving foreground in front of pillar', () => {
  const W = TEST_K.width;
  const H = TEST_K.height;
  const env = new OnlineDepthEnvelope(W, H, 3);

  // Synthesize 12 frames with static wall at z=2.0m and static front pillar at x in [36..42], z=0.80m
  for (let t = 0; t < 12; t++) {
    const d = new Float32Array(W * H);
    for (let v = 0; v < H; v++) {
      for (let u = 0; u < W; u++) {
        const idx = v * W + u;
        if (u >= 36 && u <= 42 && v >= 10 && v <= 55) {
          d[idx] = 0.80;
        } else {
          d[idx] = 2.0;
        }
      }
    }
    env.addFrame(d);
  }

  const { dMax, dMin } = env.getEnvelopes();
  const occ2D = buildDepthConditionedOcc2DBand(dMax, dMin, W, H, { dilatePx: 4 });

  assert.ok(occ2D.bandPixelCount > 100, 'Protruding static pillar edge band should be detected');

  // Flying pixel 2px outside pillar edge at z = 1.35m (behind pillar z=0.80m) MUST be rejected
  assert.equal(
    isRejectedByOcc2D(44, 30, 1.35, W, H, occ2D),
    true,
    'Flying pixel smeared behind front pillar edge must be rejected by M_occ_2D'
  );

  // Robot link sweeping IN FRONT of pillar at z = 0.62m MUST be preserved!
  assert.equal(
    isRejectedByOcc2D(44, 30, 0.62, W, H, occ2D),
    false,
    'Link passing in front of static front pillar must NOT be rejected by M_occ_2D'
  );

  // 3D V_static voxel exclusion rejects static base seam (z=1.995m near z=2.0m wall)
  // while preserving moving link well in front of rear wall (z=1.15m)
  const vStatic = new StaticVoxelExclusion3D(0.018);
  vStatic.buildFromStaticEnvelope(dMax, TEST_K, 1);

  const pOnWall: Vec3 = [((25 + 0.5 - TEST_K.cx) / TEST_K.fx) * 1.995, 0, 1.995];
  const pInFrontOfWall: Vec3 = [((25 + 0.5 - TEST_K.cx) / TEST_K.fx) * 1.15, 0, 1.15];
  assert.equal(vStatic.isNearStatic(pOnWall[0], pOnWall[1], pOnWall[2]), true);
  assert.equal(vStatic.isNearStatic(pInFrontOfWall[0], pInFrontOfWall[1], pInFrontOfWall[2]), false);
});

test('V3 Stage 2: GNC-Welsch ICP resists 25% Class III non-rigid flexible cable outliers', () => {
  const rng = new SeededRNG(20260501);
  const dT_gt = expSE3([0.18, -0.22, 0.14, 0.04, -0.03, 0.025]);

  const src: WeightedPoint[] = [];
  const dst: WeightedPoint[] = [];

  // Build non-symmetric rigid 3D surface patches (box + tilted faces so all 6 DoF are observable)
  const faceDefs: Array<{ origin: Vec3; uDir: Vec3; vDir: Vec3; n: Vec3 }> = [
    { origin: [0.0, 0.0, 1.2], uDir: [0.30, 0, 0], vDir: [0, 0.26, 0], n: [0, 0, -1] },
    { origin: [0.30, 0.0, 1.2], uDir: [0, 0, 0.24], vDir: [0, 0.26, 0], n: [1, 0, 0] },
    { origin: [0.0, 0.26, 1.2], uDir: [0.30, 0, 0], vDir: [0, 0, 0.24], n: [0, 1, 0] },
  ];

  for (const face of faceDefs) {
    for (let iu = 0; iu < 8; iu++) {
      for (let iv = 0; iv < 8; iv++) {
        const su = (iu + 0.5) / 8;
        const sv = (iv + 0.5) / 8;
        const p: Vec3 = [
          face.origin[0] + su * face.uDir[0] + sv * face.vDir[0],
          face.origin[1] + su * face.uDir[1] + sv * face.vDir[1],
          face.origin[2] + su * face.uDir[2] + sv * face.vDir[2],
        ];
        const pMoved = transformPoint(dT_gt, p);
        const nMoved = mat3MulVec(dT_gt.R, face.n);
        src.push({ p, n: face.n, w: 1000, u: 0, v: 0, z: p[2] });
        dst.push({ p: pMoved, n: nMoved, w: 1000, u: 0, v: 0, z: pMoved[2] });
      }
    }
  }

  // Inject 25% severe Class III non-rigid flexible cable outliers (45..95mm non-rigid whipping!)
  const nOutliers = Math.floor(src.length * 0.25);
  for (let i = 0; i < nOutliers; i++) {
    const pCable: Vec3 = [rng.uniform(-0.1, 0.35), rng.uniform(-0.1, 0.3), rng.uniform(1.1, 1.4)];
    const nCable = normalize3([rng.gaussian(), rng.gaussian(), -1]);
    src.push({ p: pCable, n: nCable, w: 1000, u: 0, v: 0, z: pCable[2] });
    // Whipping cable violates rigid dT_gt by 50..90mm
    const pRigid = transformPoint(dT_gt, pCable);
    const pWhipped: Vec3 = [
      pRigid[0] + rng.uniform(0.05, 0.09),
      pRigid[1] - rng.uniform(0.04, 0.08),
      pRigid[2] + rng.uniform(-0.06, 0.06),
    ];
    dst.push({ p: pWhipped, n: nCable, w: 1000, u: 0, v: 0, z: pWhipped[2] });
  }

  const res = runGncWelschPointToPlaneICP(src, dst);
  const err = computePoseError(res.T, dT_gt);
  assert.ok(
    err.rotDeg < 0.45 && err.transMm < 3.0,
    `GNC-Welsch ICP failed under 25% Class III cable outliers: rot=${err.rotDeg.toFixed(3)} deg, trans=${err.transMm.toFixed(2)} mm`
  );
});

test('V3 Stage 2: Blind-probe linear independence gate sigma_min distinguishes orthogonal vs coplanar axes', () => {
  const sOrtho = computeAxisIndependenceSigmaMin([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]);
  assert.ok(Math.abs(sOrtho - 1.0) < 1e-6, `Expected sigma_min=1 for orthonormal axes, got ${sOrtho}`);

  const sCoplanar = computeAxisIndependenceSigmaMin([
    [1, 0, 0],
    [0, 1, 0],
    [0.7071, 0.7071, 0],
  ]);
  assert.ok(sCoplanar < 1e-3, `Expected sigma_min~0 for coplanar axes, got ${sCoplanar}`);

  // 10-degree nearly collinear axes should fail sin(25 deg) threshold
  const rad10 = (10 * Math.PI) / 180;
  const sCollinear = computeAxisIndependenceSigmaMin([
    [1, 0, 0],
    [Math.cos(rad10), Math.sin(rad10), 0],
  ]);
  assert.ok(sCollinear < V3_PROBE_MIN_SIGMA_SIN);
});

test('V3 Stage 4: 3-component Annealed EM-LM separates Class I/II/III and recovers true 25ms time offset delta_t', () => {
  const rng = new SeededRNG(20260509);
  const T_gt = lookAtCV([1.25, -0.35, 0.85], [0.1, 0.0, 0.4], 0.05);
  const dtTrueSec = 0.025; // +25 ms encoder lag

  // Start from perturbed initial pose (~6 deg, ~45 mm) with delta_t = 0
  const T_init = composeSE3(expSE3([0.06, -0.07, 0.05, 0.028, -0.024, 0.022]), T_gt);

  // Build nominal CAD surface in base frame {B}
  const cadSamples: PointNormalSample[] = [];
  const faces: Array<{ origin: Vec3; uDir: Vec3; vDir: Vec3; n: Vec3 }> = [
    { origin: [0.0, -0.14, 0.25], uDir: [0.28, 0, 0], vDir: [0, 0.28, 0], n: [0, 0, 1] },
    { origin: [0.28, -0.14, 0.25], uDir: [0, 0, 0.30], vDir: [0, 0.28, 0], n: [1, 0, 0] },
    { origin: [0.0, -0.14, 0.25], uDir: [0.28, 0, 0], vDir: [0, 0, 0.30], n: [0, -1, 0] },
  ];
  for (const f of faces) {
    for (let iu = 0; iu < 8; iu++) {
      for (let iv = 0; iv < 8; iv++) {
        const su = (iu + 0.5) / 8;
        const sv = (iv + 0.5) / 8;
        cadSamples.push({
          p: [
            f.origin[0] + su * f.uDir[0] + sv * f.vDir[0],
            f.origin[1] + su * f.uDir[1] + sv * f.vDir[1],
            f.origin[2] + su * f.uDir[2] + sv * f.vDir[2],
          ],
          n: f.n,
        });
      }
    }
  }

  const grid = new SpatialHashGrid3D<PointNormalSample>(0.045);
  grid.insertAll(cadSamples);

  // 1. Static pause frame observations (Class I + Class II unmodeled rigid tool far from CAD)
  const staticPts: WeightedPoint[] = [];
  for (let i = 0; i < cadSamples.length; i += 2) {
    const s = cadSamples[i];
    const qC = transformPoint(T_gt, s.p);
    const nC = mat3MulVec(T_gt.R, s.n);
    staticPts.push({ p: qC, n: nC, w: 1.0, u: 0, v: 0, z: qC[2] });
  }
  const staticFrames: V3StaticFrameObs[] = [{ qEncoder: [0], points: staticPts, modelGrid: grid }];

  // 2. Dynamic sweep observations under 2 orthogonal twists (Class I, Class II, Class III)
  const twistsBase: Vec6[] = [
    [0, 0, 1.4, 0, 0, 0],
    [0, 1.3, 0, -0.35, 0, 0],
  ];
  const dynGrids: V3DynamicFrameGrid[] = twistsBase.map((_, idx) => ({
    frameIdx: idx,
    actionIdx: idx,
    modelGrid: grid,
  }));

  const dynObs: V3JointDynamicObs[] = [];
  const trueClassLabels: number[] = [];

  for (let twIdx = 0; twIdx < twistsBase.length; twIdx++) {
    const xiBaseDot = twistsBase[twIdx];
    const xiCamDot = applyAdjoint(T_gt, xiBaseDot);
    const wC: Vec3 = [xiCamDot[0], xiCamDot[1], xiCamDot[2]];
    const vC: Vec3 = [xiCamDot[3], xiCamDot[4], xiCamDot[5]];

    // Class I: 90 nominal CAD points shifted forward by true dtTrueSec = +25ms
    for (let i = 0; i < cadSamples.length; i += 2) {
      const s = cadSamples[i];
      const sCam0 = transformPoint(T_gt, s.p);
      const nC = mat3MulVec(T_gt.R, s.n);
      const wCrossS = cross3(wC, sCam0);
      const vPred: Vec3 = [wCrossS[0] + vC[0], wCrossS[1] + vC[1], wCrossS[2] + vC[2]];
      // Observed point at camera shutter time t = t_enc + dtTrueSec
      const qCam: Vec3 = [
        sCam0[0] + vPred[0] * dtTrueSec,
        sCam0[1] + vPred[1] * dtTrueSec,
        sCam0[2] + vPred[2] * dtTrueSec,
      ];
      const wCrossQ = cross3(wC, qCam);
      const vTrueQ: Vec3 = [wCrossQ[0] + vC[0], wCrossQ[1] + vC[1], wCrossQ[2] + vC[2]];
      const vNObs = dot3(nC, vTrueQ);

      dynObs.push({
        u: 0,
        v: 0,
        qCam,
        nCam: nC,
        zCam: qCam[2],
        vNObs,
        xiBaseDot,
        wOpt: 1.0,
        frameGroupIdx: twIdx,
      });
      trueClassLabels.push(1);
    }

    // Class II: 20 unmodeled rigid custom-tool points (0.25m offset from CAD, exact rigid velocity!)
    for (let i = 0; i < 20; i++) {
      const pToolB: Vec3 = [0.55 + rng.uniform(-0.05, 0.05), 0.25 + rng.uniform(-0.05, 0.05), 0.45 + rng.uniform(-0.05, 0.05)];
      const nToolB = normalize3([1, rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)]);
      const qCam = transformPoint(T_gt, pToolB);
      const nC = mat3MulVec(T_gt.R, nToolB);
      const wCrossQ = cross3(wC, qCam);
      const vNObs = dot3(nC, [wCrossQ[0] + vC[0], wCrossQ[1] + vC[1], wCrossQ[2] + vC[2]]);
      dynObs.push({
        u: 0,
        v: 0,
        qCam,
        nCam: nC,
        zCam: qCam[2],
        vNObs,
        xiBaseDot,
        wOpt: 1.0,
        frameGroupIdx: twIdx,
      });
      trueClassLabels.push(2);
    }

    // Class III: 20 coupled non-rigid whipping cable points (offset from CAD AND velocity off by 0.45 m/s!)
    for (let i = 0; i < 20; i++) {
      const pCableB: Vec3 = [-0.25 + rng.uniform(-0.04, 0.04), -0.35 + rng.uniform(-0.04, 0.04), 0.35 + rng.uniform(-0.04, 0.04)];
      const nCableB = normalize3([0, -1, 0.2]);
      const qCam = transformPoint(T_gt, pCableB);
      const nC = mat3MulVec(T_gt.R, nCableB);
      const wCrossQ = cross3(wC, qCam);
      const vRigid = dot3(nC, [wCrossQ[0] + vC[0], wCrossQ[1] + vC[1], wCrossQ[2] + vC[2]]);
      dynObs.push({
        u: 0,
        v: 0,
        qCam,
        nCam: nC,
        zCam: qCam[2],
        vNObs: vRigid + 0.45, // violates rigid velocity
        xiBaseDot,
        wOpt: 1.0,
        frameGroupIdx: twIdx,
      });
      trueClassLabels.push(3);
    }
  }

  // Sanity-check annealed posterior directly
  const [g1I, g2I, g3I] = computeAnnealedEmPosterior(0.4, 0.5, 1.0, [0.7, 0.15, 0.15]);
  assert.ok(g1I > 0.85 && g1I > g2I && g1I > g3I, 'Class I point should have dominant gamma_I');
  const [g1II, g2II, g3II] = computeAnnealedEmPosterior(null, 0.4, 1.0, [0.7, 0.15, 0.15]);
  assert.ok(g2II > 0.85 && g2II > g1II && g2II > g3II, 'Class II point should have dominant gamma_II');
  const [g1III, g2III, g3III] = computeAnnealedEmPosterior(null, 8.5, 1.0, [0.7, 0.15, 0.15]);
  assert.ok(g3III > 0.90 && g3III > g1III && g3III > g2III, 'Class III point should have dominant gamma_III');

  const res = runStage4AnnealedEmLM(T_init, staticFrames, dynObs, dynGrids, {
    useVelocityField: true,
    estimateTimeOffset: true,
    temperatures: [16.0, 4.0, 1.0],
    gates: [0.16, 0.08, 0.045],
    itersPerStage: [8, 8, 10],
  });

  const finalErr = computePoseError(res.T_est, T_gt);
  const estDtMs = res.deltaTSec * 1000;

  assert.ok(
    finalErr.transMm < 1.2 && finalErr.rotDeg < 0.15,
    `Stage 4 EM-LM pose error too high: trans=${finalErr.transMm.toFixed(3)} mm, rot=${finalErr.rotDeg.toFixed(4)} deg`
  );
  assert.ok(
    Math.abs(estDtMs - 25.0) < 2.0,
    `Stage 4 7-DoF delta_t failed to recover 25ms: got ${estDtMs.toFixed(2)} ms`
  );

  // Verify EM posterior class separation accuracy > 95%
  let correct = 0;
  for (let i = 0; i < trueClassLabels.length; i++) {
    if (res.dynamicPointClasses[i] === trueClassLabels[i]) correct++;
  }
  const acc = correct / trueClassLabels.length;
  assert.ok(acc > 0.95, `EM 3-class separation accuracy too low: ${(acc * 100).toFixed(1)}%`);
});
