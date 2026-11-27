import {
  applyAdjoint,
  cross3,
  dot3,
  mat3MulVec,
  norm3,
  transformPoint,
  type SE3,
  type SeededRNG,
  type Vec3,
  type Vec6,
} from '../../math/se3';
import type { MujocoPandaEngine } from '../../sim/mujocoEngine';
import {
  addMat6InPlace,
  computeVelocityJacobianRow,
  evaluateFIMMetrics,
  FIM_EPSILON,
  FIM_LAMBDA_VEL,
} from '../fimPlanner';
import type { CameraIntrinsics, MotionAction } from '../types';

export const V3_FIM_FOV_MARGIN_RATIO = 0.10; // 10% border margin from V3 spec
export const V3_FIM_ZBUF_MARGIN_M = 0.025; // Z-buffer static occlusion clearance

/**
 * Characteristic length L_c (meters) for dimensionless FIM scaling:
 *   J_tilde = [J_rot / L_c,  J_trans]
 * Eliminates the rad-vs-meter unit mismatch in D-optimality and E-optimality (lambda_min).
 */
export function getRobotCharacteristicLength(engine: MujocoPandaEngine): number {
  switch (engine.spec.id) {
    case 'aloha':
      return 0.32;
    case 'delta_parallel':
      return 0.42;
    case 'openarm':
      return 0.46;
    default:
      return 0.50;
  }
}

export interface V3ActionFimResult {
  I: Float64Array;
  visibleCount: number;
  occludedCount: number;
  totalFovCount: number;
}

/**
 * V3 Stage 3: Static Z-Buffer Occlusion-Aware Dimensionless FIM Computation.
 *  - Reuses Stage 1's static background depth envelope D_max(u,v) as a 3D static occlusion Z-Buffer.
 *  - Candidate surface sample p_i^c enters visible set V(M) ONLY if:
 *      1. Inside 10% FOV margin;
 *      2. Front-facing (cosInc >= 0.22);
 *      3. Unoccluded by static front obstacles: z_i < D_max(u_i, v_i) - tau_occ (when D_max > 0.14).
 *  - Applies characteristic-length L_c scaling on rotation Jacobian columns so I(M) is dimensionless.
 */
export function computeOcclusionAwareDimensionlessFIM(
  engine: MujocoPandaEngine,
  action: MotionAction,
  T_cb_hat: SE3,
  K: CameraIntrinsics,
  sigmaZ0: number,
  staticZBuffer: Float32Array | null,
  includeVelocity = true
): V3ActionFimResult {
  const I = new Float64Array(36);
  const sigma0Sq = Math.max(sigmaZ0, 0.0004) ** 2;
  const Lc = getRobotCharacteristicLength(engine);
  const invLc = 1.0 / Lc;

  const q = [...action.qBase];
  const sweepDurationSec = 0.65;
  const peakQdot = (Math.PI / 2) * ((action.qEnd - action.qStart) / sweepDurationSec);

  const uMin = K.width * V3_FIM_FOV_MARGIN_RATIO;
  const uMax = K.width * (1 - V3_FIM_FOV_MARGIN_RATIO);
  const vMin = K.height * V3_FIM_FOV_MARGIN_RATIO;
  const vMax = K.height * (1 - V3_FIM_FOV_MARGIN_RATIO);

  const sampleFracs = [0.15, 0.5, 0.85];
  const speedFracs = [0.55, 1.0, 0.55];

  let visibleCount = 0;
  let occludedCount = 0;
  let totalFovCount = 0;

  for (let s = 0; s < sampleFracs.length; s++) {
    const f = sampleFracs[s];
    q[action.jointIndex] = action.qStart + f * (action.qEnd - action.qStart);
    const qdot = peakQdot * speedFracs[s];

    const samplesB = engine.getRobotSurfacePointsBase(q, action.jointIndex, 4);
    const xiBase = engine.getEffectiveActionTwistBase(action.jointIndex, q);
    const xiBaseScaled: Vec6 = [
      xiBase[0] * qdot,
      xiBase[1] * qdot,
      xiBase[2] * qdot,
      xiBase[3] * qdot,
      xiBase[4] * qdot,
      xiBase[5] * qdot,
    ];
    const xiCam = applyAdjoint(T_cb_hat, xiBaseScaled);
    const wTilde: Vec3 = [xiCam[0], xiCam[1], xiCam[2]];
    const vTilde: Vec3 = [xiCam[3], xiCam[4], xiCam[5]];

    for (let i = 0; i < samplesB.length; i++) {
      const sp = samplesB[i];
      const qCam = transformPoint(T_cb_hat, sp.p);
      const z = qCam[2];
      if (z < 0.25 || z > 3.5) continue;

      const u = K.fx * (qCam[0] / z) + K.cx;
      const v = K.fy * (qCam[1] / z) + K.cy;
      if (u < uMin || u > uMax || v < vMin || v > vMax) continue;

      const nCam = mat3MulVec(T_cb_hat.R, sp.n);
      const qNorm = norm3(qCam);
      const cosInc = -dot3(nCam, qCam) / Math.max(qNorm, 1e-6);
      if (cosInc < 0.22) continue;

      totalFovCount++;

      // Static Z-Buffer occlusion culling: if a static obstacle sits closer than z_i, cull this sample!
      if (staticZBuffer) {
        const ui = Math.min(K.width - 1, Math.max(0, Math.floor(u)));
        const vi = Math.min(K.height - 1, Math.max(0, Math.floor(v)));
        const zStatic = staticZBuffer[vi * K.width + ui];
        if (zStatic > 0.14 && z >= zStatic - V3_FIM_ZBUF_MARGIN_M) {
          occludedCount++;
          continue;
        }
      }

      visibleCount++;

      const wPhys = Math.pow(cosInc, 1.5) / (sigma0Sq * z * z * z * z);
      const wGeo = wPhys * 1e-5;

      // Characteristic-length dimensionless static Jacobian: [ (q^c x n^c)/L_c, n^c ]
      const qxn = cross3(qCam, nCam);
      const Jg: Vec6 = [
        qxn[0] * invLc,
        qxn[1] * invLc,
        qxn[2] * invLc,
        nCam[0],
        nCam[1],
        nCam[2],
      ];

      for (let r = 0; r < 6; r++) {
        const wJr = wGeo * Jg[r];
        for (let c = r; c < 6; c++) {
          I[r * 6 + c] += wJr * Jg[c];
        }
      }

      if (includeVelocity && Math.abs(qdot) > 0.1) {
        const wCrossQ = cross3(wTilde, qCam);
        const vPred: Vec3 = [
          wCrossQ[0] + vTilde[0],
          wCrossQ[1] + vTilde[1],
          wCrossQ[2] + vTilde[2],
        ];
        const JvRaw = computeVelocityJacobianRow(qCam, nCam, wTilde, vPred);
        const Jv: Vec6 = [
          JvRaw[0] * invLc,
          JvRaw[1] * invLc,
          JvRaw[2] * invLc,
          JvRaw[3],
          JvRaw[4],
          JvRaw[5],
        ];
        const wVel = wGeo * FIM_LAMBDA_VEL;
        for (let r = 0; r < 6; r++) {
          const wJr = wVel * Jv[r];
          for (let c = r; c < 6; c++) {
            I[r * 6 + c] += wJr * Jv[c];
          }
        }
      }
    }
  }

  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < r; c++) {
      I[r * 6 + c] = I[c * 6 + r];
    }
  }

  return { I, visibleCount, occludedCount, totalFovCount };
}

/**
 * Select next V3 active motion action using Static Z-Buffer Occlusion-Aware Dimensionless FIM
 * (or deterministic random baseline for paired ablation).
 */
export function selectNextActionV3(
  engine: MujocoPandaEngine,
  pool: readonly MotionAction[],
  usedIds: ReadonlySet<string>,
  strategy: 'fim' | 'random',
  I_acc: Float64Array,
  T_cb_hat: SE3,
  K: CameraIntrinsics,
  sigmaZ0: number,
  staticZBuffer: Float32Array | null,
  includeVelocity: boolean,
  rng: SeededRNG
): { action: MotionAction; fimRes: V3ActionFimResult } | null {
  const remaining = pool.filter((a) => !usedIds.has(a.id));
  if (remaining.length === 0) return null;

  if (strategy === 'random') {
    const idx = Math.floor(rng.next() * remaining.length);
    const chosen = remaining[idx];
    const fimRes = computeOcclusionAwareDimensionlessFIM(
      engine,
      chosen,
      T_cb_hat,
      K,
      sigmaZ0,
      staticZBuffer,
      includeVelocity
    );
    return { action: chosen, fimRes };
  }

  let bestScore = -Infinity;
  let bestAction = remaining[0];
  let bestRes: V3ActionFimResult | null = null;

  for (let i = 0; i < remaining.length; i++) {
    const cand = remaining[i];
    const res = computeOcclusionAwareDimensionlessFIM(
      engine,
      cand,
      T_cb_hat,
      K,
      sigmaZ0,
      staticZBuffer,
      includeVelocity
    );
    const combined = new Float64Array(I_acc);
    addMat6InPlace(combined, res.I);
    const { score } = evaluateFIMMetrics(combined, FIM_EPSILON);
    if (score > bestScore) {
      bestScore = score;
      bestAction = cand;
      bestRes = res;
    }
  }

  return bestRes ? { action: bestAction, fimRes: bestRes } : null;
}
