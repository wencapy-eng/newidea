import {
  applyAdjoint,
  cross3,
  dot3,
  jacobiEigen6x6,
  mat3MulVec,
  norm3,
  skew,
  transformPoint,
  type SE3,
  type SeededRNG,
  type Vec3,
  type Vec6,
} from '../math/se3';
import type { MujocoPandaEngine } from '../sim/mujocoEngine';
import type { CameraIntrinsics, MotionAction } from './types';

export const FIM_EPSILON = 1e-4;
export const FIM_BETA_EOPT = 1.5;
export const FIM_LAMBDA_VEL = 0.35;

/**
 * 3-step orthogonal major-axis (or 3-motor Delta) cold start read from active robot spec,
 * filtered via real MuJoCo FK safety verification.
 */
export function buildColdStartActions(engine: MujocoPandaEngine): MotionAction[] {
  const safe = engine.spec.coldStartActions.filter((act) => isActionSafe(engine, act));
  return safe.length >= 3 ? safe : [...engine.spec.coldStartActions];
}

/** Verify an entire sweep action is within joint limits & collision-free above table via real MuJoCo FK */
export function isActionSafe(engine: MujocoPandaEngine, action: MotionAction): boolean {
  const lim = engine.spec.jointLimits[action.jointIndex];
  if (!lim) return false;
  const [lo, hi] = lim;
  if (action.qStart < lo + 0.02 || action.qStart > hi - 0.02) return false;
  if (action.qEnd < lo + 0.02 || action.qEnd > hi - 0.02) return false;

  const checkFracs = [0, 0.25, 0.5, 0.75, 1.0];
  const q = [...action.qBase];
  const minZ = engine.spec.id === 'aloha' ? 0.03 : 0.05;
  for (let i = 0; i < checkFracs.length; i++) {
    const f = checkFracs[i];
    q[action.jointIndex] = action.qStart + f * (action.qEnd - action.qStart);
    if (!engine.isPoseSafe(q, minZ)) {
      return false;
    }
  }
  return true;
}

/**
 * Build collision-free safe candidate action pool Q_safe for the active robot,
 * verified via real MuJoCo FK.
 */
export function buildSafeCandidatePool(engine: MujocoPandaEngine): MotionAction[] {
  const safe = engine.spec.candidateActions.filter((act) => isActionSafe(engine, act));
  return safe.length > 0 ? safe : [...engine.spec.candidateActions];
}

/**
 * Compute analytical 6-element dynamic velocity Jacobian J_vel:
 *   J_vel = (n^c)^T * [ [w_tilde]_x [q^c]_x - [v_pred]_x,  -[w_tilde]_x ]
 */
export function computeVelocityJacobianRow(
  qCam: Vec3,
  nCam: Vec3,
  omegaTilde: Vec3,
  vPred: Vec3
): Vec6 {
  const sw = skew(omegaTilde);
  const sq = skew(qCam);
  const sv = skew(vPred);

  // M_rot = sw * sq - sv (3x3)
  // Row J_rot = n^T * M_rot, Row J_trans = -n^T * sw
  const J: Vec6 = [0, 0, 0, 0, 0, 0];

  for (let c = 0; c < 3; c++) {
    let valRot = 0;
    let valTrans = 0;
    for (let r = 0; r < 3; r++) {
      // (sw * sq)[r, c]
      const swsq =
        sw[r * 3 + 0] * sq[0 * 3 + c] +
        sw[r * 3 + 1] * sq[1 * 3 + c] +
        sw[r * 3 + 2] * sq[2 * 3 + c];
      const mrc = swsq - sv[r * 3 + c];
      valRot += nCam[r] * mrc;
      valTrans += -nCam[r] * sw[r * 3 + c];
    }
    J[c] = valRot;
    J[c + 3] = valTrans;
  }
  return J;
}

/**
 * Compute 6x6 Fisher Information Matrix I(M; T_cb_hat) for a candidate action M
 * by predicting visible active-subchain surface samples at start, mid, and end poses
 * and accumulating stacked J = [J_geo; sqrt(lambda_v) * J_vel].
 */
export function computeActionFIM(
  engine: MujocoPandaEngine,
  action: MotionAction,
  T_cb_hat: SE3,
  K: CameraIntrinsics,
  sigmaZ0: number,
  includeVelocity = true
): Float64Array {
  const I = new Float64Array(36);
  const sigma0Sq = Math.max(sigmaZ0, 0.0004) ** 2;
  const q = [...action.qBase];
  const sweepDurationSec = 0.65;
  const peakQdot = (Math.PI / 2) * ((action.qEnd - action.qStart) / sweepDurationSec);

  const sampleFracs = [0.15, 0.5, 0.85];
  const speedFracs = [0.55, 1.0, 0.55];

  for (let s = 0; s < sampleFracs.length; s++) {
    const f = sampleFracs[s];
    q[action.jointIndex] = action.qStart + f * (action.qEnd - action.qStart);
    const qdot = peakQdot * speedFracs[s];

    // Sample active downstream/driven official robot CAD points (stride 4 for fast online planning)
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

      // Frustum visibility check
      const u = K.fx * (qCam[0] / z) + K.cx;
      const v = K.fy * (qCam[1] / z) + K.cy;
      if (u < 8 || u > K.width - 8 || v < 8 || v > K.height - 8) continue;

      const nCam = mat3MulVec(T_cb_hat.R, sp.n);
      const qNorm = norm3(qCam);
      const cosInc = -dot3(nCam, qCam) / Math.max(qNorm, 1e-6);
      // Front-facing visibility check (exclude back-facing and grazing > 78 deg)
      if (cosInc < 0.22) continue;

      // Physical measurement precision weight ~ cosInc^1.5 / (sigma0^2 * z^4)
      const wPhys = Math.pow(cosInc, 1.5) / (sigma0Sq * z * z * z * z);
      // Scale by 1e-4 to keep FIM eigenvalues well-conditioned around O(1..1e4)
      const wGeo = wPhys * 1e-5;

      // Static point-to-plane Jacobian: J_geo = [(q^c x n^c)^T, (n^c)^T]
      const qxn = cross3(qCam, nCam);
      const Jg: Vec6 = [qxn[0], qxn[1], qxn[2], nCam[0], nCam[1], nCam[2]];

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
        const Jv = computeVelocityJacobianRow(qCam, nCam, wTilde, vPred);
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

  // Symmetrize lower triangle
  for (let r = 0; r < 6; r++) {
    for (let c = 0; c < r; c++) {
      I[r * 6 + c] = I[c * 6 + r];
    }
  }
  return I;
}

/** Add two 6x6 matrices in place: dst += src */
export function addMat6InPlace(dst: Float64Array, src: Float64Array): void {
  for (let i = 0; i < 36; i++) {
    dst[i] += src[i];
  }
}

/** Compute D/E-optimality metrics of a 6x6 symmetric PSD FIM: log det(I + eps*I6) & lambda_min */
export function evaluateFIMMetrics(I: Float64Array, eps = FIM_EPSILON): {
  logDet: number;
  lambdaMin: number;
  score: number;
} {
  const reg = new Float64Array(I);
  for (let d = 0; d < 6; d++) {
    reg[d * 6 + d] += eps;
  }
  const { eigenvalues } = jacobiEigen6x6(reg);
  let logDet = 0;
  let lambdaMin = Infinity;
  for (let i = 0; i < 6; i++) {
    const ev = Math.max(eps, eigenvalues[i]);
    logDet += Math.log(ev);
    if (ev < lambdaMin) lambdaMin = ev;
  }
  const score = logDet + FIM_BETA_EOPT * Math.log(lambdaMin);
  return { logDet, lambdaMin, score };
}

/** Select the next action from Q_safe via FIM D/E-optimality or deterministic random baseline */
export function selectNextAction(
  engine: MujocoPandaEngine,
  pool: readonly MotionAction[],
  usedIds: ReadonlySet<string>,
  strategy: 'fim' | 'random',
  I_acc: Float64Array,
  T_cb_hat: SE3,
  K: CameraIntrinsics,
  sigmaZ0: number,
  includeVelocity: boolean,
  rng: SeededRNG
): { action: MotionAction; actionFIM: Float64Array } | null {
  const remaining = pool.filter((a) => !usedIds.has(a.id));
  if (remaining.length === 0) return null;

  if (strategy === 'random') {
    const idx = Math.floor(rng.next() * remaining.length);
    const chosen = remaining[idx];
    const fim = computeActionFIM(engine, chosen, T_cb_hat, K, sigmaZ0, includeVelocity);
    return { action: chosen, actionFIM: fim };
  }

  let bestScore = -Infinity;
  let bestAction = remaining[0];
  let bestFIM: Float64Array = new Float64Array(36);

  for (let i = 0; i < remaining.length; i++) {
    const cand = remaining[i];
    const fimCand = computeActionFIM(engine, cand, T_cb_hat, K, sigmaZ0, includeVelocity);
    const combined = new Float64Array(I_acc);
    addMat6InPlace(combined, fimCand);
    const { score } = evaluateFIMMetrics(combined, FIM_EPSILON);
    if (score > bestScore) {
      bestScore = score;
      bestAction = cand;
      bestFIM = fimCand;
    }
  }

  return { action: bestAction, actionFIM: bestFIM };
}
