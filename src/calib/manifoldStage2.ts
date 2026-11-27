import {
  applyAdjoint,
  cloneSE3,
  composeSE3,
  computePoseError,
  cross3,
  dot3,
  expSE3,
  inverseTransformPoint,
  mat3MulVec,
  mat3TransposeMulVec,
  norm3,
  solveLinearSystem,
  SpatialHashGrid3D,
  transformPoint,
  type PointNormalSample,
  type SE3,
  type Vec3,
  type Vec6,
} from '../math/se3';
import { computeVelocityJacobianRow } from './fimPlanner';
import { backprojectPixel, erodeMask, fillInteriorMaskPinholes } from './frontend';
import type { CameraIntrinsics, LMTracePoint, RGBDFrame, WeightedPoint } from './types';

export interface StaticGeoFrameObs {
  qEncoder: number[];
  points: WeightedPoint[];
  modelGrid: SpatialHashGrid3D<PointNormalSample>;
}

export interface DynamicVelObs {
  qCam: Vec3;
  nCam: Vec3;
  zCam: number;
  vNObs: number;
  /** Base-frame spatial velocity twist xi^b * qdot_k */
  xiBaseDot: Vec6;
  w: number;
}

export function huberLoss(r: number, delta: number): number {
  const a = Math.abs(r);
  return a <= delta ? 0.5 * r * r : delta * (a - 0.5 * delta);
}

export function huberWeight(r: number, delta: number): number {
  const a = Math.abs(r);
  return a <= delta ? 1.0 : delta / a;
}

/**
 * Pure helper for static point-to-plane residual and analytical left-manifold Jacobian:
 *   e_geo = (n^c)^T * (s^c - q^c)
 *   J_geo = [(q^c x n^c)^T, (n^c)^T]
 */
export function evaluateStaticGeoResidual(
  T_cb: SE3,
  sBase: Vec3,
  qCam: Vec3,
  nCam: Vec3
): { e: number; J: Vec6 } {
  const sCam = transformPoint(T_cb, sBase);
  const e =
    nCam[0] * (sCam[0] - qCam[0]) +
    nCam[1] * (sCam[1] - qCam[1]) +
    nCam[2] * (sCam[2] - qCam[2]);
  // Exact derivative under left perturbation sCam' = exp(dxi^) sCam ≈ sCam + dphi x sCam + drho
  const sxn = cross3(sCam, nCam);
  const J: Vec6 = [sxn[0], sxn[1], sxn[2], nCam[0], nCam[1], nCam[2]];
  return { e, J };
}

/**
 * Pure helper for dynamic normal velocity residual and analytical left-manifold Jacobian:
 *   v_n_obs = +(dD/dt) * ((n^c)^T * q^c) / z^c   (STRICT POSITIVE SIGN)
 *   xi_tilde^c = Ad_{T_cb} * (xi^b * qdot_k) = [w_tilde^c; v_tilde^c]
 *   v_pred = w_tilde^c x q^c + v_tilde^c
 *   e_vel = (n^c)^T * v_pred - v_n_obs
 *   J_vel = (n^c)^T * [ [w_tilde^c]_x [q^c]_x - [v_pred]_x,  -[w_tilde^c]_x ]
 */
export function evaluateVelocityResidual(
  T_cb: SE3,
  obs: DynamicVelObs
): { e: number; J: Vec6 } {
  const xiTilde = applyAdjoint(T_cb, obs.xiBaseDot);
  const wTilde: Vec3 = [xiTilde[0], xiTilde[1], xiTilde[2]];
  const vTilde: Vec3 = [xiTilde[3], xiTilde[4], xiTilde[5]];
  const wCrossQ = cross3(wTilde, obs.qCam);
  const vPred: Vec3 = [
    wCrossQ[0] + vTilde[0],
    wCrossQ[1] + vTilde[1],
    wCrossQ[2] + vTilde[2],
  ];
  const e = dot3(obs.nCam, vPred) - obs.vNObs;
  const J = computeVelocityJacobianRow(obs.qCam, obs.nCam, wTilde, vPred);
  return { e, J };
}

/**
 * Extract dynamic normal velocity field observations from consecutive triples (t-1, t, t+1)
 * within an active single-joint sweep.
 *
 * Strictly uses the positive sign formula:
 *   v_n_obs = +(dD/dt) * ((n^c)^T * q^c) / z^c
 * where dD/dt = (D_{t+1} - D_{t-1}) / (2 * dt).
 */
export function extractDynamicVelocityObservations(
  frames: readonly RGBDFrame[],
  frameMasks: readonly Uint8Array[],
  xiBaseAtFrame: readonly Vec6[],
  K: CameraIntrinsics,
  opts: {
    filterEnabled: boolean;
    sigmaZ0: number;
    maxObsPerAction?: number;
  }
): DynamicVelObs[] {
  const T = frames.length;
  const W = K.width;
  const H = K.height;
  const maxObs = opts.maxObsPerAction ?? 550;
  const filterOn = opts.filterEnabled;
  const sigma0 = Math.max(opts.sigmaZ0, 0.0004);
  const sigma0Sq = sigma0 * sigma0;
  const sigmaG = 0.014;
  const twoSigmaGSq = 2 * sigmaG * sigmaG;
  const gamma = 1.5;
  const stride = 3;
  const nStep = 2;

  const raw: DynamicVelObs[] = [];

  for (let t = 1; t < T - 1; t++) {
    const fPrev = frames[t - 1];
    const fCur = frames[t];
    const fNext = frames[t + 1];
    const activeJ = fCur.activeJoint;
    const qdot = fCur.qdotEncoder[activeJ] ?? 0;
    if (Math.abs(qdot) < 0.14) continue;

    const dt = 0.5 * (fNext.timestamp - fPrev.timestamp);
    if (dt <= 1e-5) continue;

    const xiB = xiBaseAtFrame[t];
    const xiBaseDot: Vec6 = [
      xiB[0] * qdot,
      xiB[1] * qdot,
      xiB[2] * qdot,
      xiB[3] * qdot,
      xiB[4] * qdot,
      xiB[5] * qdot,
    ];

    const mPrev = frameMasks[t - 1];
    const mCur = filterOn
      ? erodeMask(fillInteriorMaskPinholes(frameMasks[t], W, H), W, H, 2)
      : frameMasks[t];
    const mNext = frameMasks[t + 1];
    const dPrev = fPrev.depth;
    const dCur = fCur.depth;
    const dNext = fNext.depth;

    for (let v = nStep + 1; v < H - nStep - 1; v += stride) {
      const row = v * W;
      for (let u = nStep + 1; u < W - nStep - 1; u += stride) {
        const idx = row + u;
        // Must be inside active eroded mask in all 3 consecutive frames
        if (!mPrev[idx] || !mCur[idx] || !mNext[idx]) continue;

        const zPrev = dPrev[idx];
        const zCur = dCur[idx];
        const zNext = dNext[idx];
        if (zPrev <= 0.1 || zCur <= 0.1 || zNext <= 0.1) continue;

        // Enforce smooth temporal continuity on same link surface (reject occlusion crossings & spikes)
        if (
          Math.abs(zNext - zCur) > 0.055 ||
          Math.abs(zCur - zPrev) > 0.055 ||
          Math.abs(zNext - zPrev) > 0.095
        ) {
          continue;
        }

        const zL = dCur[idx - nStep];
        const zR = dCur[idx + nStep];
        const zU = dCur[idx - nStep * W];
        const zD = dCur[idx + nStep * W];
        if (zL <= 0.08 || zR <= 0.08 || zU <= 0.08 || zD <= 0.08) continue;

        // Guard against single-pixel stereo mismatch spikes on center or tangent neighbors
        if (filterOn) {
          const z1L = dCur[idx - 1];
          const z1R = dCur[idx + 1];
          const z1U = dCur[idx - W];
          const z1D = dCur[idx + W];
          if (
            z1L <= 0.08 || z1R <= 0.08 || z1U <= 0.08 || z1D <= 0.08 ||
            Math.abs(z1L - zCur) > 0.032 ||
            Math.abs(z1R - zCur) > 0.032 ||
            Math.abs(z1U - zCur) > 0.032 ||
            Math.abs(z1D - zCur) > 0.032 ||
            Math.abs(zL - zCur) > 0.045 ||
            Math.abs(zR - zCur) > 0.045 ||
            Math.abs(zU - zCur) > 0.045 ||
            Math.abs(zD - zCur) > 0.045
          ) {
            continue;
          }
        }

        const dzU = (zR - zL) / (2 * nStep);
        const dzV = (zD - zU) / (2 * nStep);
        const gradSq = dzU * dzU + dzV * dzV;
        if (filterOn && Math.sqrt(gradSq) > 0.028) continue;

        const q = backprojectPixel(u, v, zCur, K);
        const pL = backprojectPixel(u - nStep, v, zL, K);
        const pR = backprojectPixel(u + nStep, v, zR, K);
        const pU = backprojectPixel(u, v - nStep, zU, K);
        const pD = backprojectPixel(u, v + nStep, zD, K);

        const dPdu: Vec3 = [pR[0] - pL[0], pR[1] - pL[1], pR[2] - pL[2]];
        const dPdv: Vec3 = [pD[0] - pU[0], pD[1] - pU[1], pD[2] - pU[2]];
        let n = cross3(dPdv, dPdu);
        const nn = norm3(n);
        if (nn < 1e-8) continue;
        n = [n[0] / nn, n[1] / nn, n[2] / nn];

        // Force camera-facing normal (n^T q < 0)
        if (dot3(n, q) > 0) {
          n = [-n[0], -n[1], -n[2]];
        }

        const qNorm = norm3(q);
        const cosInc = Math.max(0, -dot3(n, q) / Math.max(qNorm, 1e-6));
        if (filterOn && cosInc < 0.24) continue;

        // Center-difference depth rate dD/dt and exact positive-sign normal velocity
        const dDdt = (zNext - zPrev) / (2 * dt);
        if (Math.abs(dDdt) > 1.8) continue;

        const vNObs = +(dDdt) * (dot3(n, q) / zCur);

        let w = 1.0;
        if (filterOn) {
          const z4 = zCur * zCur * zCur * zCur;
          w = (Math.pow(cosInc, gamma) / (sigma0Sq * z4)) * Math.exp(-gradSq / twoSigmaGSq);
        }

        raw.push({
          qCam: q,
          nCam: n,
          zCam: zCur,
          vNObs,
          xiBaseDot,
          w,
        });
      }
    }
  }

  if (raw.length === 0) return [];

  // Normalize weights to (0, 1]
  let maxW = 0;
  for (let i = 0; i < raw.length; i++) {
    if (raw[i].w > maxW) maxW = raw[i].w;
  }
  if (maxW > 1e-18) {
    for (let i = 0; i < raw.length; i++) {
      raw[i].w = filterOn ? Math.max(1e-4, raw[i].w / maxW) : 1.0;
    }
  }

  if (raw.length > maxObs) {
    const out: DynamicVelObs[] = [];
    const step = raw.length / maxObs;
    for (let i = 0; i < maxObs; i++) {
      out.push(raw[Math.floor(i * step)]);
    }
    return out;
  }
  return raw;
}

export interface Stage2LMOptions {
  useVelocityField: boolean;
  lambdaVel?: number;
  huberDeltaGeo?: number;
  huberDeltaVel?: number;
  gates?: readonly number[]; // Coarse-to-fine gating distances [0.12, 0.07, 0.045]
  itersPerStage?: readonly number[];
  /** Optional ground truth ONLY for logging quantitative convergence curve (NEVER used by optimizer!) */
  T_cb_gt_for_log?: SE3;
}

export interface Stage2LMIterStep {
  point: LMTracePoint;
  T_est: SE3;
  trace: LMTracePoint[];
}

/**
 * Generator version of Stage 2 SE(3) Manifold LM Optimizer.
 * Yields after iter 0 and each accepted LM iteration so live UI charts & 3D frustums
 * can render real-time convergence trajectories without duplicating solver logic.
 */
export function* iterateStage2ManifoldLM(
  T_init: SE3,
  staticFrames: readonly StaticGeoFrameObs[],
  velObs: readonly DynamicVelObs[],
  opts: Stage2LMOptions
): Generator<Stage2LMIterStep, { T_est: SE3; trace: LMTracePoint[] }, void> {
  const gates = opts.gates ?? [0.13, 0.08, 0.045];
  const itersPerStage = opts.itersPerStage ?? [8, 7, 10];
  const deltaGeo = opts.huberDeltaGeo ?? 0.014;
  const deltaVel = opts.huberDeltaVel ?? 0.045;
  // Dimensionally balanced velocity weight (m^2 vs (m/s)^2 via dt^2 ~ 0.0042)
  const lambdaVel = opts.useVelocityField ? (opts.lambdaVel ?? 0.0025) : 0;

  let T = cloneSE3(T_init);
  let lmLambda = 1e-2;
  const trace: LMTracePoint[] = [];

  const logError = (T_cur: SE3) => {
    if (!opts.T_cb_gt_for_log) return { rotDeg: 0, transMm: 0 };
    return computePoseError(T_cur, opts.T_cb_gt_for_log);
  };

  interface GeoMatch {
    sBase: Vec3;
    qCam: Vec3;
    nCam: Vec3;
    w: number;
  }

  const collectGeoMatches = (T_cur: SE3, gateM: number): GeoMatch[] => {
    const matches: GeoMatch[] = [];
    for (let f = 0; f < staticFrames.length; f++) {
      const sf = staticFrames[f];
      const pts = sf.points;
      const grid = sf.modelGrid;
      for (let i = 0; i < pts.length; i++) {
        const pt = pts[i];
        const qBase = inverseTransformPoint(T_cur, pt.p);
        const nBaseObs = mat3TransposeMulVec(T_cur.R, pt.n);
        const hit = grid.findNearest(
          qBase,
          gateM,
          (cand) => dot3(cand.n, nBaseObs) > 0.35
        );
        if (!hit) continue;
        matches.push({
          sBase: hit.item.p,
          qCam: pt.p,
          nCam: pt.n,
          w: pt.w,
        });
      }
    }
    return matches;
  };

  const evaluateTotalCost = (T_cur: SE3, geoMatches: readonly GeoMatch[]): number => {
    let geoSum = 0;
    let geoW = 0;
    for (let i = 0; i < geoMatches.length; i++) {
      const m = geoMatches[i];
      const sCam = transformPoint(T_cur, m.sBase);
      const e =
        m.nCam[0] * (sCam[0] - m.qCam[0]) +
        m.nCam[1] * (sCam[1] - m.qCam[1]) +
        m.nCam[2] * (sCam[2] - m.qCam[2]);
      geoSum += m.w * huberLoss(e, deltaGeo);
      geoW += m.w;
    }

    let velSum = 0;
    let velW = 0;
    if (lambdaVel > 0 && velObs.length > 0) {
      for (let i = 0; i < velObs.length; i++) {
        const vo = velObs[i];
        const { e } = evaluateVelocityResidual(T_cur, vo);
        velSum += vo.w * huberLoss(e, deltaVel);
        velW += vo.w;
      }
    }

    const normGeo = geoW > 1e-9 ? geoSum / geoW : 0;
    const normVel = velW > 1e-9 ? velSum / velW : 0;
    return normGeo + lambdaVel * normVel;
  };

  // Initial trace point (iter 0 = Stage 1 or external initial extrinsic)
  let initMatches = collectGeoMatches(T, gates[0]);
  if (initMatches.length < 40 && gates[0] < 0.24) {
    initMatches = collectGeoMatches(T, 0.24);
  }
  const initCost = evaluateTotalCost(T, initMatches);
  const initErr = logError(T);
  const initPoint: LMTracePoint = {
    iter: 0,
    stage: 0,
    cost: initCost,
    rotErrDeg: initErr.rotDeg,
    transErrMm: initErr.transMm,
    geoCount: initMatches.length,
    velCount: lambdaVel > 0 ? velObs.length : 0,
    lambda: lmLambda,
  };
  trace.push(initPoint);
  yield { point: initPoint, T_est: cloneSE3(T), trace: [...trace] };

  let globalIter = 0;

  for (let stageIdx = 0; stageIdx < gates.length; stageIdx++) {
    const gateM = gates[stageIdx];
    const maxStageIters = itersPerStage[stageIdx] ?? 6;

    for (let sIter = 0; sIter < maxStageIters; sIter++) {
      globalIter++;
      let geoMatches = collectGeoMatches(T, gateM);
      // Data-driven fallback: only widen coarse Stage 0 gate if current pose has sparse (<40) matches
      if (stageIdx === 0 && geoMatches.length < 40 && gateM < 0.24) {
        geoMatches = collectGeoMatches(T, 0.24);
      }
      if (geoMatches.length < 25 && (lambdaVel <= 0 || velObs.length < 25)) {
        break;
      }

      // Compute normalized scale factors so geometric & velocity terms stay well balanced
      let totalGeoW = 0;
      for (let i = 0; i < geoMatches.length; i++) totalGeoW += geoMatches[i].w;
      const geoNormScale = totalGeoW > 1e-9 ? 1.0 / totalGeoW : 0;

      let totalVelW = 0;
      if (lambdaVel > 0) {
        for (let i = 0; i < velObs.length; i++) totalVelW += velObs[i].w;
      }
      const velNormScale = totalVelW > 1e-9 ? lambdaVel / totalVelW : 0;

      const H = new Float64Array(36);
      const g = new Float64Array(6);

      // Accumulate static geometric normal equations
      for (let i = 0; i < geoMatches.length; i++) {
        const m = geoMatches[i];
        const { e, J } = evaluateStaticGeoResidual(T, m.sBase, m.qCam, m.nCam);
        const wEff = geoNormScale * m.w * huberWeight(e, deltaGeo);
        for (let r = 0; r < 6; r++) {
          const wJr = wEff * J[r];
          g[r] += wJr * e;
          for (let c = r; c < 6; c++) {
            H[r * 6 + c] += wJr * J[c];
          }
        }
      }

      // Accumulate dynamic normal velocity field normal equations
      if (velNormScale > 0) {
        for (let i = 0; i < velObs.length; i++) {
          const vo = velObs[i];
          const { e, J } = evaluateVelocityResidual(T, vo);
          const wEff = velNormScale * vo.w * huberWeight(e, deltaVel);
          for (let r = 0; r < 6; r++) {
            const wJr = wEff * J[r];
            g[r] += wJr * e;
            for (let c = r; c < 6; c++) {
              H[r * 6 + c] += wJr * J[c];
            }
          }
        }
      }

      for (let r = 0; r < 6; r++) {
        for (let c = 0; c < r; c++) {
          H[r * 6 + c] = H[c * 6 + r];
        }
      }

      const curCost = evaluateTotalCost(T, geoMatches);
      let accepted = false;
      let stepNorm = 0;

      // Adaptive Levenberg-Marquardt inner damping attempts
      for (let attempt = 0; attempt < 5; attempt++) {
        const H_damped = new Float64Array(H);
        const negG = new Float64Array(6);
        for (let r = 0; r < 6; r++) {
          negG[r] = -g[r];
          H_damped[r * 6 + r] += lmLambda * Math.max(H[r * 6 + r], 1e-4) + 1e-7;
        }

        const sol = solveLinearSystem(H_damped, negG, 6);
        if (!sol) {
          lmLambda = Math.min(lmLambda * 5, 1e4);
          continue;
        }

        const dxi: Vec6 = [sol[0], sol[1], sol[2], sol[3], sol[4], sol[5]];
        stepNorm = Math.hypot(dxi[0], dxi[1], dxi[2], dxi[3], dxi[4], dxi[5]);
        if (!Number.isFinite(stepNorm) || stepNorm > 0.6) {
          lmLambda = Math.min(lmLambda * 5, 1e4);
          continue;
        }

        const T_trial = composeSE3(expSE3(dxi), T);
        const trialCost = evaluateTotalCost(T_trial, geoMatches);

        if (trialCost <= curCost + 1e-12) {
          T = T_trial;
          lmLambda = Math.max(lmLambda * 0.4, 1e-6);
          accepted = true;
          const err = logError(T);
          const stepPoint: LMTracePoint = {
            iter: globalIter,
            stage: stageIdx + 1,
            cost: trialCost,
            rotErrDeg: err.rotDeg,
            transErrMm: err.transMm,
            geoCount: geoMatches.length,
            velCount: lambdaVel > 0 ? velObs.length : 0,
            lambda: lmLambda,
          };
          trace.push(stepPoint);
          yield { point: stepPoint, T_est: cloneSE3(T), trace: [...trace] };
          break;
        } else {
          lmLambda = Math.min(lmLambda * 4, 1e4);
        }
      }

      if (!accepted || stepNorm < 1e-5) {
        break;
      }
    }
  }

  return { T_est: T, trace };
}

/**
 * Stage 2: Joint Static Point-to-Plane + Dynamic Velocity Field SE(3) Manifold LM Optimizer.
 * Synchronous wrapper over iterateStage2ManifoldLM for unit tests and fast Monte Carlo runs.
 */
export function runStage2ManifoldLM(
  T_init: SE3,
  staticFrames: readonly StaticGeoFrameObs[],
  velObs: readonly DynamicVelObs[],
  opts: Stage2LMOptions
): { T_est: SE3; trace: LMTracePoint[] } {
  const gen = iterateStage2ManifoldLM(T_init, staticFrames, velObs, opts);
  let next = gen.next();
  while (!next.done) {
    next = gen.next();
  }
  return next.value;
}
