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
} from '../../math/se3';
import { computeVelocityJacobianRow } from '../fimPlanner';
import { backprojectPixel, erodeMask, fillInteriorMaskPinholes } from '../frontend';
import type { CameraIntrinsics, LMTracePoint, RGBDFrame, WeightedPoint } from '../types';

export interface V3StaticFrameObs {
  qEncoder: number[];
  points: WeightedPoint[];
  modelGrid: SpatialHashGrid3D<PointNormalSample>;
}

/**
 * Unified dynamic frame observation carrying BOTH 3D surface point (for CAD geo matching + delta_t)
 * AND 5-frame Savitzky-Golay normal velocity (for Class I vs Class II vs Class III EM separation).
 */
export interface V3JointDynamicObs {
  u: number;
  v: number;
  qCam: Vec3;
  nCam: Vec3;
  zCam: number;
  vNObs: number;
  /** Base-frame instantaneous spatial velocity twist xi_k^b * qdot_k */
  xiBaseDot: Vec6;
  /** Optical physical weight (normalized) */
  wOpt: number;
  /** Index into dynamicFrameGrids[] for CAD point-to-plane lookup at this frame's q_enc(t) */
  frameGroupIdx: number;
}

export interface V3DynamicFrameGrid {
  frameIdx: number;
  actionIdx: number;
  modelGrid: SpatialHashGrid3D<PointNormalSample>;
}

/**
 * 5-Frame Savitzky-Golay Quadratic Depth Flow Estimator (zero phase lag, lower variance):
 *   dD/dt = (-2 D_{t-2} - D_{t-1} + D_{t+1} + 2 D_{t+2}) / (10 * dt)
 *   v_n_obs = +(dD/dt) * ((n^c)^T * q^c) / z^c
 */
export function extractSavitzkyGolayVelocityObservations(
  frames: readonly RGBDFrame[],
  frameMasks: readonly Uint8Array[],
  xiBaseAtFrame: readonly Vec6[],
  K: CameraIntrinsics,
  actionIdx: number,
  buildGridForQ: (qEnc: readonly number[]) => SpatialHashGrid3D<PointNormalSample>,
  opts: {
    filterEnabled: boolean;
    sigmaZ0: number;
    maxObsPerAction?: number;
  }
): { obs: V3JointDynamicObs[]; frameGrids: V3DynamicFrameGrid[] } {
  const T = frames.length;
  const W = K.width;
  const H = K.height;
  const maxObs = opts.maxObsPerAction ?? 520;
  const filterOn = opts.filterEnabled;
  const sigma0 = Math.max(opts.sigmaZ0, 0.0004);
  const sigma0Sq = sigma0 * sigma0;
  const sigmaG = 0.016;
  const twoSigmaGSq = 2 * sigmaG * sigmaG;
  const stride = 3;
  const nStep = 2;

  const raw: V3JointDynamicObs[] = [];
  const frameGrids: V3DynamicFrameGrid[] = [];

  for (let t = 2; t < T - 2; t++) {
    const fM2 = frames[t - 2];
    const fM1 = frames[t - 1];
    const f0 = frames[t];
    const fP1 = frames[t + 1];
    const fP2 = frames[t + 2];

    const activeJ = f0.activeJoint;
    const qdot = f0.qdotEncoder[activeJ] ?? 0;
    if (Math.abs(qdot) < 0.14) continue;

    const dt = (fP2.timestamp - fM2.timestamp) / 4;
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

    const mM2 = frameMasks[t - 2];
    const mM1 = frameMasks[t - 1];
    const m0 = filterOn
      ? erodeMask(fillInteriorMaskPinholes(frameMasks[t], W, H), W, H, 1)
      : frameMasks[t];
    const mP1 = frameMasks[t + 1];
    const mP2 = frameMasks[t + 2];

    const dM2 = fM2.depth;
    const dM1 = fM1.depth;
    const d0 = f0.depth;
    const dP1 = fP1.depth;
    const dP2 = fP2.depth;

    const frameStartCount = raw.length;
    const groupIdx = frameGrids.length;

    for (let v = nStep + 1; v < H - nStep - 1; v += stride) {
      const row = v * W;
      for (let u = nStep + 1; u < W - nStep - 1; u += stride) {
        const idx = row + u;
        if (!m0[idx] || !mM1[idx] || !mP1[idx]) continue;

        const z0 = d0[idx];
        const zM1 = dM1[idx];
        const zP1 = dP1[idx];
        if (z0 <= 0.12 || zM1 <= 0.12 || zP1 <= 0.12) continue;
        if (Math.abs(zP1 - z0) > 0.06 || Math.abs(z0 - zM1) > 0.06) continue;

        // Use 5-frame Savitzky-Golay quadratic derivative when outer frames are continuous,
        // gracefully falling back to 3-frame central difference near sweep start/end edges
        const zM2 = dM2[idx];
        const zP2 = dP2[idx];
        const has5 =
          mM2[idx] &&
          mP2[idx] &&
          zM2 > 0.12 &&
          zP2 > 0.12 &&
          Math.abs(zM1 - zM2) <= 0.065 &&
          Math.abs(zP2 - zP1) <= 0.065;

        const dDdt = has5
          ? (-2 * zM2 - zM1 + zP1 + 2 * zP2) / (10 * dt)
          : (zP1 - zM1) / (2 * dt);

        if (Math.abs(dDdt) > 1.8) continue;

        const zL = d0[idx - nStep];
        const zR = d0[idx + nStep];
        const zU = d0[idx - nStep * W];
        const zD = d0[idx + nStep * W];
        if (zL <= 0.10 || zR <= 0.10 || zU <= 0.10 || zD <= 0.10) continue;

        if (filterOn) {
          const z1L = d0[idx - 1];
          const z1R = d0[idx + 1];
          const z1U = d0[idx - W];
          const z1D = d0[idx + W];
          if (
            z1L <= 0.10 || z1R <= 0.10 || z1U <= 0.10 || z1D <= 0.10 ||
            Math.abs(z1L - z0) > 0.035 ||
            Math.abs(z1R - z0) > 0.035 ||
            Math.abs(z1U - z0) > 0.035 ||
            Math.abs(z1D - z0) > 0.035 ||
            Math.abs(zL - z0) > 0.048 ||
            Math.abs(zR - z0) > 0.048 ||
            Math.abs(zU - z0) > 0.048 ||
            Math.abs(zD - z0) > 0.048
          ) {
            continue;
          }
        }

        const dzU = (zR - zL) / (2 * nStep);
        const dzV = (zD - zU) / (2 * nStep);
        const gradSq = dzU * dzU + dzV * dzV;
        if (filterOn && Math.sqrt(gradSq) > 0.030) continue;

        const q = backprojectPixel(u, v, z0, K);
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
        if (dot3(n, q) > 0) n = [-n[0], -n[1], -n[2]];

        const cosInc = Math.max(0, -dot3(n, q) / Math.max(norm3(q), 1e-6));
        if (filterOn && cosInc < 0.22) continue;

        const vNObs = +(dDdt) * (dot3(n, q) / z0);
        const wOpt = filterOn
          ? (Math.pow(cosInc, 1.5) / (sigma0Sq * z0 * z0 * z0 * z0)) * Math.exp(-gradSq / twoSigmaGSq)
          : 1.0;

        raw.push({
          u,
          v,
          qCam: q,
          nCam: n,
          zCam: z0,
          vNObs,
          xiBaseDot,
          wOpt,
          frameGroupIdx: groupIdx,
        });
      }
    }

    if (raw.length > frameStartCount) {
      frameGrids.push({
        frameIdx: t,
        actionIdx,
        modelGrid: buildGridForQ(f0.qEncoder),
      });
    }
  }

  if (raw.length === 0) return { obs: [], frameGrids: [] };

  // Normalize wOpt to (0, 1]
  let maxW = 0;
  for (let i = 0; i < raw.length; i++) if (raw[i].wOpt > maxW) maxW = raw[i].wOpt;
  if (maxW > 1e-18) {
    for (let i = 0; i < raw.length; i++) {
      raw[i].wOpt = filterOn ? Math.max(1e-3, raw[i].wOpt / maxW) : 1.0;
    }
  }

  if (raw.length > maxObs) {
    const out: V3JointDynamicObs[] = [];
    const step = raw.length / maxObs;
    for (let i = 0; i < maxObs; i++) {
      out.push(raw[Math.floor(i * step)]);
    }
    return { obs: out, frameGrids };
  }

  return { obs: raw, frameGrids };
}

export interface Stage4EmLmOptions {
  useVelocityField: boolean;
  estimateTimeOffset: boolean;
  /** Annealing temperatures (default [16.0, 4.0, 1.0]) */
  temperatures?: readonly number[];
  /** Coarse-to-fine CAD correspondence gates matching temperatures */
  gates?: readonly number[];
  itersPerStage?: readonly number[];
  sigmaGeoM?: number;
  sigmaVelMs?: number;
  tauGeo?: number;
  tauVel?: number;
  lambdaVel?: number;
  /** Optional ground truth ONLY for quantitative convergence logging (NEVER read by optimizer) */
  T_cb_gt_for_log?: SE3;
}

export interface Stage4EmLmResult {
  T_est: SE3;
  /** Estimated camera-encoder time offset delta_t in seconds (0 when disabled) */
  deltaTSec: number;
  /** Converged EM mixture priors [pi_I, pi_II, pi_III] */
  classPriors: [number, number, number];
  /** Posterior-weighted counts [Class I, Class II, Class III] on final stage */
  classCounts: [number, number, number];
  /** Per-observation class assignments (1, 2, or 3) for dynamic observations in final stage */
  dynamicPointClasses: Uint8Array;
  trace: LMTracePoint[];
}

/**
 * Pure helper: compute 3-component annealed EM posterior probabilities (gamma_I, gamma_II, gamma_III)
 * for a single joint observation given normalized residuals dGeo = |e_geo|/sigma_geo and dVel = |e_vel|/sigma_vel.
 */
export function computeAnnealedEmPosterior(
  dGeo: number | null, // null when no CAD match within gate (e.g., Class II custom tool or Class III cable)
  dVel: number | null, // null on static pause frames (marginalizes velocity out)
  T_temp: number,
  priors: readonly [number, number, number],
  tauGeo = 2.6,
  tauVel = 2.8
): [number, number, number] {
  const inv2T = 1 / (2 * Math.max(T_temp, 0.25));
  const tauG2 = tauGeo * tauGeo;
  const tauV2 = tauVel * tauVel;

  // Unmatched points receive a large effective geometric residual (outside CAD manifold)
  const dG2 = dGeo !== null ? Math.min(dGeo * dGeo, 250) : 64.0;
  const alphaGeo = Math.exp(-(dG2 - tauG2) * inv2T);

  if (dVel === null) {
    // Static pause frame: marginalize velocity -> Class I vs non-CAD (Class II+III)
    const p1 = priors[0] * alphaGeo;
    const pOther = priors[1] + priors[2];
    const Z = p1 + pOther + 1e-18;
    const g1 = p1 / Z;
    const rem = 1 - g1;
    const split = priors[1] / Math.max(priors[1] + priors[2], 1e-9);
    return [g1, rem * split, rem * (1 - split)];
  }

  const dV2 = Math.min(dVel * dVel, 250);
  const alphaVel = Math.exp(-(dV2 - tauV2) * inv2T);

  const u1 = priors[0] * alphaGeo * alphaVel;
  const u2 = priors[1] * alphaVel;
  const u3 = priors[2];
  const Z = u1 + u2 + u3 + 1e-18;
  return [u1 / Z, u2 / Z, u3 / Z];
}

export interface Stage4EmLmIterStep {
  point: LMTracePoint;
  T_est: SE3;
  deltaTSec: number;
  classPriors: [number, number, number];
  trace: LMTracePoint[];
}

/**
 * Generator version of V3 Stage 4 3-Component Annealed EM-LM Optimizer.
 * Yields after iter 0 and every accepted EM-LM iteration for live UI convergence tracking.
 */
export function* iterateStage4AnnealedEmLM(
  T_init: SE3,
  staticFrames: readonly V3StaticFrameObs[],
  dynObs: readonly V3JointDynamicObs[],
  dynFrameGrids: readonly V3DynamicFrameGrid[],
  opts: Stage4EmLmOptions
): Generator<Stage4EmLmIterStep, Stage4EmLmResult, void> {
  const temperatures = opts.temperatures ?? [16.0, 4.0, 1.0];
  const gates = opts.gates ?? [0.14, 0.08, 0.048];
  const itersPerStage = opts.itersPerStage ?? [7, 7, 9];
  const sigmaGeo = opts.sigmaGeoM ?? 0.0085;
  const sigmaVel = opts.sigmaVelMs ?? 0.045;
  const tauGeo = opts.tauGeo ?? 2.6;
  const tauVel = opts.tauVel ?? 2.8;
  const useVel = opts.useVelocityField;
  const lambdaVel = useVel ? (opts.lambdaVel ?? 0.0032) : 0;
  const estDt = opts.estimateTimeOffset && dynObs.length >= 25;
  const dim = estDt ? 7 : 6;

  let T = cloneSE3(T_init);
  let deltaT = 0.0; // seconds (range clamped to [-0.10, +0.15] s)
  let priors: [number, number, number] = [0.76, 0.14, 0.10];
  let lmLambda = 1e-2;
  const trace: LMTracePoint[] = [];

  const logError = (T_cur: SE3) => {
    if (!opts.T_cb_gt_for_log) return { rotDeg: 0, transMm: 0 };
    return computePoseError(T_cur, opts.T_cb_gt_for_log);
  };

  interface StaticMatch {
    sBase: Vec3 | null;
    qCam: Vec3;
    nCam: Vec3;
    wOpt: number;
  }

  interface DynMatch {
    obsIdx: number;
    sBase: Vec3 | null;
    qCam: Vec3;
    nCam: Vec3;
    vNObs: number;
    xiBaseDot: Vec6;
    wOpt: number;
  }

  const collectStaticMatches = (T_cur: SE3, gateM: number): StaticMatch[] => {
    const out: StaticMatch[] = [];
    for (let f = 0; f < staticFrames.length; f++) {
      const sf = staticFrames[f];
      const pts = sf.points;
      const grid = sf.modelGrid;
      for (let i = 0; i < pts.length; i++) {
        const pt = pts[i];
        const qBase = inverseTransformPoint(T_cur, pt.p);
        const nBaseObs = mat3TransposeMulVec(T_cur.R, pt.n);
        const hit = grid.findNearest(qBase, gateM, (cand) => dot3(cand.n, nBaseObs) > 0.35);
        out.push({
          sBase: hit ? hit.item.p : null,
          qCam: pt.p,
          nCam: pt.n,
          wOpt: pt.w,
        });
      }
    }
    return out;
  };

  const collectDynMatches = (T_cur: SE3, dtSec: number, gateM: number): DynMatch[] => {
    const out: DynMatch[] = [];
    for (let i = 0; i < dynObs.length; i++) {
      const d = dynObs[i];
      const gridEntry = dynFrameGrids[d.frameGroupIdx];
      let sBase: Vec3 | null = null;
      if (gridEntry) {
        // Back-shift observed point in base frame by -v_b * dtSec when querying q_enc(t) CAD grid
        const qBase = inverseTransformPoint(T_cur, d.qCam);
        const wB: Vec3 = [d.xiBaseDot[0], d.xiBaseDot[1], d.xiBaseDot[2]];
        const vB: Vec3 = [d.xiBaseDot[3], d.xiBaseDot[4], d.xiBaseDot[5]];
        const wCrossQ = cross3(wB, qBase);
        const qBaseShifted: Vec3 = [
          qBase[0] - (wCrossQ[0] + vB[0]) * dtSec,
          qBase[1] - (wCrossQ[1] + vB[1]) * dtSec,
          qBase[2] - (wCrossQ[2] + vB[2]) * dtSec,
        ];
        const nBaseObs = mat3TransposeMulVec(T_cur.R, d.nCam);
        const hit = gridEntry.modelGrid.findNearest(
          qBaseShifted,
          gateM,
          (cand) => dot3(cand.n, nBaseObs) > 0.35
        );
        if (hit) sBase = hit.item.p;
      }
      out.push({
        obsIdx: i,
        sBase,
        qCam: d.qCam,
        nCam: d.nCam,
        vNObs: d.vNObs,
        xiBaseDot: d.xiBaseDot,
        wOpt: d.wOpt,
      });
    }
    return out;
  };

  const evalGeoWithDt = (
    T_cur: SE3,
    dtSec: number,
    sBase: Vec3,
    qCam: Vec3,
    nCam: Vec3,
    xiBaseDot: Vec6 | null
  ): { e: number; J6: Vec6; Jdt: number } => {
    const sCam0 = transformPoint(T_cur, sBase);
    let sCam = sCam0;
    let Jdt = 0;

    if (xiBaseDot) {
      const xiCam = applyAdjoint(T_cur, xiBaseDot);
      const wC: Vec3 = [xiCam[0], xiCam[1], xiCam[2]];
      const vC: Vec3 = [xiCam[3], xiCam[4], xiCam[5]];
      const wCrossS = cross3(wC, sCam0);
      const vPredS: Vec3 = [wCrossS[0] + vC[0], wCrossS[1] + vC[1], wCrossS[2] + vC[2]];
      sCam = [
        sCam0[0] + vPredS[0] * dtSec,
        sCam0[1] + vPredS[1] * dtSec,
        sCam0[2] + vPredS[2] * dtSec,
      ];
      Jdt = dot3(nCam, vPredS);
    }

    const e =
      nCam[0] * (sCam[0] - qCam[0]) +
      nCam[1] * (sCam[1] - qCam[1]) +
      nCam[2] * (sCam[2] - qCam[2]);
    const sxn = cross3(sCam, nCam);
    return {
      e,
      J6: [sxn[0], sxn[1], sxn[2], nCam[0], nCam[1], nCam[2]],
      Jdt,
    };
  };

  const evalVel = (T_cur: SE3, qCam: Vec3, nCam: Vec3, vNObs: number, xiBaseDot: Vec6) => {
    const xiCam = applyAdjoint(T_cur, xiBaseDot);
    const wC: Vec3 = [xiCam[0], xiCam[1], xiCam[2]];
    const vC: Vec3 = [xiCam[3], xiCam[4], xiCam[5]];
    const wCrossQ = cross3(wC, qCam);
    const vPred: Vec3 = [wCrossQ[0] + vC[0], wCrossQ[1] + vC[1], wCrossQ[2] + vC[2]];
    const e = dot3(nCam, vPred) - vNObs;
    const J6 = computeVelocityJacobianRow(qCam, nCam, wC, vPred);
    return { e, J6 };
  };

  // Log initial state (iter 0)
  const initErr = logError(T);
  const initPoint: LMTracePoint = {
    iter: 0,
    stage: 0,
    cost: 0,
    rotErrDeg: initErr.rotDeg,
    transErrMm: initErr.transMm,
    geoCount: 0,
    velCount: useVel ? dynObs.length : 0,
    lambda: lmLambda,
  };
  trace.push(initPoint);
  yield {
    point: initPoint,
    T_est: cloneSE3(T),
    deltaTSec: deltaT,
    classPriors: [...priors] as [number, number, number],
    trace: [...trace],
  };

  let globalIter = 0;
  let finalClassCounts: [number, number, number] = [0, 0, 0];
  const dynamicPointClasses = new Uint8Array(dynObs.length);

  for (let stageIdx = 0; stageIdx < temperatures.length; stageIdx++) {
    const T_temp = temperatures[stageIdx];
    const gateM = gates[Math.min(stageIdx, gates.length - 1)];
    const maxIters = itersPerStage[Math.min(stageIdx, itersPerStage.length - 1)];

    for (let sIter = 0; sIter < maxIters; sIter++) {
      globalIter++;

      let staticMatches = collectStaticMatches(T, gateM);
      let matchedStaticCount = 0;
      for (let i = 0; i < staticMatches.length; i++) {
        if (staticMatches[i].sBase !== null) matchedStaticCount++;
      }
      // Data-driven fallback: widen early stage gate only if matches are very sparse
      if (stageIdx === 0 && matchedStaticCount < 40 && gateM < 0.28) {
        staticMatches = collectStaticMatches(T, 0.28);
        matchedStaticCount = staticMatches.filter((m) => m.sBase !== null).length;
      }

      const dynMatches = collectDynMatches(T, deltaT, gateM);
      if (matchedStaticCount < 20 && (!useVel || dynMatches.length < 25)) break;

      // ===== E-STEP: Compute annealed posteriors gamma_{i,c} for static & dynamic observations =====
      const staticGamma1 = new Float64Array(staticMatches.length);
      const dynGamma1 = new Float64Array(dynMatches.length);
      const dynGamma2 = new Float64Array(dynMatches.length);
      const dynGamma3 = new Float64Array(dynMatches.length);

      let sumG1 = 0;
      let sumG2 = 0;
      let sumG3 = 0;
      let totalGeoW = 0;
      let totalVelW = 0;

      for (let i = 0; i < staticMatches.length; i++) {
        const m = staticMatches[i];
        let dGeo: number | null = null;
        if (m.sBase) {
          const { e } = evalGeoWithDt(T, 0, m.sBase, m.qCam, m.nCam, null);
          dGeo = Math.abs(e) / sigmaGeo;
        }
        const [g1] = computeAnnealedEmPosterior(dGeo, null, T_temp, priors, tauGeo, tauVel);
        staticGamma1[i] = m.sBase ? g1 : 0;
        if (m.sBase) {
          totalGeoW += m.wOpt * staticGamma1[i];
        }
      }

      for (let i = 0; i < dynMatches.length; i++) {
        const dm = dynMatches[i];
        let dGeo: number | null = null;
        if (dm.sBase) {
          const { e } = evalGeoWithDt(T, deltaT, dm.sBase, dm.qCam, dm.nCam, dm.xiBaseDot);
          dGeo = Math.abs(e) / sigmaGeo;
        }
        const { e: eVel } = evalVel(T, dm.qCam, dm.nCam, dm.vNObs, dm.xiBaseDot);
        const dVel = Math.abs(eVel) / sigmaVel;

        const [g1, g2, g3] = computeAnnealedEmPosterior(
          dGeo,
          useVel ? dVel : null,
          T_temp,
          priors,
          tauGeo,
          tauVel
        );
        dynGamma1[i] = dm.sBase ? g1 : 0;
        dynGamma2[i] = g2;
        dynGamma3[i] = g3;

        sumG1 += g1;
        sumG2 += g2;
        sumG3 += g3;

        if (dm.sBase) {
          totalGeoW += dm.wOpt * dynGamma1[i];
        }
        if (useVel) {
          // Class I + Class II BOTH drive velocity; Class III silenced!
          totalVelW += dm.wOpt * (g1 + g2);
        }

        // Record argmax class for visualization overlay
        dynamicPointClasses[dm.obsIdx] =
          g1 >= g2 && g1 >= g3 ? 1 : g2 >= g3 ? 2 : 3;
      }

      // Adaptive Dirichlet-regularized prior update (prevents any component collapse)
      if (dynMatches.length >= 20) {
        const kappa0 = 6.0;
        const denom = dynMatches.length + 3 * kappa0;
        priors = [
          Math.max(0.20, Math.min(0.92, (sumG1 + kappa0) / denom)),
          Math.max(0.04, Math.min(0.45, (sumG2 + kappa0) / denom)),
          Math.max(0.04, Math.min(0.45, (sumG3 + kappa0) / denom)),
        ];
        const pSum = priors[0] + priors[1] + priors[2];
        priors = [priors[0] / pSum, priors[1] / pSum, priors[2] / pSum];
        finalClassCounts = [sumG1, sumG2, sumG3];
      }

      // ===== M-STEP: Weighted Gauss-Newton / LM update on SE(3) [x R for delta_t] =====
      const geoNormScale = totalGeoW > 1e-9 ? 1.0 / totalGeoW : 0;
      const velNormScale = useVel && totalVelW > 1e-9 ? lambdaVel / totalVelW : 0;

      // Evaluate fixed-posterior expected negative log-likelihood Q(theta | theta^k)
      const evalExpectedCost = (T_cand: SE3, dtCand: number): number => {
        let gSum = 0;
        for (let i = 0; i < staticMatches.length; i++) {
          const m = staticMatches[i];
          if (!m.sBase || staticGamma1[i] < 1e-4) continue;
          const { e } = evalGeoWithDt(T_cand, 0, m.sBase, m.qCam, m.nCam, null);
          gSum += 0.5 * geoNormScale * m.wOpt * staticGamma1[i] * e * e;
        }
        let vSum = 0;
        for (let i = 0; i < dynMatches.length; i++) {
          const dm = dynMatches[i];
          if (dm.sBase && dynGamma1[i] >= 1e-4) {
            const { e } = evalGeoWithDt(T_cand, dtCand, dm.sBase, dm.qCam, dm.nCam, dm.xiBaseDot);
            // Weight moving-frame geo slightly higher when estimating delta_t so time offset converges cleanly
            const dynGeoWeight = estDt ? 1.35 : 0.75;
            gSum += 0.5 * geoNormScale * dynGeoWeight * dm.wOpt * dynGamma1[i] * e * e;
          }
          if (velNormScale > 0) {
            const wVelClass = dynGamma1[i] + dynGamma2[i];
            if (wVelClass >= 1e-4) {
              const { e } = evalVel(T_cand, dm.qCam, dm.nCam, dm.vNObs, dm.xiBaseDot);
              vSum += 0.5 * velNormScale * dm.wOpt * wVelClass * e * e;
            }
          }
        }
        return gSum + vSum;
      };

      const H = new Float64Array(dim * dim);
      const grad = new Float64Array(dim);

      // 1. Static pause frame Class I geometry
      for (let i = 0; i < staticMatches.length; i++) {
        const m = staticMatches[i];
        if (!m.sBase || staticGamma1[i] < 1e-4) continue;
        const { e, J6 } = evalGeoWithDt(T, 0, m.sBase, m.qCam, m.nCam, null);
        const wEff = geoNormScale * m.wOpt * staticGamma1[i];
        for (let r = 0; r < 6; r++) {
          const wJr = wEff * J6[r];
          grad[r] += wJr * e;
          for (let c = r; c < 6; c++) {
            H[r * dim + c] += wJr * J6[c];
          }
        }
      }

      // 2. Dynamic sweep frame Class I geometry (+ delta_t Jacobian) & Class I+II velocity
      const dynGeoMult = estDt ? 1.35 : 0.75;
      for (let i = 0; i < dynMatches.length; i++) {
        const dm = dynMatches[i];

        if (dm.sBase && dynGamma1[i] >= 1e-4) {
          const { e, J6, Jdt } = evalGeoWithDt(
            T,
            deltaT,
            dm.sBase,
            dm.qCam,
            dm.nCam,
            dm.xiBaseDot
          );
          const wEff = geoNormScale * dynGeoMult * dm.wOpt * dynGamma1[i];
          const Jrow = estDt ? [J6[0], J6[1], J6[2], J6[3], J6[4], J6[5], Jdt] : J6;
          for (let r = 0; r < dim; r++) {
            const wJr = wEff * Jrow[r];
            grad[r] += wJr * e;
            for (let c = r; c < dim; c++) {
              H[r * dim + c] += wJr * Jrow[c];
            }
          }
        }

        if (velNormScale > 0) {
          // Class I + Class II BOTH contribute to dynamic velocity; Class III is silenced!
          const wVelClass = dynGamma1[i] + dynGamma2[i];
          if (wVelClass >= 1e-4) {
            const { e, J6 } = evalVel(T, dm.qCam, dm.nCam, dm.vNObs, dm.xiBaseDot);
            const wEff = velNormScale * dm.wOpt * wVelClass;
            for (let r = 0; r < 6; r++) {
              const wJr = wEff * J6[r];
              grad[r] += wJr * e;
              for (let c = r; c < 6; c++) {
                H[r * dim + c] += wJr * J6[c];
              }
            }
          }
        }
      }

      for (let r = 0; r < dim; r++) {
        for (let c = 0; c < r; c++) {
          H[r * dim + c] = H[c * dim + r];
        }
      }

      const curCost = evalExpectedCost(T, deltaT);
      let accepted = false;
      let stepNorm = 0;

      for (let attempt = 0; attempt < 5; attempt++) {
        const H_damped = new Float64Array(H);
        const negG = new Float64Array(dim);
        for (let r = 0; r < dim; r++) {
          negG[r] = -grad[r];
          H_damped[r * dim + r] += lmLambda * Math.max(H[r * dim + r], 1e-4) + 1e-7;
        }

        const sol = solveLinearSystem(H_damped, negG, dim);
        if (!sol) {
          lmLambda = Math.min(lmLambda * 5, 1e4);
          continue;
        }

        const dxi: Vec6 = [sol[0], sol[1], sol[2], sol[3], sol[4], sol[5]];
        const dDt = estDt ? Math.max(-0.04, Math.min(0.04, sol[6])) : 0;
        stepNorm = Math.hypot(dxi[0], dxi[1], dxi[2], dxi[3], dxi[4], dxi[5], dDt);
        if (!Number.isFinite(stepNorm) || stepNorm > 0.65) {
          lmLambda = Math.min(lmLambda * 5, 1e4);
          continue;
        }

        const T_trial = composeSE3(expSE3(dxi), T);
        const dtTrial = estDt ? Math.max(-0.10, Math.min(0.15, deltaT + dDt)) : 0;
        const trialCost = evalExpectedCost(T_trial, dtTrial);

        if (trialCost <= curCost + 1e-12) {
          T = T_trial;
          deltaT = dtTrial;
          lmLambda = Math.max(lmLambda * 0.4, 1e-6);
          accepted = true;
          const err = logError(T);
          const stepPoint: LMTracePoint = {
            iter: globalIter,
            stage: stageIdx + 1,
            cost: trialCost,
            rotErrDeg: err.rotDeg,
            transErrMm: err.transMm,
            geoCount: matchedStaticCount,
            velCount: useVel ? dynMatches.length : 0,
            lambda: lmLambda,
          };
          trace.push(stepPoint);
          yield {
            point: stepPoint,
            T_est: cloneSE3(T),
            deltaTSec: deltaT,
            classPriors: [...priors] as [number, number, number],
            trace: [...trace],
          };
          break;
        } else {
          lmLambda = Math.min(lmLambda * 4, 1e4);
        }
      }

      if (!accepted || stepNorm < 1e-5) break;
    }
  }

  return {
    T_est: T,
    deltaTSec: deltaT,
    classPriors: priors,
    classCounts: finalClassCounts,
    dynamicPointClasses,
    trace,
  };
}

/**
 * V3 Stage 4: 3-Component Latent-Variable Deterministic Annealing EM-LM Optimizer.
 * Synchronous wrapper over iterateStage4AnnealedEmLM for unit tests and fast Monte Carlo runs.
 */
export function runStage4AnnealedEmLM(
  T_init: SE3,
  staticFrames: readonly V3StaticFrameObs[],
  dynObs: readonly V3JointDynamicObs[],
  dynFrameGrids: readonly V3DynamicFrameGrid[],
  opts: Stage4EmLmOptions
): Stage4EmLmResult {
  const gen = iterateStage4AnnealedEmLM(T_init, staticFrames, dynObs, dynFrameGrids, opts);
  let next = gen.next();
  while (!next.done) {
    next = gen.next();
  }
  return next.value;
}
