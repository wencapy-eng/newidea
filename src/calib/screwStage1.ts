import {
  cloneSE3,
  composeSE3,
  cross3,
  dot3,
  expSE3,
  identitySE3,
  logSE3,
  mat3Det,
  mat3Mul,
  mat3MulVec,
  mat3Transpose,
  norm3,
  normalize3,
  skew,
  solveLinearSystem,
  SpatialHashGrid3D,
  svd3x3,
  transformPoint,
  type Mat3,
  type SE3,
  type Vec3,
  type Vec6,
} from '../math/se3';
import type { MotionAction, ScrewObservation, WeightedPoint } from './types';

function huberWeight(residual: number, delta: number): number {
  const absR = Math.abs(residual);
  return absR <= delta ? 1.0 : delta / absR;
}

/**
 * Weighted Trimmed Point-to-Plane ICP on SE(3):
 *  - Finds nearest target point via 3D SpatialHashGrid3D
 *  - Keeps top `trimFraction` (default 0.78) smallest point-to-plane residuals
 *  - Applies Huber kernel + physical point weights
 *  - Solves 6x6 damped Gauss-Newton update on SE(3) left manifold: T <- exp(dxi^) T
 *
 * Returns transform `T` mapping source cloud -> target cloud (p_tgt ≈ T * p_src).
 */
export function runTrimmedPointToPlaneICP(
  srcCloud: readonly WeightedPoint[],
  tgtCloud: readonly WeightedPoint[],
  initT: SE3 = identitySE3(),
  opts?: {
    maxIters?: number;
    maxDist?: number;
    trimFraction?: number;
    huberDelta?: number;
    p2pWeight?: number;
    maxStepRot?: number;
    maxStepTrans?: number;
  }
): { T: SE3; rms: number; inlierCount: number } {
  const maxIters = opts?.maxIters ?? 16;
  const maxDist = opts?.maxDist ?? 0.12;
  const trimFraction = opts?.trimFraction ?? 0.78;
  const huberDelta = opts?.huberDelta ?? 0.012;
  const baseP2P = opts?.p2pWeight ?? 0.08;
  const maxStepRot = opts?.maxStepRot ?? 0.12;
  const maxStepTrans = opts?.maxStepTrans ?? 0.07;

  if (srcCloud.length < 18 || tgtCloud.length < 18) {
    return { T: cloneSE3(initT), rms: 0.05, inlierCount: 0 };
  }

  const tgtGrid = new SpatialHashGrid3D<WeightedPoint>(0.036);
  tgtGrid.insertAll(tgtCloud);

  let T = cloneSE3(initT);
  let lastRms = 0.05;
  let lastInliers = 0;

  interface Corr {
    ps: Vec3;
    pt: Vec3;
    nt: Vec3;
    e: number;
    absE: number;
    dist: number;
    w: number;
  }

  for (let iter = 0; iter < maxIters; iter++) {
    const corrs: Corr[] = [];
    const curMaxDist =
      iter < 4 ? maxDist : iter < 9 ? maxDist * 0.72 : Math.max(0.045, maxDist * 0.52);
    const curP2P = iter < 4 ? baseP2P : baseP2P * 0.25;

    for (let i = 0; i < srcCloud.length; i++) {
      const sp = srcCloud[i];
      const ps = transformPoint(T, sp.p);
      const ns = mat3MulVec(T.R, sp.n);

      const hit = tgtGrid.findNearest(ps, curMaxDist, (cand) => dot3(ns, cand.n) > 0.42);
      if (!hit) continue;

      const tp = hit.item;
      const e =
        tp.n[0] * (ps[0] - tp.p[0]) +
        tp.n[1] * (ps[1] - tp.p[1]) +
        tp.n[2] * (ps[2] - tp.p[2]);
      corrs.push({
        ps,
        pt: tp.p,
        nt: tp.n,
        e,
        absE: Math.abs(e),
        dist: Math.sqrt(hit.distSq),
        w: Math.sqrt(sp.w * tp.w),
      });
    }

    if (corrs.length < 15) break;

    // Trim to top `trimFraction` smallest point-to-plane residuals
    corrs.sort((a, b) => a.absE - b.absE);
    const keepCount = Math.max(12, Math.floor(corrs.length * trimFraction));
    lastInliers = keepCount;

    const H = new Float64Array(36);
    const g = new Float64Array(6);
    let sqErrSum = 0;

    for (let k = 0; k < keepCount; k++) {
      const c = corrs[k];
      sqErrSum += c.e * c.e;
      const wEff = c.w * huberWeight(c.e, huberDelta);

      // Left-perturbation on ps' = exp(dxi^) ps ≈ ps + dphi x ps + drho
      // Point-to-plane: e = nt^T (ps - pt) => de/dxi = [(ps x nt)^T, nt^T]
      const pxn = cross3(c.ps, c.nt);
      const J = [pxn[0], pxn[1], pxn[2], c.nt[0], c.nt[1], c.nt[2]];

      for (let r = 0; r < 6; r++) {
        const wJr = wEff * J[r];
        g[r] += wJr * c.e;
        for (let col = r; col < 6; col++) {
          H[r * 6 + col] += wJr * J[col];
        }
      }

      // Mild point-to-point regularization prevents tangent sliding along cylindrical Panda links:
      // ep = ps - pt, d(ep)/dxi = [-[ps]_x, I_3]
      if (curP2P > 1e-6) {
        const ex = c.ps[0] - c.pt[0];
        const ey = c.ps[1] - c.pt[1];
        const ez = c.ps[2] - c.pt[2];
        const wP = curP2P * c.w * huberWeight(c.dist, huberDelta * 2.5);
        const px = c.ps[0];
        const py = c.ps[1];
        const pz = c.ps[2];
        // Rows of [-[ps]_x, I_3]:
        // row 0: [  0,  pz, -py, 1, 0, 0 ]
        // row 1: [-pz,   0,  px, 0, 1, 0 ]
        // row 2: [ py, -px,   0, 0, 0, 1 ]
        const rows: [number[], number][] = [
          [[0, pz, -py, 1, 0, 0], ex],
          [[-pz, 0, px, 0, 1, 0], ey],
          [[py, -px, 0, 0, 0, 1], ez],
        ];
        for (let axis = 0; axis < 3; axis++) {
          const Jr = rows[axis][0];
          const errAxis = rows[axis][1];
          for (let r = 0; r < 6; r++) {
            const jr = Jr[r];
            if (jr === 0) continue;
            const wJr = wP * jr;
            g[r] += wJr * errAxis;
            for (let col = r; col < 6; col++) {
              const jc = Jr[col];
              if (jc !== 0) {
                H[r * 6 + col] += wJr * jc;
              }
            }
          }
        }
      }
    }

    lastRms = Math.sqrt(sqErrSum / keepCount);

    for (let r = 0; r < 6; r++) {
      for (let col = 0; col < r; col++) {
        H[r * 6 + col] = H[col * 6 + r];
      }
      H[r * 6 + r] += 2e-4 + 1e-3 * H[r * 6 + r];
    }

    const negG = new Float64Array(6);
    for (let r = 0; r < 6; r++) negG[r] = -g[r];

    const step = solveLinearSystem(H, negG, 6);
    if (!step) break;

    let dxi: Vec6 = [step[0], step[1], step[2], step[3], step[4], step[5]];
    const rotStep = Math.hypot(dxi[0], dxi[1], dxi[2]);
    const transStep = Math.hypot(dxi[3], dxi[4], dxi[5]);
    if (!Number.isFinite(rotStep) || !Number.isFinite(transStep)) break;

    // Trust-region step clamping prevents single-iteration divergence on thin links
    let scale = 1.0;
    if (rotStep > maxStepRot) scale = Math.min(scale, maxStepRot / rotStep);
    if (transStep > maxStepTrans) scale = Math.min(scale, maxStepTrans / transStep);
    if (scale < 1.0) {
      dxi = [
        dxi[0] * scale,
        dxi[1] * scale,
        dxi[2] * scale,
        dxi[3] * scale,
        dxi[4] * scale,
        dxi[5] * scale,
      ];
    }

    T = composeSE3(expSE3(dxi), T);
    if (rotStep * scale < 2e-5 && transStep * scale < 2e-5) break;
  }

  return { T, rms: lastRms, inlierCount: lastInliers };
}

/**
 * Estimate a single action's camera-frame rigid motion dT_k^c and unit screw axis xi_k^c:
 *  1. Accumulate frame-to-frame small-step Trimmed Point-to-Plane ICP across the active sweep
 *     with constant-velocity initialization from the previous step's SE(3) twist
 *  2. Refine with direct start-to-end keyframe ICP initialized from accumulated dT to eliminate drift
 *  3. Extract xi_k^c via exact logSE3(dT_k^c), normalized by theta_obs with sign from deltaQ
 *  4. Compute quality weight alpha_k combining angle consistency, zero-pitch constraint, and ICP RMS
 */
export function extractScrewObservationFromSweep(
  action: MotionAction,
  sweepClouds: readonly (readonly WeightedPoint[])[],
  startIdx: number,
  endIdx: number,
  xiBase: Vec6,
  deltaQ: number,
  frameQEnc?: readonly number[]
): ScrewObservation | null {
  if (endIdx <= startIdx || endIdx >= sweepClouds.length) return null;

  let dTAccum = identitySE3();
  let prevStepTwist: Vec6 | null = null;
  let prevDqStep = 0;
  let accumValidSteps = 0;
  let accumRmsSum = 0;

  for (let t = startIdx; t < endIdx; t++) {
    const c0 = sweepClouds[t];
    const c1 = sweepClouds[t + 1];
    if (c0.length < 20 || c1.length < 20) continue;

    // Known encoder step magnitude (real sensor data, zero GT leakage) for velocity prediction & sanity bound
    const dqStepEnc =
      frameQEnc && frameQEnc.length > t + 1
        ? frameQEnc[t + 1] - frameQEnc[t]
        : deltaQ / Math.max(1, endIdx - startIdx);
    const absDqStep = Math.abs(dqStepEnc);

    let initStepT = identitySE3();
    if (prevStepTwist && Math.abs(prevDqStep) > 0.008 && absDqStep > 0.004) {
      const ratio = Math.max(0.25, Math.min(2.2, dqStepEnc / prevDqStep));
      const predTwist: Vec6 = [
        prevStepTwist[0] * ratio,
        prevStepTwist[1] * ratio,
        prevStepTwist[2] * ratio,
        prevStepTwist[3] * ratio,
        prevStepTwist[4] * ratio,
        prevStepTwist[5] * ratio,
      ];
      initStepT = expSE3(predTwist);
    }

    let stepRes = runTrimmedPointToPlaneICP(c0, c1, initStepT, {
      maxIters: 14,
      maxDist: 0.125,
      trimFraction: 0.78,
      huberDelta: 0.011,
      p2pWeight: 0.12,
      maxStepRot: 0.10,
      maxStepTrans: 0.06,
    });

    let tw = logSE3(stepRes.T);
    let stepTheta = norm3([tw[0], tw[1], tw[2]]);
    const maxPlausibleTheta = Math.max(0.16, 2.3 * absDqStep + 0.055);

    // If a step jumped beyond physical joint speed bound, retry with stronger p2p anchor
    if (stepTheta > maxPlausibleTheta || stepRes.inlierCount < 20) {
      const retryRes = runTrimmedPointToPlaneICP(c0, c1, initStepT, {
        maxIters: 16,
        maxDist: 0.16,
        trimFraction: 0.78,
        huberDelta: 0.015,
        p2pWeight: 0.28,
        maxStepRot: 0.08,
        maxStepTrans: 0.05,
      });
      const retryTw = logSE3(retryRes.T);
      const retryTheta = norm3([retryTw[0], retryTw[1], retryTw[2]]);
      if (retryRes.inlierCount >= 18 && retryTheta <= maxPlausibleTheta * 1.15) {
        stepRes = retryRes;
        tw = retryTw;
        stepTheta = retryTheta;
      }
    }

    if (stepRes.inlierCount >= 18 && stepTheta <= maxPlausibleTheta * 1.35) {
      dTAccum = composeSE3(stepRes.T, dTAccum);
      prevStepTwist = tw;
      prevDqStep = dqStepEnc;
      accumValidSteps++;
      accumRmsSum += stepRes.rms;
    } else if (prevStepTwist && Math.abs(prevDqStep) > 0.008) {
      // Fallback to predicted constant-velocity increment if a single frame is severely occluded
      dTAccum = composeSE3(initStepT, dTAccum);
      accumValidSteps++;
    }
  }

  if (accumValidSteps === 0) return null;

  const absDq = Math.max(Math.abs(deltaQ), 0.1);
  const twAccum = logSE3(dTAccum);
  const thetaAccum = norm3([twAccum[0], twAccum[1], twAccum[2]]);

  // Direct start-to-end keyframe refinement initialized by accumulated frame-to-frame motion
  const startCloud = sweepClouds[startIdx];
  const endCloud = sweepClouds[endIdx];
  let dTFinal = dTAccum;
  let finalRms = accumValidSteps > 0 ? accumRmsSum / accumValidSteps : 0.015;

  if (startCloud.length >= 25 && endCloud.length >= 25) {
    const directRes = runTrimmedPointToPlaneICP(startCloud, endCloud, dTAccum, {
      maxIters: 22,
      maxDist: 0.095,
      trimFraction: 0.78,
      huberDelta: 0.010,
      p2pWeight: 0.04,
      maxStepRot: 0.08,
      maxStepTrans: 0.05,
    });
    const minNeededInliers = Math.max(28, Math.floor(0.24 * Math.min(startCloud.length, endCloud.length)));
    if (directRes.inlierCount >= minNeededInliers) {
      const twDir = logSE3(directRes.T);
      const thetaDir = norm3([twDir[0], twDir[1], twDir[2]]);
      // Only accept direct keyframe update if it stays consistent with joint encoder amplitude
      const dirErr = Math.abs(thetaDir - absDq);
      const accErr = Math.abs(thetaAccum - absDq);
      if (dirErr <= Math.max(0.22, accErr + 0.08) && directRes.rms < 0.025) {
        dTFinal = directRes.T;
        finalRms = directRes.rms;
      }
    }
  }

  // Exact se(3) log map: [dOmega, dV] = deltaQ * [omega_c, v_c]
  const twist = logSE3(dTFinal);
  const dOmega: Vec3 = [twist[0], twist[1], twist[2]];
  const dV: Vec3 = [twist[3], twist[4], twist[5]];
  const thetaObs = norm3(dOmega);
  if (thetaObs < 0.06) return null;

  const sign = deltaQ >= 0 ? 1 : -1;
  const invScale = sign / thetaObs;
  const omegaCam: Vec3 = [dOmega[0] * invScale, dOmega[1] * invScale, dOmega[2] * invScale];
  const vCamRaw: Vec3 = [dV[0] * invScale, dV[1] * invScale, dV[2] * invScale];

  // Revolute joint pitch h = omega^T v (theoretically 0); project out pitch noise for clean Plucker line
  const pitch = dot3(omegaCam, vCamRaw);
  const vCam: Vec3 = [
    vCamRaw[0] - pitch * omegaCam[0],
    vCamRaw[1] - pitch * omegaCam[1],
    vCamRaw[2] - pitch * omegaCam[2],
  ];

  // Quality weight alpha_k: angle consistency + zero-pitch constraint + ICP RMS
  const relThetaErr = Math.abs(thetaObs - absDq) / absDq;
  const wTheta = Math.exp(-(relThetaErr * relThetaErr) / (2 * 0.16 * 0.16));
  const wPitch = Math.exp(-(pitch * pitch) / (2 * 0.08 * 0.08));
  const wRms = Math.exp(-(finalRms * finalRms) / (2 * 0.016 * 0.016));
  const weight = Math.max(0.02, wTheta * wPitch * wRms);

  return {
    action,
    xiCam: [omegaCam[0], omegaCam[1], omegaCam[2], vCam[0], vCam[1], vCam[2]],
    xiBase,
    deltaQ,
    thetaObs,
    pitch,
    icpRms: finalRms,
    weight,
    dTCam: dTFinal,
  };
}

/**
 * Closed-form 3D Screw-Axis Adjoint Hand-Eye Calibration (Stage 1 / Mode A)
 * with Consensus-Triplet + Huber IRLS robustness against corrupted/occluded sweeps:
 *  - Rotation via Wahba SVD:
 *      H = sum_k w_k * omega_k^b * (omega_k^c)^T = U * Sigma * V^T
 *      R_cb^(0) = V * diag(1, 1, det(V * U^T)) * U^T
 *  - Translation via Huber-weighted Plucker moment least squares:
 *      -[R_cb^(0) * omega_k^b]_x * t_cb = v_k^c - R_cb^(0) * v_k^b
 * Note: For exact/noiseless observations, residuals are 0 => Huber weights = 1.0,
 * yielding the exact closed-form Wahba + Plucker WLS solution to machine precision.
 */
export function solveScrewHandEyeClosedForm(
  observations: readonly { xiBase: Vec6; xiCam: Vec6; weight: number }[]
): SE3 {
  const N = observations.length;
  if (N < 2) {
    return identitySE3();
  }

  const wbs: Vec3[] = [];
  const wcs: Vec3[] = [];
  const vbs: Vec3[] = [];
  const vcs: Vec3[] = [];
  const baseWeights: number[] = [];

  for (let k = 0; k < N; k++) {
    const obs = observations[k];
    wbs.push(normalize3([obs.xiBase[0], obs.xiBase[1], obs.xiBase[2]]));
    wcs.push(normalize3([obs.xiCam[0], obs.xiCam[1], obs.xiCam[2]]));
    vbs.push([obs.xiBase[3], obs.xiBase[4], obs.xiBase[5]]);
    vcs.push([obs.xiCam[3], obs.xiCam[4], obs.xiCam[5]]);
    baseWeights.push(Math.max(1e-6, obs.weight));
  }

  const solveWahba = (indices: readonly number[], weights: readonly number[]): Mat3 => {
    const H: Mat3 = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    for (let idx = 0; idx < indices.length; idx++) {
      const k = indices[idx];
      const alpha = weights[idx];
      const wb = wbs[k];
      const wc = wcs[k];
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          H[r * 3 + c] += alpha * wb[r] * wc[c];
        }
      }
    }
    const { U, V } = svd3x3(H);
    const Ut = mat3Transpose(U);
    const VUt = mat3Mul(V, Ut);
    const detSign = mat3Det(VUt) < 0 ? -1 : 1;
    const D: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, detSign];
    return mat3Mul(mat3Mul(V, D), Ut);
  };

  const scoreRotHuber = (R: Mat3, delta = 0.08): number => {
    let loss = 0;
    for (let k = 0; k < N; k++) {
      const rwb = mat3MulVec(R, wbs[k]);
      const err = Math.hypot(
        rwb[0] - wcs[k][0],
        rwb[1] - wcs[k][1],
        rwb[2] - wcs[k][2]
      );
      const h = err <= delta ? 0.5 * err * err : delta * (err - 0.5 * delta);
      loss += baseWeights[k] * h;
    }
    return loss;
  };

  const allIdx = Array.from({ length: N }, (_, i) => i);
  let R_cb = solveWahba(allIdx, baseWeights);
  let bestScore = scoreRotHuber(R_cb);

  // Consensus triplet search (exhaustive for N <= 8, max 56 triplets, <0.2ms) rejects single corrupted sweeps
  if (N >= 4 && N <= 10) {
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const cIJ = cross3(wbs[i], wbs[j]);
        if (norm3(cIJ) < 0.15) continue;
        for (let m = j + 1; m < N; m++) {
          // Ensure the 3 base axes span 3D well before testing triplet
          const vol = Math.abs(dot3(cIJ, wbs[m]));
          if (vol < 0.10) continue;
          const R_cand = solveWahba([i, j, m], [baseWeights[i], baseWeights[j], baseWeights[m]]);
          const sc = scoreRotHuber(R_cand);
          if (sc < bestScore - 1e-12) {
            bestScore = sc;
            R_cb = R_cand;
          }
        }
      }
    }
  }

  // 3 iterations of Huber-reweighted Wahba SVD to refine R_cb using all inlier sweeps
  const rotWeights = [...baseWeights];
  if (N >= 4) {
    const deltaRot = 0.065;
    for (let iter = 0; iter < 3; iter++) {
      for (let k = 0; k < N; k++) {
        const rwb = mat3MulVec(R_cb, wbs[k]);
        const err = Math.hypot(
          rwb[0] - wcs[k][0],
          rwb[1] - wcs[k][1],
          rwb[2] - wcs[k][2]
        );
        const hw = huberWeight(err, deltaRot);
        // Stronger quadratic Tukey-like falloff for severe >20 deg axis outliers
        rotWeights[k] = baseWeights[k] * (err > 0.35 ? hw * 0.15 : hw);
      }
      R_cb = solveWahba(allIdx, rotWeights);
    }
  }

  // 2. Huber IRLS Plucker moment WLS for t_cb:
  //    A_k = -[R_cb * omega_k^b]_x,  b_k = v_k^c - R_cb * v_k^b
  const A_list: Mat3[] = [];
  const b_list: Vec3[] = [];
  for (let k = 0; k < N; k++) {
    const Rwb = mat3MulVec(R_cb, wbs[k]);
    const Rvb = mat3MulVec(R_cb, vbs[k]);
    const sk = skew(Rwb);
    A_list.push([
      -sk[0], -sk[1], -sk[2],
      -sk[3], -sk[4], -sk[5],
      -sk[6], -sk[7], -sk[8],
    ]);
    b_list.push([vcs[k][0] - Rvb[0], vcs[k][1] - Rvb[1], vcs[k][2] - Rvb[2]]);
  }

  const solveTransWLS = (weights: readonly number[]): Vec3 => {
    const AtWA = new Float64Array(9);
    const AtWb = new Float64Array(3);
    for (let k = 0; k < N; k++) {
      const alpha = weights[k];
      const A = A_list[k];
      const b = b_list[k];
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          let s = 0;
          for (let m = 0; m < 3; m++) {
            s += A[m * 3 + r] * A[m * 3 + c];
          }
          AtWA[r * 3 + c] += alpha * s;
        }
        let sb = 0;
        for (let m = 0; m < 3; m++) {
          sb += A[m * 3 + r] * b[m];
        }
        AtWb[r] += alpha * sb;
      }
    }
    let tSol = solveLinearSystem(AtWA, AtWb, 3);
    if (!tSol) {
      AtWA[0] += 1e-8;
      AtWA[4] += 1e-8;
      AtWA[8] += 1e-8;
      tSol = solveLinearSystem(AtWA, AtWb, 3);
    }
    return tSol ? [tSol[0], tSol[1], tSol[2]] : [0, 0, 1.2];
  };

  let t_cb = solveTransWLS(rotWeights);
  if (N >= 4) {
    const deltaMom = 0.045;
    for (let iter = 0; iter < 4; iter++) {
      const transWeights = new Array<number>(N);
      for (let k = 0; k < N; k++) {
        const At = mat3MulVec(A_list[k], t_cb);
        const err = Math.hypot(
          At[0] - b_list[k][0],
          At[1] - b_list[k][1],
          At[2] - b_list[k][2]
        );
        transWeights[k] = rotWeights[k] * huberWeight(err, deltaMom);
      }
      t_cb = solveTransWLS(transWeights);
    }
  }

  return { R: R_cb, t: t_cb };
}

/**
 * Pure 3D Translation Trimmed Point-to-Plane ICP (R = I_3, optimizes t in R^3).
 * Used for Delta parallel robot closed-loop platform sweeps: trimming keeps the
 * dominant rigid translating cluster (moving platform + wrist) while automatically
 * rejecting the rotating upper arm and tilting carbon rods!
 */
export function runTrimmedTranslationICP(
  srcCloud: readonly WeightedPoint[],
  tgtCloud: readonly WeightedPoint[],
  initT: Vec3 = [0, 0, 0],
  opts?: {
    maxIters?: number;
    maxDist?: number;
    trimFraction?: number;
    huberDelta?: number;
    p2pWeight?: number;
  }
): { dt: Vec3; rms: number; inlierCount: number; inlierCentroidSrc: Vec3 } {
  const maxIters = opts?.maxIters ?? 16;
  const maxDist = opts?.maxDist ?? 0.14;
  const trimFraction = opts?.trimFraction ?? 0.68;
  const huberDelta = opts?.huberDelta ?? 0.012;
  const baseP2P = opts?.p2pWeight ?? 0.18;

  if (srcCloud.length < 15 || tgtCloud.length < 15) {
    return { dt: [...initT], rms: 0.05, inlierCount: 0, inlierCentroidSrc: [0, 0, 1.2] };
  }

  const tgtGrid = new SpatialHashGrid3D<WeightedPoint>(0.036);
  tgtGrid.insertAll(tgtCloud);

  const dt: Vec3 = [initT[0], initT[1], initT[2]];
  let lastRms = 0.05;
  let lastInliers = 0;
  let inlierCentroidSrc: Vec3 = [0, 0, 1.2];

  interface Corr3 {
    sp: Vec3;
    ps: Vec3;
    pt: Vec3;
    nt: Vec3;
    e: number;
    absE: number;
    dist: number;
    w: number;
  }

  for (let iter = 0; iter < maxIters; iter++) {
    const corrs: Corr3[] = [];
    const curMaxDist = iter < 4 ? maxDist : iter < 9 ? maxDist * 0.72 : Math.max(0.045, maxDist * 0.52);
    const curP2P = iter < 5 ? baseP2P : baseP2P * 0.35;

    for (let i = 0; i < srcCloud.length; i++) {
      const sp = srcCloud[i];
      const ps: Vec3 = [sp.p[0] + dt[0], sp.p[1] + dt[1], sp.p[2] + dt[2]];
      const hit = tgtGrid.findNearest(ps, curMaxDist, (cand) => dot3(sp.n, cand.n) > 0.50);
      if (!hit) continue;
      const tp = hit.item;
      const e =
        tp.n[0] * (ps[0] - tp.p[0]) +
        tp.n[1] * (ps[1] - tp.p[1]) +
        tp.n[2] * (ps[2] - tp.p[2]);
      corrs.push({
        sp: sp.p,
        ps,
        pt: tp.p,
        nt: tp.n,
        e,
        absE: Math.abs(e),
        dist: Math.sqrt(hit.distSq),
        w: Math.sqrt(sp.w * tp.w),
      });
    }

    if (corrs.length < 12) break;
    corrs.sort((a, b) => a.absE - b.absE);
    const keepCount = Math.max(10, Math.floor(corrs.length * trimFraction));
    lastInliers = keepCount;

    const H = new Float64Array(9);
    const g = new Float64Array(3);
    let sqErr = 0;
    let cx = 0, cy = 0, cz = 0, wSum = 0;

    for (let k = 0; k < keepCount; k++) {
      const c = corrs[k];
      sqErr += c.e * c.e;
      const wEff = c.w * huberWeight(c.e, huberDelta);
      cx += wEff * c.sp[0];
      cy += wEff * c.sp[1];
      cz += wEff * c.sp[2];
      wSum += wEff;

      for (let r = 0; r < 3; r++) {
        const wJr = wEff * c.nt[r];
        g[r] += wJr * c.e;
        for (let col = r; col < 3; col++) {
          H[r * 3 + col] += wJr * c.nt[col];
        }
      }
      if (curP2P > 1e-6) {
        const wP = curP2P * c.w * huberWeight(c.dist, huberDelta * 2.5);
        for (let r = 0; r < 3; r++) {
          H[r * 3 + r] += wP;
          g[r] += wP * (c.ps[r] - c.pt[r]);
        }
      }
    }

    lastRms = Math.sqrt(sqErr / keepCount);
    if (wSum > 1e-9) {
      inlierCentroidSrc = [cx / wSum, cy / wSum, cz / wSum];
    }

    for (let r = 0; r < 3; r++) {
      for (let col = 0; col < r; col++) H[r * 3 + col] = H[col * 3 + r];
      H[r * 3 + r] += 2e-4 + 1e-3 * H[r * 3 + r];
    }

    const negG = new Float64Array([-g[0], -g[1], -g[2]]);
    const step = solveLinearSystem(H, negG, 3);
    if (!step) break;
    const stepNorm = Math.hypot(step[0], step[1], step[2]);
    if (!Number.isFinite(stepNorm)) break;
    const scale = stepNorm > 0.06 ? 0.06 / stepNorm : 1.0;
    dt[0] += step[0] * scale;
    dt[1] += step[1] * scale;
    dt[2] += step[2] * scale;
    if (stepNorm * scale < 2e-5) break;
  }

  return { dt, rms: lastRms, inlierCount: lastInliers, inlierCentroidSrc };
}

/**
 * Extract a closed-loop pure-translation observation for Delta parallel motors 0..2:
 *  - Filters active cloud to the lower translating platform region (keeping lower 72% by camera +Y)
 *    and runs frame-to-frame + direct keyframe pure-translation Trimmed ICP
 *  - Returns deltaPCam (3D platform displacement in {C}) and mid-sweep centroidCam in {C}.
 */
export function extractParallelTranslationObservationFromSweep(
  action: MotionAction,
  sweepClouds: readonly (readonly WeightedPoint[])[],
  startIdx: number,
  endIdx: number,
  deltaPBase: Vec3,
  platformMidBase: Vec3,
  deltaQ: number
): ScrewObservation | null {
  if (endIdx <= startIdx || endIdx >= sweepClouds.length) return null;
  const distBase = norm3(deltaPBase);
  if (distBase < 0.025) return null;

  // On Delta overhead gantry, the translating platform + wrist hangs below the rotating upper arm
  // (higher camera +Y in CV frame). Keep lower 70% of points to isolate platform cleanly.
  const filterPlatformRegion = (cloud: readonly WeightedPoint[]): WeightedPoint[] => {
    if (cloud.length < 35) return [...cloud];
    const sortedV = cloud.map((pt) => pt.v).sort((a, b) => a - b);
    const vCut = sortedV[Math.floor(sortedV.length * 0.28)];
    const lower = cloud.filter((pt) => pt.v >= vCut);
    return lower.length >= 24 ? lower : [...cloud];
  };

  const platClouds = sweepClouds.map((c) => (c.length > 0 ? filterPlatformRegion(c) : []));

  const dtAccum: Vec3 = [0, 0, 0];
  let prevStep: Vec3 = [0, 0, 0];
  let validSteps = 0;
  let rmsSum = 0;

  for (let t = startIdx; t < endIdx; t++) {
    const c0 = platClouds[t];
    const c1 = platClouds[t + 1];
    if (c0.length < 18 || c1.length < 18) continue;

    const res = runTrimmedTranslationICP(c0, c1, prevStep, {
      maxIters: 14,
      maxDist: 0.11,
      trimFraction: 0.68,
      huberDelta: 0.012,
      p2pWeight: 0.22,
    });
    const stepLen = norm3(res.dt);
    if (res.inlierCount >= 16 && stepLen < 0.08) {
      dtAccum[0] += res.dt[0];
      dtAccum[1] += res.dt[1];
      dtAccum[2] += res.dt[2];
      prevStep = res.dt;
      rmsSum += res.rms;
      validSteps++;
    }
  }

  if (validSteps === 0) return null;

  let dtFinal: Vec3 = [...dtAccum];
  let finalRms = rmsSum / validSteps;
  let startCentroid: Vec3 = [0, 0, 1.2];

  const startCloud = platClouds[startIdx];
  const endCloud = platClouds[endIdx];
  if (startCloud.length >= 20 && endCloud.length >= 20) {
    const directRes = runTrimmedTranslationICP(startCloud, endCloud, dtAccum, {
      maxIters: 20,
      maxDist: 0.095,
      trimFraction: 0.66,
      huberDelta: 0.010,
      p2pWeight: 0.14,
    });
    startCentroid = directRes.inlierCentroidSrc;
    const directLen = norm3(directRes.dt);
    const accumLen = norm3(dtAccum);
    if (
      directRes.inlierCount >= 18 &&
      Math.abs(directLen - distBase) <= Math.max(0.045, Math.abs(accumLen - distBase) + 0.02)
    ) {
      dtFinal = directRes.dt;
      finalRms = directRes.rms;
    }
  }

  const distCam = norm3(dtFinal);
  if (distCam < 0.02) return null;

  // Mid-sweep centroid in camera frame {C} corresponds to platformMidBase in {B}
  const centroidCam: Vec3 = [
    startCentroid[0] + 0.5 * dtFinal[0],
    startCentroid[1] + 0.5 * dtFinal[1],
    startCentroid[2] + 0.5 * dtFinal[2],
  ];

  // Platform center + wrist visual mass sits slightly below delta_platform origin (~-0.025m in Z)
  const centroidBase: Vec3 = [
    platformMidBase[0],
    platformMidBase[1],
    platformMidBase[2] - 0.022,
  ];

  const relLenErr = Math.abs(distCam - distBase) / Math.max(distBase, 0.04);
  const wLen = Math.exp(-(relLenErr * relLenErr) / (2 * 0.18 * 0.18));
  const wRms = Math.exp(-(finalRms * finalRms) / (2 * 0.018 * 0.018));
  const weight = Math.max(0.03, wLen * wRms);

  const uBase = normalize3(deltaPBase);
  const uCam = normalize3(dtFinal);

  return {
    action,
    kind: 'translation',
    xiBase: [0, 0, 0, uBase[0], uBase[1], uBase[2]],
    xiCam: [0, 0, 0, uCam[0], uCam[1], uCam[2]],
    deltaQ,
    thetaObs: distCam,
    pitch: 0,
    icpRms: finalRms,
    weight,
    dTCam: {
      R: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      t: dtFinal,
    },
    deltaPBase,
    deltaPCam: dtFinal,
    centroidBase,
    centroidCam,
  };
}

/**
 * Closed-form Stage 1 Hand-Eye Calibration for 4-DoF Delta Parallel Robot:
 *  - Rotation R_cb^(0): Wahba SVD aligning unit 3D platform translation directions
 *    (deltaP_k^b / ||deltaP_k^b|| -> deltaP_k^c / ||deltaP_k^c||) PLUS any revolute wrist axes!
 *  - Translation t_cb^(0): Weighted joint least-squares combining tracked platform centroids
 *    (c_k^c - R_cb * c_k^b) with any rotary wrist Plucker moment constraints!
 */
export function solveParallelHandEyeClosedForm(
  observations: readonly ScrewObservation[]
): SE3 {
  if (observations.length < 2) {
    return identitySE3();
  }

  // 1. Wahba SVD for R_cb^(0) from unit displacement vectors + optional wrist screw axes
  const H: Mat3 = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let k = 0; k < observations.length; k++) {
    const obs = observations[k];
    const alpha = Math.max(1e-6, obs.weight);
    let vb: Vec3;
    let vc: Vec3;
    if (obs.kind === 'translation' && obs.deltaPBase && obs.deltaPCam) {
      vb = normalize3(obs.deltaPBase);
      vc = normalize3(obs.deltaPCam);
    } else {
      vb = normalize3([obs.xiBase[0], obs.xiBase[1], obs.xiBase[2]]);
      vc = normalize3([obs.xiCam[0], obs.xiCam[1], obs.xiCam[2]]);
    }
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        H[r * 3 + c] += alpha * vb[r] * vc[c];
      }
    }
  }

  const { U, V } = svd3x3(H);
  const Ut = mat3Transpose(U);
  const VUt = mat3Mul(V, Ut);
  const detSign = mat3Det(VUt) < 0 ? -1 : 1;
  const D: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, detSign];
  const R_cb = mat3Mul(mat3Mul(V, D), Ut);

  // 2. Joint WLS for t_cb^(0):
  //    - Pure translation sweeps: t_cb ≈ centroidCam - R_cb * centroidBase (with front-surface offset compensation)
  //    - Rotary wrist sweeps (if present): -[R_cb * omega_b]_x * t_cb = v_c - R_cb * v_b
  const AtWA = new Float64Array(9);
  const AtWb = new Float64Array(3);

  for (let k = 0; k < observations.length; k++) {
    const obs = observations[k];
    const alpha = Math.max(1e-6, obs.weight);

    if (obs.kind === 'translation' && obs.centroidBase && obs.centroidCam) {
      const rcb = mat3MulVec(R_cb, obs.centroidBase);
      // Camera viewing ray points along normalize(centroidCam); front surface of r~0.06m hub
      // sits ~0.04m closer to camera than center -> shift centroidCam back along ray by 0.038m
      const ray = normalize3(obs.centroidCam);
      const tEst: Vec3 = [
        obs.centroidCam[0] + 0.038 * ray[0] - rcb[0],
        obs.centroidCam[1] + 0.038 * ray[1] - rcb[1],
        obs.centroidCam[2] + 0.038 * ray[2] - rcb[2],
      ];
      for (let d = 0; d < 3; d++) {
        AtWA[d * 3 + d] += alpha;
        AtWb[d] += alpha * tEst[d];
      }
    } else if (obs.kind !== 'translation') {
      const wb: Vec3 = [obs.xiBase[0], obs.xiBase[1], obs.xiBase[2]];
      const vb: Vec3 = [obs.xiBase[3], obs.xiBase[4], obs.xiBase[5]];
      const vc: Vec3 = [obs.xiCam[3], obs.xiCam[4], obs.xiCam[5]];
      const Rwb = mat3MulVec(R_cb, wb);
      const Rvb = mat3MulVec(R_cb, vb);
      const sk = skew(Rwb);
      const A: Mat3 = [
        -sk[0], -sk[1], -sk[2],
        -sk[3], -sk[4], -sk[5],
        -sk[6], -sk[7], -sk[8],
      ];
      const b: Vec3 = [vc[0] - Rvb[0], vc[1] - Rvb[1], vc[2] - Rvb[2]];
      const wWrist = alpha * 1.4;
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          let s = 0;
          for (let m = 0; m < 3; m++) s += A[m * 3 + r] * A[m * 3 + c];
          AtWA[r * 3 + c] += wWrist * s;
        }
        let sb = 0;
        for (let m = 0; m < 3; m++) sb += A[m * 3 + r] * b[m];
        AtWb[r] += wWrist * sb;
      }
    }
  }

  let tSol = solveLinearSystem(AtWA, AtWb, 3);
  if (!tSol) {
    AtWA[0] += 1e-6;
    AtWA[4] += 1e-6;
    AtWA[8] += 1e-6;
    tSol = solveLinearSystem(AtWA, AtWb, 3);
  }
  const t_cb: Vec3 = tSol ? [tSol[0], tSol[1], tSol[2]] : [0, 0, 1.2];
  return { R: R_cb, t: t_cb };
}
