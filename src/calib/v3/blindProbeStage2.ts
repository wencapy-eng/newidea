import {
  composeSE3,
  cross3,
  dot3,
  expSE3,
  identitySE3,
  logSE3,
  mat3MulVec,
  norm3,
  normalize3,
  solveLinearSystem,
  SpatialHashGrid3D,
  svd3x3,
  transformPoint,
  type Mat3,
  type SE3,
  type Vec3,
  type Vec6,
} from '../../math/se3';
import {
  solveParallelHandEyeClosedForm,
  solveScrewHandEyeClosedForm,
} from '../screwStage1';
import type {
  MotionAction,
  ScrewObservation,
  V3ProbeLogEntry,
  WeightedPoint,
} from '../types';

export const V3_PROBE_MIN_POINTS = 55;
export const V3_PROBE_MAX_ANGLE_ERR_DEG = 3.5;
export const V3_PROBE_MAX_TRANS_ERR_MM = 8.0;
export const V3_PROBE_MIN_SIGMA_SIN = Math.sin((25 * Math.PI) / 180); // ~0.4226

/**
 * Purify active moving sub-chain points by removing stationary upstream points
 * whose 3D nearest-neighbor distance between start and end clouds is < staticThreshM (10 mm).
 */
export function purifyActiveSubchain(
  srcPts: readonly WeightedPoint[],
  dstPts: readonly WeightedPoint[],
  staticThreshM = 0.010
): { activeSrc: WeightedPoint[]; activeDst: WeightedPoint[] } {
  if (srcPts.length < 40 || dstPts.length < 40) {
    return { activeSrc: [...srcPts], activeDst: [...dstPts] };
  }

  const dstGrid = new SpatialHashGrid3D<WeightedPoint>(0.025);
  dstGrid.insertAll(dstPts);
  const srcGrid = new SpatialHashGrid3D<WeightedPoint>(0.025);
  srcGrid.insertAll(srcPts);

  const activeSrc = srcPts.filter((pt) => dstGrid.findNearest(pt.p, staticThreshM) === null);
  const activeDst = dstPts.filter((pt) => srcGrid.findNearest(pt.p, staticThreshM) === null);

  if (activeSrc.length >= 45 && activeDst.length >= 45) {
    return { activeSrc, activeDst };
  }
  return { activeSrc: [...srcPts], activeDst: [...dstPts] };
}

/**
 * Graduated Non-Convexity (GNC) Welsch Point-to-Plane ICP on SE(3):
 *   w(r; mu) = exp(-r^2 / (2 * mu^2))
 * Annealing mu: 0.060 -> 0.028 -> 0.013 m.
 * Because the Welsch kernel is redescending (w -> 0 for |r| >> mu), Class III non-rigid
 * flexible cable points are completely silenced while Class I + Class II rigid attachment
 * points jointly lock the exact rigid motion dT_cam.
 */
export function runGncWelschPointToPlaneICP(
  srcPts: readonly WeightedPoint[],
  dstPts: readonly WeightedPoint[],
  initT: SE3 = identitySE3(),
  opts?: {
    muSchedule?: readonly number[];
    itersPerStage?: number;
    damping?: number;
  }
): { T: SE3; rms: number; inlierRatio: number; inlierCount: number } {
  const muSchedule = opts?.muSchedule ?? [0.060, 0.028, 0.013];
  const itersPerStage = opts?.itersPerStage ?? 8;
  const damping = opts?.damping ?? 1e-4;

  if (srcPts.length < 18 || dstPts.length < 18) {
    return { T: initT, rms: 1.0, inlierRatio: 0, inlierCount: 0 };
  }

  const dstGrid = new SpatialHashGrid3D<WeightedPoint>(0.045);
  dstGrid.insertAll(dstPts);

  let T: SE3 = {
    R: [...initT.R] as Mat3,
    t: [...initT.t] as Vec3,
  };

  // Check initial overlap; if sparse (e.g., standalone large-displacement call without chained init),
  // warm-start translation via robust coordinate-wise median centroid shift (immune to 25%+ outliers)
  const probeCount = Math.min(40, srcPts.length);
  let initHits = 0;
  for (let i = 0; i < probeCount; i++) {
    const pT = transformPoint(T, srcPts[Math.floor((i * srcPts.length) / probeCount)].p);
    if (dstGrid.findNearest(pT, 0.14)) initHits++;
  }
  if (initHits < probeCount * 0.25) {
    const medianAxis = (pts: readonly WeightedPoint[], ax: 0 | 1 | 2) => {
      const vals = pts.map((pt) => pt.p[ax]).sort((a, b) => a - b);
      return vals[Math.floor(vals.length / 2)] ?? 0;
    };
    const cSrc: Vec3 = [medianAxis(srcPts, 0), medianAxis(srcPts, 1), medianAxis(srcPts, 2)];
    const cDst: Vec3 = [medianAxis(dstPts, 0), medianAxis(dstPts, 1), medianAxis(dstPts, 2)];
    const RcSrc = mat3MulVec(T.R, cSrc);
    T = {
      R: T.R,
      t: [cDst[0] - RcSrc[0], cDst[1] - RcSrc[1], cDst[2] - RcSrc[2]],
    };
  }

  let lastRms = 1.0;
  let lastInliers = 0;

  for (let s = 0; s < muSchedule.length; s++) {
    const mu = muSchedule[s];
    const maxDist = Math.max(0.055, mu * 3.0);
    const invTwoMuSq = 1 / (2 * mu * mu);

    for (let it = 0; it < itersPerStage; it++) {
      const H = new Float64Array(36);
      const g = new Float64Array(6);
      let sumW = 0;
      let sumSq = 0;
      let inliers = 0;

      for (let i = 0; i < srcPts.length; i++) {
        const sp = srcPts[i];
        const pT = transformPoint(T, sp.p);
        const nT = mat3MulVec(T.R, sp.n);

        const match = dstGrid.findNearest(pT, maxDist, (cand) => dot3(nT, cand.n) > 0.55);
        if (!match) continue;

        const q = match.item;
        const diff: Vec3 = [pT[0] - q.p[0], pT[1] - q.p[1], pT[2] - q.p[2]];
        const rPlane = dot3(q.n, diff);
        const dist3D = Math.sqrt(match.distSq);

        // Redescending Welsch kernel on combined point-to-plane & 3D consistency residual
        const wWelsch = Math.exp(-(rPlane * rPlane + 0.18 * dist3D * dist3D) * invTwoMuSq);
        if (wWelsch < 0.02) continue;

        // Normalized optical weight blended with Welsch kernel
        const wOpt = Math.min(4.0, Math.max(0.15, Math.sqrt(sp.w * q.w) * 1e-3));
        const wEff = wWelsch * wOpt;

        const pxn = cross3(pT, q.n);
        // Point-to-plane residual r = n^T (exp(dxi) pT - q) => dr/dxi = [(pT x n)^T, n^T]
        const J: Vec6 = [pxn[0], pxn[1], pxn[2], q.n[0], q.n[1], q.n[2]];

        for (let r = 0; r < 6; r++) {
          const wJr = wEff * J[r];
          g[r] -= wJr * rPlane;
          for (let c = r; c < 6; c++) {
            H[r * 6 + c] += wJr * J[c];
          }
        }

        // Mild isotropic point-to-point stabilizer in early stage to prevent sliding along cylinders
        if (s === 0) {
          const wPt = wEff * 0.06;
          for (let axis = 0; axis < 3; axis++) {
            const eVec: Vec3 = [0, 0, 0];
            eVec[axis] = 1;
            const pxe = cross3(pT, eVec);
            const Jp: Vec6 = [pxe[0], pxe[1], pxe[2], eVec[0], eVec[1], eVec[2]];
            const rp = diff[axis];
            for (let r = 0; r < 6; r++) {
              const wJr = wPt * Jp[r];
              g[r] -= wJr * rp;
              for (let c = r; c < 6; c++) {
                H[r * 6 + c] += wJr * Jp[c];
              }
            }
          }
        }

        sumW += wWelsch;
        sumSq += wWelsch * rPlane * rPlane;
        if (wWelsch > 0.25) inliers++;
      }

      if (inliers < 14) break;
      lastRms = Math.sqrt(sumSq / Math.max(sumW, 1e-9));
      lastInliers = inliers;

      for (let r = 0; r < 6; r++) {
        H[r * 6 + r] += damping * Math.max(1.0, sumW * 0.01);
        for (let c = 0; c < r; c++) {
          H[r * 6 + c] = H[c * 6 + r];
        }
      }

      const dxi = solveLinearSystem(H, g, 6);
      if (!dxi) break;

      const stepRot = Math.hypot(dxi[0], dxi[1], dxi[2]);
      const stepTrans = Math.hypot(dxi[3], dxi[4], dxi[5]);
      // Clamp single-iteration step to maintain local linearization validity
      let scale = 1.0;
      if (stepRot > 0.28) scale = Math.min(scale, 0.28 / stepRot);
      if (stepTrans > 0.14) scale = Math.min(scale, 0.14 / stepTrans);

      const dT = expSE3([
        dxi[0] * scale,
        dxi[1] * scale,
        dxi[2] * scale,
        dxi[3] * scale,
        dxi[4] * scale,
        dxi[5] * scale,
      ]);
      T = composeSE3(dT, T);

      if (stepRot * scale < 1e-4 && stepTrans * scale < 1e-4) break;
    }
  }

  return {
    T,
    rms: lastRms,
    inlierRatio: lastInliers / Math.max(1, srcPts.length),
    inlierCount: lastInliers,
  };
}

/**
 * GNC-Welsch Pure-Translation ICP (R = I_3, 3-DoF translation) for Delta Parallel Robots.
 * Silences non-rigid flexible cables and rotary upper arms while locking onto the pure-translating platform.
 */
export function runGncWelschTranslationICP(
  srcPts: readonly WeightedPoint[],
  dstPts: readonly WeightedPoint[],
  initTrans: Vec3 = [0, 0, 0]
): { deltaT: Vec3; rms: number; inlierCount: number; inlierCentroidSrc: Vec3 } {
  const muSchedule = [0.065, 0.028, 0.013];
  let t: Vec3 = [initTrans[0], initTrans[1], initTrans[2]];
  let inlierCentroidSrc: Vec3 = [0, 0, 1.2];

  if (srcPts.length < 15 || dstPts.length < 15) {
    return { deltaT: t, rms: 1.0, inlierCount: 0, inlierCentroidSrc };
  }

  const dstGrid = new SpatialHashGrid3D<WeightedPoint>(0.045);
  dstGrid.insertAll(dstPts);
  let lastRms = 1.0;
  let lastInliers = 0;

  for (let s = 0; s < muSchedule.length; s++) {
    const mu = muSchedule[s];
    const maxDist = Math.max(0.06, mu * 3.0);
    const invTwoMuSq = 1 / (2 * mu * mu);

    for (let it = 0; it < 8; it++) {
      const H = new Float64Array(9);
      const g = new Float64Array(3);
      let sumW = 0;
      let sumSq = 0;
      let inliers = 0;
      let cx = 0;
      let cy = 0;
      let cz = 0;
      let cW = 0;

      for (let i = 0; i < srcPts.length; i++) {
        const sp = srcPts[i];
        const pT: Vec3 = [sp.p[0] + t[0], sp.p[1] + t[1], sp.p[2] + t[2]];
        const match = dstGrid.findNearest(pT, maxDist, (cand) => dot3(sp.n, cand.n) > 0.60);
        if (!match) continue;

        const q = match.item;
        const dx = pT[0] - q.p[0];
        const dy = pT[1] - q.p[1];
        const dz = pT[2] - q.p[2];
        const rPlane = q.n[0] * dx + q.n[1] * dy + q.n[2] * dz;
        const d3Sq = dx * dx + dy * dy + dz * dz;

        const wWelsch = Math.exp(-(rPlane * rPlane + 0.25 * d3Sq) * invTwoMuSq);
        if (wWelsch < 0.02) continue;

        // Point-to-plane + mild isotropic stabilization
        const wPt = wWelsch * 0.14;
        for (let r = 0; r < 3; r++) {
          g[r] -= wWelsch * q.n[r] * rPlane + wPt * (r === 0 ? dx : r === 1 ? dy : dz);
          for (let c = 0; c < 3; c++) {
            H[r * 3 + c] += wWelsch * q.n[r] * q.n[c] + (r === c ? wPt : 0);
          }
        }
        sumW += wWelsch;
        sumSq += wWelsch * rPlane * rPlane;
        if (wWelsch > 0.25) {
          inliers++;
          cx += wWelsch * sp.p[0];
          cy += wWelsch * sp.p[1];
          cz += wWelsch * sp.p[2];
          cW += wWelsch;
        }
      }

      if (inliers < 12) break;
      lastRms = Math.sqrt(sumSq / Math.max(sumW, 1e-9));
      lastInliers = inliers;
      if (cW > 1e-9) {
        inlierCentroidSrc = [cx / cW, cy / cW, cz / cW];
      }

      H[0] += 1e-4;
      H[4] += 1e-4;
      H[8] += 1e-4;
      const dt = solveLinearSystem(H, g, 3);
      if (!dt) break;

      const step = Math.hypot(dt[0], dt[1], dt[2]);
      const sc = step > 0.08 ? 0.08 / step : 1.0;
      t = [t[0] + dt[0] * sc, t[1] + dt[1] * sc, t[2] + dt[2] * sc];
      if (step * sc < 1e-4) break;
    }
  }

  return { deltaT: t, rms: lastRms, inlierCount: lastInliers, inlierCentroidSrc };
}

/**
 * Compute minimum singular value sigma_min([u_1, ..., u_k]) of unit direction vectors
 * to verify linear independence (Gate 3: sigma_min > sin(25 deg) ~ 0.4226).
 */
export function computeAxisIndependenceSigmaMin(unitAxes: readonly Vec3[]): number {
  const k = unitAxes.length;
  if (k === 0) return 1.0;
  if (k === 1) return norm3(unitAxes[0]);

  if (k === 2) {
    // For two unit vectors u1, u2: sigma_min = sqrt(1 - |u1 . u2|)
    const u1 = normalize3(unitAxes[0]);
    const u2 = normalize3(unitAxes[1]);
    const c = Math.abs(dot3(u1, u2));
    // Angle between vectors: sin(theta) = ||u1 x u2||; matches 2-axis independence check
    return Math.sqrt(Math.max(0, 1 - c * c));
  }

  // For k >= 3: smallest singular value of 3 x k matrix U = sqrt(lambda_min(U U^T))
  const M: Mat3 = [0, 0, 0, 0, 0, 0, 0, 0, 0];
  for (let i = 0; i < k; i++) {
    const u = normalize3(unitAxes[i]);
    M[0] += u[0] * u[0]; M[1] += u[0] * u[1]; M[2] += u[0] * u[2];
    M[3] += u[1] * u[0]; M[4] += u[1] * u[1]; M[5] += u[1] * u[2];
    M[6] += u[2] * u[0]; M[7] += u[2] * u[1]; M[8] += u[2] * u[2];
  }
  const { S } = svd3x3(M);
  // S are eigenvalues of U U^T (since M is symmetric PSD)
  return Math.sqrt(Math.max(0, Math.min(S[0], S[1], S[2])));
}

/**
 * Extract V3 Stage 2 serial revolute screw observation via:
 *  1. Active sub-chain purification (removes stationary upstream links);
 *  2. Chained micro-step GNC-Welsch ICP across the sweep to build an accurate initialization;
 *  3. Start-to-end keyframe GNC-Welsch refinement (silencing Class III flexible cable while
 *     keeping Class I + Class II rigid attachment!).
 */
export function extractV3ScrewObservationFromSweep(
  action: MotionAction,
  sweepClouds: readonly WeightedPoint[][],
  startIdx: number,
  endIdx: number,
  xiBase: Vec6,
  deltaQ: number
): { obs: ScrewObservation | null; pointCount: number; ampErrDeg: number } {
  const startRaw = sweepClouds[startIdx] ?? [];
  const endRaw = sweepClouds[endIdx] ?? [];
  if (startRaw.length < 30 || endRaw.length < 30 || Math.abs(deltaQ) < 0.08) {
    return { obs: null, pointCount: Math.min(startRaw.length, endRaw.length), ampErrDeg: 99 };
  }

  const { activeSrc, activeDst } = purifyActiveSubchain(startRaw, endRaw, 0.010);

  // Chained micro-step GNC-Welsch across intermediate frames (step = 2 frames)
  let T_chain = identitySE3();
  let prevCloud = startRaw;
  for (let f = startIdx + 2; f <= endIdx; f += 2) {
    const curCloud = sweepClouds[f];
    if (!curCloud || curCloud.length < 25) continue;
    const stepRes = runGncWelschPointToPlaneICP(prevCloud, curCloud, identitySE3(), {
      muSchedule: [0.055, 0.024],
      itersPerStage: 5,
    });
    T_chain = composeSE3(stepRes.T, T_chain);
    prevCloud = curCloud;
  }

  // Final start-to-end GNC-Welsch refinement on purified active sub-chain
  const refine = runGncWelschPointToPlaneICP(activeSrc, activeDst, T_chain, {
    muSchedule: [0.055, 0.025, 0.012],
    itersPerStage: 9,
  });

  const dT = refine.T;
  const xiTotal = logSE3(dT);
  const thetaObs = Math.hypot(xiTotal[0], xiTotal[1], xiTotal[2]);
  const ampErrDeg = (Math.abs(thetaObs - Math.abs(deltaQ)) * 180) / Math.PI;

  if (thetaObs < 0.04) {
    return { obs: null, pointCount: activeSrc.length, ampErrDeg };
  }

  // Unit screw twist signed by deltaQ: dT = exp(deltaQ * xi_cam)
  const invDq = 1 / deltaQ;
  const xiCam: Vec6 = [
    xiTotal[0] * invDq,
    xiTotal[1] * invDq,
    xiTotal[2] * invDq,
    xiTotal[3] * invDq,
    xiTotal[4] * invDq,
    xiTotal[5] * invDq,
  ];
  const omegaC: Vec3 = [xiCam[0], xiCam[1], xiCam[2]];
  const vC: Vec3 = [xiCam[3], xiCam[4], xiCam[5]];
  const pitch = dot3(omegaC, vC) / Math.max(dot3(omegaC, omegaC), 1e-8);

  const anglePenalty = Math.exp(-(ampErrDeg * ampErrDeg) / (2 * 4.5 * 4.5));
  const pitchPenalty = Math.exp(-(pitch * pitch) / (2 * 0.03 * 0.03));
  const rmsPenalty = 1 / (1 + (refine.rms / 0.008) ** 2);
  const weight = Math.max(0.05, anglePenalty * pitchPenalty * rmsPenalty * Math.min(1, activeSrc.length / 180));

  return {
    obs: {
      action,
      kind: 'screw',
      xiCam,
      xiBase,
      deltaQ,
      thetaObs,
      pitch,
      icpRms: refine.rms,
      weight,
      dTCam: dT,
    },
    pointCount: activeSrc.length,
    ampErrDeg,
  };
}

/**
 * Extract V3 Stage 2 Delta parallel pure-translation observation via chained GNC-Welsch translation ICP.
 * Isolates the translating lower platform region and computes the tracked platform centroid exclusively
 * from Welsch pure-translation inliers (automatically rejecting rotating upper arms and tilting rods).
 */
export function extractV3ParallelTranslationFromSweep(
  action: MotionAction,
  sweepClouds: readonly WeightedPoint[][],
  startIdx: number,
  endIdx: number,
  deltaPBase: Vec3,
  centroidBase: Vec3,
  deltaQ: number
): { obs: ScrewObservation | null; pointCount: number; ampErrMm: number } {
  const startRaw = sweepClouds[startIdx] ?? [];
  const endRaw = sweepClouds[endIdx] ?? [];
  const baseDist = norm3(deltaPBase);
  if (startRaw.length < 30 || endRaw.length < 30 || baseDist < 0.015) {
    return { obs: null, pointCount: Math.min(startRaw.length, endRaw.length), ampErrMm: 999 };
  }

  // On Delta overhead gantry, the translating platform + wrist hangs below the rotating upper arms
  // (higher camera +Y pixel coordinate v). Keep lower 70% to isolate the translating cluster cleanly.
  const filterPlatformRegion = (cloud: readonly WeightedPoint[]): WeightedPoint[] => {
    if (cloud.length < 35) return [...cloud];
    const sortedV = cloud.map((pt) => pt.v).sort((a, b) => a - b);
    const vCut = sortedV[Math.floor(sortedV.length * 0.28)];
    const lower = cloud.filter((pt) => pt.v >= vCut);
    return lower.length >= 24 ? lower : [...cloud];
  };

  const platClouds = sweepClouds.map((c) => (c.length > 0 ? filterPlatformRegion(c) : []));
  const startPlat = platClouds[startIdx] ?? [];
  const endPlat = platClouds[endIdx] ?? [];
  if (startPlat.length < 20 || endPlat.length < 20) {
    return { obs: null, pointCount: Math.min(startPlat.length, endPlat.length), ampErrMm: 999 };
  }

  // Chained frame-to-frame GNC-Welsch translation initialization
  let tChain: Vec3 = [0, 0, 0];
  let prevStep: Vec3 = [0, 0, 0];
  let validSteps = 0;
  let rmsSum = 0;
  for (let f = startIdx; f < endIdx; f++) {
    const c0 = platClouds[f];
    const c1 = platClouds[f + 1];
    if (!c0 || !c1 || c0.length < 18 || c1.length < 18) continue;
    const step = runGncWelschTranslationICP(c0, c1, prevStep);
    const stepLen = norm3(step.deltaT);
    if (step.inlierCount >= 14 && stepLen < 0.085) {
      tChain = [tChain[0] + step.deltaT[0], tChain[1] + step.deltaT[1], tChain[2] + step.deltaT[2]];
      prevStep = step.deltaT;
      rmsSum += step.rms;
      validSteps++;
    }
  }
  if (validSteps === 0) {
    return { obs: null, pointCount: startPlat.length, ampErrMm: 999 };
  }

  // Direct start-to-end keyframe GNC-Welsch refinement
  const refined = runGncWelschTranslationICP(startPlat, endPlat, tChain);
  let deltaPCam = tChain;
  let finalRms = rmsSum / validSteps;
  const startCentroid = refined.inlierCentroidSrc;

  const directLen = norm3(refined.deltaT);
  const chainLen = norm3(tChain);
  if (
    refined.inlierCount >= 16 &&
    Math.abs(directLen - baseDist) <= Math.max(0.045, Math.abs(chainLen - baseDist) + 0.02)
  ) {
    deltaPCam = refined.deltaT;
    finalRms = refined.rms;
  }

  const camDist = norm3(deltaPCam);
  if (camDist < 0.015) {
    return { obs: null, pointCount: startPlat.length, ampErrMm: 999 };
  }
  const ampErrMm = Math.abs(camDist - baseDist) * 1000;

  // Mid-sweep translating platform centroid in {C} from Welsch inliers + 0.5 * deltaPCam
  const centroidCam: Vec3 = [
    startCentroid[0] + 0.5 * deltaPCam[0],
    startCentroid[1] + 0.5 * deltaPCam[1],
    startCentroid[2] + 0.5 * deltaPCam[2],
  ];

  // Platform + wrist visual mass sits ~22 mm below delta_platform origin in Z
  const centroidBaseAdjusted: Vec3 = [
    centroidBase[0],
    centroidBase[1],
    centroidBase[2] - 0.022,
  ];

  const ampPenalty = Math.exp(-(ampErrMm * ampErrMm) / (2 * 10 * 10));
  const rmsPenalty = 1 / (1 + (finalRms / 0.008) ** 2);
  const weight = Math.max(0.05, ampPenalty * rmsPenalty);

  const unitCam = normalize3(deltaPCam);
  const unitBase = normalize3(deltaPBase);

  return {
    obs: {
      action,
      kind: 'translation',
      xiCam: [0, 0, 0, unitCam[0], unitCam[1], unitCam[2]],
      xiBase: [0, 0, 0, unitBase[0], unitBase[1], unitBase[2]],
      deltaQ,
      thetaObs: camDist,
      pitch: 0,
      icpRms: finalRms,
      weight,
      dTCam: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: deltaPCam },
      deltaPBase,
      deltaPCam,
      centroidBase: centroidBaseAdjusted,
      centroidCam,
    },
    pointCount: startPlat.length,
    ampErrMm,
  };
}

/**
 * Evaluate the 3 local, zero-extrinsic blind-probe admission gates for a cold-start probe.
 */
export function evaluateBlindProbeGates(
  obs: ScrewObservation | null,
  pointCount: number,
  ampResidual: number,
  acceptedUnitAxes: readonly Vec3[],
  isParallelTranslation: boolean
): V3ProbeLogEntry {
  const ampUnit = isParallelTranslation ? 'mm' : 'deg';
  if (!obs) {
    return {
      stepLabel: 'probe',
      jointIndex: -1,
      accepted: false,
      pointCount,
      ampResidual,
      ampUnit,
      sigmaMin: 0,
      reason: '有效前景点云不足或运动幅值退化',
    };
  }

  const candAxis: Vec3 = isParallelTranslation
    ? normalize3(obs.deltaPCam ?? [0, 0, 1])
    : normalize3([obs.xiCam[0], obs.xiCam[1], obs.xiCam[2]]);

  const stacked = [...acceptedUnitAxes, candAxis];
  const sigmaMin = computeAxisIndependenceSigmaMin(stacked);

  if (pointCount < V3_PROBE_MIN_POINTS) {
    return {
      stepLabel: obs.action.label,
      jointIndex: obs.action.jointIndex,
      accepted: false,
      pointCount,
      ampResidual,
      ampUnit,
      sigmaMin,
      reason: `门控1未通过: 前景有效点 ${pointCount} < ${V3_PROBE_MIN_POINTS}`,
    };
  }

  const maxAmp = isParallelTranslation ? V3_PROBE_MAX_TRANS_ERR_MM : V3_PROBE_MAX_ANGLE_ERR_DEG;
  if (ampResidual > maxAmp) {
    return {
      stepLabel: obs.action.label,
      jointIndex: obs.action.jointIndex,
      accepted: false,
      pointCount,
      ampResidual,
      ampUnit,
      sigmaMin,
      reason: `门控2未通过: 幅值残差 ${ampResidual.toFixed(2)}${ampUnit} > ${maxAmp}${ampUnit}`,
    };
  }

  if (stacked.length >= 2 && sigmaMin < V3_PROBE_MIN_SIGMA_SIN) {
    return {
      stepLabel: obs.action.label,
      jointIndex: obs.action.jointIndex,
      accepted: false,
      pointCount,
      ampResidual,
      ampUnit,
      sigmaMin,
      reason: `门控3未通过: 线性无关性 σ_min=${sigmaMin.toFixed(3)} < ${V3_PROBE_MIN_SIGMA_SIN.toFixed(3)}`,
    };
  }

  return {
    stepLabel: obs.action.label,
    jointIndex: obs.action.jointIndex,
    accepted: true,
    pointCount,
    ampResidual,
    ampUnit,
    sigmaMin,
    reason: '通过全部 3 项盲探针局部准入门控',
  };
}

/** Cross-topology V3 Stage 2 generalized screw closed-form solver */
export function solveV3GeneralizedScrewClosedForm(
  observations: readonly ScrewObservation[],
  isParallelDelta: boolean
): SE3 {
  if (isParallelDelta) {
    return solveParallelHandEyeClosedForm(observations);
  }
  return solveScrewHandEyeClosedForm(observations);
}
