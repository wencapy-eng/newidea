import {
  applyAdjoint,
  cloneSE3,
  computePoseError,
  cross3,
  dot3,
  inverseTransformPoint,
  mat3TransposeMulVec,
  norm3,
  normalize3,
  SeededRNG,
  SpatialHashGrid3D,
  transformPoint,
  type PointNormalSample,
  type SE3,
  type Vec3,
  type Vec6,
} from '../../math/se3';
import type { MujocoPandaEngine } from '../../sim/mujocoEngine';
import type { PandaRGBDScene } from '../../sim/rgbdScene';
import {
  addMat6InPlace,
  buildColdStartActions,
  buildSafeCandidatePool,
  evaluateFIMMetrics,
} from '../fimPlanner';
import { backprojectPixel } from '../frontend';
import {
  evaluateActionTrajectory,
  evaluateDistractorPos,
  FRAME_DT_SEC,
  FRAMES_PER_ACTION,
  generateInitialHandEyePose,
  nextAnimationFrame,
  POST_PAUSE_FRAMES,
  PRE_PAUSE_FRAMES,
  sampleRandomCameraPose,
  SWEEP_FRAMES,
  type LiveStepUpdate,
} from '../pipeline';
import { appendErrorHistoryPoint } from '../../utils/errorHistory';
import { traceLogger } from '../../utils/traceLogger';
import type {
  AblationRow,
  CalibrationConfig,
  CalibrationResult,
  ErrorHistoryPoint,
  ExtrinsicInitMode,
  LMTracePoint,
  MotionAction,
  RGBDFrame,
  ScrewObservation,
  SyncDiagnostics,
  V3Diagnostics,
  V3ProbeLogEntry,
} from '../types';
import {
  computeAnnealedEmPosterior,
  extractSavitzkyGolayVelocityObservations,
  iterateStage4AnnealedEmLM,
  runStage4AnnealedEmLM,
  type V3DynamicFrameGrid,
  type V3JointDynamicObs,
  type V3StaticFrameObs,
} from './annealedEmLmStage4';
import {
  evaluateBlindProbeGates,
  extractV3ParallelTranslationFromSweep,
  extractV3ScrewObservationFromSweep,
  solveV3GeneralizedScrewClosedForm,
} from './blindProbeStage2';
import {
  buildDepthConditionedOcc2DBand,
  computeDualDomainForegroundMask,
  extractV3OpticalWeightedCloud,
  OnlineDepthEnvelope,
  segmentActionV3MultiFragment,
  StaticVoxelExclusion3D,
  V3_STATIC_VOXEL_RADIUS_M,
  type Occ2DExclusionMap,
} from './dualDomainFrontend';
import {
  computeOcclusionAwareDimensionlessFIM,
  selectNextActionV3,
} from './occlusionFimPlanner';

/**
 * Build a crisp dense per-pixel Class I / II / III / M_occ_2D posterior overlay map
 * on a mid-sweep frame using the converged V3 Stage 4 (T_final, deltaT, priors) and real depth flow.
 *  0 = background, 1 = Class I (Cyan), 2 = Class II (Amber), 3 = Class III (Magenta), 4 = M_occ_2D (Crimson)
 */
function buildDensePosteriorOverlay(
  frames: readonly RGBDFrame[],
  frameMasks: readonly Uint8Array[],
  xiBaseAtFrame: readonly Vec6[],
  midIdx: number,
  occ2D: Occ2DExclusionMap | null,
  engine: MujocoPandaEngine,
  T_final: SE3,
  deltaTSec: number,
  priors: readonly [number, number, number],
  K: ReturnType<PandaRGBDScene['getIntrinsics']>
): Uint8Array {
  const W = K.width;
  const H = K.height;
  const overlay = new Uint8Array(W * H);

  // Paint 2D static front-occlusion edge band first (value 4)
  if (occ2D) {
    for (let i = 0; i < overlay.length; i++) {
      if (occ2D.bandMask[i]) overlay[i] = 4;
    }
  }

  const fCur = frames[midIdx];
  const fPrev = frames[Math.max(0, midIdx - 1)];
  const fNext = frames[Math.min(frames.length - 1, midIdx + 1)];
  const mask = frameMasks[midIdx];
  if (!fCur || !fPrev || !fNext || !mask) return overlay;

  const dt = Math.max(1e-4, 0.5 * (fNext.timestamp - fPrev.timestamp));
  const activeJ = fCur.activeJoint;
  const qdot = fCur.qdotEncoder[activeJ] ?? 0;
  const xiB = xiBaseAtFrame[midIdx] ?? [0, 0, 0, 0, 0, 0];
  const xiBaseDot: Vec6 = [
    xiB[0] * qdot,
    xiB[1] * qdot,
    xiB[2] * qdot,
    xiB[3] * qdot,
    xiB[4] * qdot,
    xiB[5] * qdot,
  ];
  const xiCam = applyAdjoint(T_final, xiBaseDot);
  const wC: Vec3 = [xiCam[0], xiCam[1], xiCam[2]];
  const vC: Vec3 = [xiCam[3], xiCam[4], xiCam[5]];

  // Build CAD spatial hash grid at q_enc(midIdx)
  const cadSamples = engine.getRobotSurfacePointsBase(fCur.qEncoder, undefined, 1);
  const modelGrid = new SpatialHashGrid3D<PointNormalSample>(0.045);
  modelGrid.insertAll(cadSamples);

  const d0 = fCur.depth;
  const dPrev = fPrev.depth;
  const dNext = fNext.depth;
  const sigmaGeo = 0.0085;
  const sigmaVel = 0.045;

  for (let v = 2; v < H - 2; v++) {
    const row = v * W;
    for (let u = 2; u < W - 2; u++) {
      const idx = row + u;
      if (!mask[idx]) continue;
      const z0 = d0[idx];
      if (z0 <= 0.14 || z0 >= 5.5) continue;

      const zL = d0[idx - 2];
      const zR = d0[idx + 2];
      const zU = d0[idx - 2 * W];
      const zD = d0[idx + 2 * W];
      if (zL <= 0.10 || zR <= 0.10 || zU <= 0.10 || zD <= 0.10) continue;

      const q = backprojectPixel(u, v, z0, K);
      const pL = backprojectPixel(u - 2, v, zL, K);
      const pR = backprojectPixel(u + 2, v, zR, K);
      const pU = backprojectPixel(u, v - 2, zU, K);
      const pD = backprojectPixel(u, v + 2, zD, K);
      const dPdu: Vec3 = [pR[0] - pL[0], pR[1] - pL[1], pR[2] - pL[2]];
      const dPdv: Vec3 = [pD[0] - pU[0], pD[1] - pU[1], pD[2] - pU[2]];
      let n = cross3(dPdv, dPdu);
      const nn = norm3(n);
      if (nn < 1e-8) continue;
      n = [n[0] / nn, n[1] / nn, n[2] / nn];
      if (dot3(n, q) > 0) n = [-n[0], -n[1], -n[2]];

      // Geometric distance to official CAD (shifted by estimated deltaT)
      const qBase = inverseTransformPoint(T_final, q);
      const wB: Vec3 = [xiBaseDot[0], xiBaseDot[1], xiBaseDot[2]];
      const vB: Vec3 = [xiBaseDot[3], xiBaseDot[4], xiBaseDot[5]];
      const wCrossQ = cross3(wB, qBase);
      const qBaseShifted: Vec3 = [
        qBase[0] - (wCrossQ[0] + vB[0]) * deltaTSec,
        qBase[1] - (wCrossQ[1] + vB[1]) * deltaTSec,
        qBase[2] - (wCrossQ[2] + vB[2]) * deltaTSec,
      ];
      const nBase = mat3TransposeMulVec(T_final.R, n);
      const hit = modelGrid.findNearest(qBaseShifted, 0.048, (cand) => dot3(cand.n, nBase) > 0.30);

      let dGeo: number | null = null;
      if (hit) {
        const sCam0 = transformPoint(T_final, hit.item.p);
        const wCrossS = cross3(wC, sCam0);
        const sCam: Vec3 = [
          sCam0[0] + (wCrossS[0] + vC[0]) * deltaTSec,
          sCam0[1] + (wCrossS[1] + vC[1]) * deltaTSec,
          sCam0[2] + (wCrossS[2] + vC[2]) * deltaTSec,
        ];
        const eGeo =
          n[0] * (sCam[0] - q[0]) +
          n[1] * (sCam[1] - q[1]) +
          n[2] * (sCam[2] - q[2]);
        dGeo = Math.abs(eGeo) / sigmaGeo;
      }

      // Dynamic velocity consistency
      let dVel: number | null = null;
      const zPrev = dPrev[idx];
      const zNext = dNext[idx];
      if (
        Math.abs(qdot) >= 0.14 &&
        zPrev > 0.12 &&
        zNext > 0.12 &&
        Math.abs(zNext - z0) < 0.08 &&
        Math.abs(z0 - zPrev) < 0.08
      ) {
        const dDdt = (zNext - zPrev) / (2 * dt);
        const vNObs = +dDdt * (dot3(n, q) / z0);
        const wCrossQc = cross3(wC, q);
        const vPred: Vec3 = [wCrossQc[0] + vC[0], wCrossQc[1] + vC[1], wCrossQc[2] + vC[2]];
        const eVel = dot3(n, vPred) - vNObs;
        dVel = Math.abs(eVel) / sigmaVel;
      }

      const [g1, g2, g3] = computeAnnealedEmPosterior(dGeo, dVel, 1.0, priors, 2.6, 2.8);
      overlay[idx] = g1 >= g2 && g1 >= g3 ? 1 : g2 >= g3 ? 2 : 3;
    }
  }

  return overlay;
}

/**
 * Execute one complete real-physics, real-WebGL2 V3.0 Unified Active Hand-Eye Calibration run:
 *  - Stage 1: 2D M_occ_2D + 3D V_static Dual-Domain Exclusion & Multi-Fragment Causal Sync Aggregation
 *  - Stage 2: Self-Validating Blind-Probe Cold Start + Chained GNC-Welsch ICP + Cross-Topology Screw Solver
 *  - Stage 3: Static Z-Buffer Occlusion-Aware Dimensionless FIM Active Exploration
 *  - Stage 4: 5-Frame Savitzky-Golay Depth Flow + 3-Component Annealed EM-LM (+ Optional 7th State delta_t)
 */
export async function runSingleCalibrationV3(
  engine: MujocoPandaEngine,
  rgbdScene: PandaRGBDScene,
  T_cb_gt: SE3,
  config: CalibrationConfig,
  seed: number,
  opts?: {
    onLiveUpdate?: (upd: LiveStepUpdate) => void;
    shouldCancel?: () => boolean;
    fastMode?: boolean;
    externalInitPose?: SE3 | null;
    /** Optional ablation flag to toggle 2D+3D dual-domain static exclusion (default true) */
    useDualExclusion?: boolean;
  }
): Promise<CalibrationResult> {
  const t0 = performance.now();
  const rng = new SeededRNG(seed);
  const K = rgbdScene.getIntrinsics();
  const fastMode = opts?.fastMode ?? false;
  const useDualExclusion = opts?.useDualExclusion ?? true;
  const useExternalInit = config.initMode !== 'stage1';
  const isParallelDelta = engine.spec.mechanism === 'parallel_delta';

  traceLogger.log(
    'stage',
    'pipelineV3',
    'runSingleCalibrationV3',
    `[V3] 启动 ${engine.spec.shortName} 统一引擎标定 (${config.mode} / ${config.strategy} / ${config.initMode})`,
    {
      input: {
        robotId: config.robotId,
        mode: config.mode,
        strategy: config.strategy,
        initMode: config.initMode,
        numSteps: config.numSteps,
        useDualExclusion,
        estimateTimeOffset: config.estimateTimeOffset,
        seed,
        fastMode,
        T_cb_gt,
        sensor: config.sensor,
      },
    }
  );

  const externalInitPose: SE3 | null = useExternalInit
    ? opts?.externalInitPose
      ? cloneSE3(opts.externalInitPose)
      : traceLogger.traceCall(
          'pipelineV3',
          'generateInitialHandEyePose',
          {
            initMode: config.initMode,
            rotErrDeg: config.initRotErrDeg,
            transErrMm: config.initTransErrMm,
          },
          () =>
            generateInitialHandEyePose(
              config.initMode as Exclude<ExtrinsicInitMode, 'stage1'>,
              T_cb_gt,
              new SeededRNG((seed ^ 0x9e3779b9) >>> 0),
              {
                rotErrDeg: config.initRotErrDeg,
                transErrMm: config.initTransErrMm,
              }
            ),
          (pose) => ({ pose, initError: computePoseError(pose, T_cb_gt) })
        )
    : null;

  // Apply all physical scene elements (custom tool, distractor, front pillar, flexible cable)
  rgbdScene.updateFrontPillarForCamera(T_cb_gt);
  rgbdScene.setSceneVisibility({
    showCustomTool: config.enableCustomTool,
    showDistractor: config.enableDistractor,
    showFrontPillar: config.enableFrontPillar,
    showFlexCable: config.enableFlexCable,
  });
  rgbdScene.setFrustumPoses(T_cb_gt, externalInitPose, null);

  const coldStart = traceLogger.traceCall(
    'fimV3',
    'buildColdStartActions',
    { robotId: engine.spec.id, dof: engine.spec.dof },
    () => buildColdStartActions(engine),
    (acts) => ({ count: acts.length, ids: acts.map((a) => a.id) })
  );
  const candidatePool = traceLogger.traceCall(
    'fimV3',
    'buildSafeCandidatePool',
    { robotId: engine.spec.id, dof: engine.spec.dof },
    () => buildSafeCandidatePool(engine),
    (acts) => ({ count: acts.length, ids: acts.map((a) => a.id) })
  );
  const totalSteps = Math.max(4, Math.min(8, config.numSteps));

  const usedIds = new Set<string>();
  const executedActions: MotionAction[] = [];
  const allActionFrames: RGBDFrame[][] = [];
  const allActionMasks: Uint8Array[][] = [];
  const allActionXiBase: Vec6[][] = [];
  const screwObservations: ScrewObservation[] = [];
  const probeLogs: V3ProbeLogEntry[] = [];
  const acceptedColdAxes: Vec3[] = [];
  const globalRejectedMask = new Uint8Array(K.width * K.height);
  const globalEnvelope = new OnlineDepthEnvelope(K.width, K.height, 6);

  let I_acc = new Float64Array(36);
  let T_cb_stage2: SE3 | null = externalInitPose ? cloneSE3(externalInitPose) : null;
  let errorHistory: ErrorHistoryPoint[] = [];
  if (externalInitPose) {
    const extErr = computePoseError(externalInitPose, T_cb_gt);
    errorHistory = appendErrorHistoryPoint(errorHistory, {
      label: 'Init',
      phase: 'init',
      transErrMm: extErr.transMm,
      rotErrDeg: extErr.rotDeg,
    });
    if (!fastMode) {
      opts?.onLiveUpdate?.({
        phaseText: `[V3] 初始外参就绪 (${extErr.transMm.toFixed(1)}mm / ${extErr.rotDeg.toFixed(2)}°)`,
        progress: 0.03,
        stage1Pose: T_cb_stage2,
        errorHistory: [...errorHistory],
      });
    }
  }
  let lastSyncDiag: SyncDiagnostics | null = null;
  let lastRgb: Uint8ClampedArray | null = null;
  let lastDepth: Float32Array | null = null;
  let lastMask: Uint8Array | null = null;
  let totalFimOccluded = 0;
  let totalFimInFov = 0;

  const distractorPhase = rng.uniform(0, 2 * Math.PI);
  let tGlobal = 0;

  const executeOneActionV3 = async (
    action: MotionAction,
    stepIdx: number,
    isColdProbe: boolean
  ): Promise<{ acceptedProbe: boolean }> => {
    executedActions.push(action);
    usedIds.add(action.id);

    const frames: RGBDFrame[] = [];
    const xiBasePerFrame: Vec6[] = [];
    const delaySec = Math.max(0, config.sensor.asyncDelayMs) / 1000;

    const actionT0 = performance.now();
    for (let f = 0; f < FRAMES_PER_ACTION; f++) {
      if (opts?.shouldCancel?.()) break;
      const tLocal = f * FRAME_DT_SEC;
      const tCurGlobal = tGlobal + tLocal;

      const { q: qTrue } = evaluateActionTrajectory(action, tLocal);
      const distPos = config.enableDistractor
        ? evaluateDistractorPos(tCurGlobal, distractorPhase)
        : 0;
      const { q: qEncoder, qdot: qdotEncoder } = evaluateActionTrajectory(
        action,
        Math.max(0, tLocal - delaySec)
      );

      engine.setState(qTrue, distPos);
      rgbdScene.syncFromMujoco();
      if (config.enableFlexCable) {
        const qdotNorm = Math.abs(qdotEncoder[action.jointIndex] ?? 0);
        rgbdScene.updateFlexCable(tCurGlobal, qdotNorm);
      }

      const { rgb, depth } = rgbdScene.captureVirtualRGBD(T_cb_gt, config.sensor, rng);
      globalEnvelope.addFrame(depth);

      const xiBase = engine.getEffectiveActionTwistBase(action.jointIndex, qEncoder);
      xiBasePerFrame.push(xiBase);

      frames.push({
        width: K.width,
        height: K.height,
        rgb,
        depth,
        timestamp: tCurGlobal,
        qTrue,
        qEncoder,
        qdotEncoder,
        activeJoint: action.jointIndex,
      });

      if (!fastMode && (f % 4 === 0 || f === FRAMES_PER_ACTION - 1)) {
        engine.setState(qTrue, distPos);
        rgbdScene.syncFromMujoco();
        if (config.enableFlexCable) {
          rgbdScene.updateFlexCable(tCurGlobal, Math.abs(qdotEncoder[action.jointIndex] ?? 0));
        }
        rgbdScene.renderNow();
        opts?.onLiveUpdate?.({
          phaseText: `[V3] 步骤 ${stepIdx + 1}/${totalSteps}: ${action.label} (帧 ${f + 1}/${FRAMES_PER_ACTION})`,
          progress: (stepIdx + (f + 1) / FRAMES_PER_ACTION) / (totalSteps + 1.3),
          rgb,
          depth,
          stage1Pose: T_cb_stage2,
        });
        await nextAnimationFrame();
      }
    }

    tGlobal += FRAMES_PER_ACTION * FRAME_DT_SEC + 0.15;

    if (!fastMode) {
      traceLogger.recordCall(
        'sim',
        'captureActionSequenceV3',
        {
          stepIdx,
          isColdProbe,
          actionId: action.id,
          label: action.label,
          jointIndex: action.jointIndex,
          qStart: action.qStart,
          qEnd: action.qEnd,
          asyncDelayMs: config.sensor.asyncDelayMs,
        },
        {
          frameCount: frames.length,
          midDepthSummary: frames[Math.floor(frames.length / 2)]?.depth,
          xiBaseMid: xiBasePerFrame[Math.floor(xiBasePerFrame.length / 2)],
        },
        performance.now() - actionT0,
        `[V3] 采集动作 ${action.label} (${frames.length} 帧 RGB-D)`
      );
    }

    // V3 Stage 1: Dual-Domain (2D M_occ_2D + 3D V_static) + Multi-Fragment Sync Aggregation
    const seg = !fastMode
      ? traceLogger.traceCall(
          'frontendV3',
          'segmentActionV3MultiFragment',
          {
            actionLabel: action.label,
            frameCount: frames.length,
            useTimestampSync: config.useTimestampSync,
            useDualExclusion,
          },
          () =>
            segmentActionV3MultiFragment(frames, K, action.label, {
              useTimestampSync: config.useTimestampSync,
              useDualExclusion,
              globalEnvelope,
            }),
          (res) => ({
            diagnostics: res.diagnostics,
            midFrameMaskPx: res.frameMasks[Math.floor(res.frameMasks.length / 2)]?.reduce(
              (acc, v) => acc + (v ? 1 : 0),
              0
            ),
          })
        )
      : segmentActionV3MultiFragment(frames, K, action.label, {
          useTimestampSync: config.useTimestampSync,
          useDualExclusion,
          globalEnvelope,
        });
    lastSyncDiag = seg.diagnostics;

    if (config.useTimestampSync) {
      for (let i = 0; i < globalRejectedMask.length; i++) {
        if (seg.rejectedRegionMask[i]) globalRejectedMask[i] = 1;
      }
    }

    allActionFrames.push(frames);
    allActionMasks.push(seg.frameMasks);
    allActionXiBase.push(xiBasePerFrame);

    const midIdx = Math.floor(FRAMES_PER_ACTION / 2);
    lastRgb = frames[midIdx].rgb;
    lastDepth = frames[midIdx].depth;
    lastMask = seg.frameMasks[midIdx];

    // Extract optical-weighted clouds across sweep frames for Stage 2 GNC-Welsch ICP
    const startIdx = PRE_PAUSE_FRAMES - 1;
    const endIdx = PRE_PAUSE_FRAMES + SWEEP_FRAMES;
    const sweepClouds = !fastMode
      ? traceLogger.traceCall(
          'frontendV3',
          'extractV3OpticalWeightedCloudBatch',
          {
            startIdx,
            endIdx,
            filterEnabled: config.useFlyingPixelFilter,
            sigmaZ0: config.sensor.sigmaZ0,
            maxPoints: 760,
          },
          () =>
            frames.map((fr, idx) => {
              if (idx < startIdx || idx > endIdx) return [];
              return extractV3OpticalWeightedCloud(fr.depth, seg.frameMasks[idx], K, {
                filterEnabled: config.useFlyingPixelFilter,
                sigmaZ0: config.sensor.sigmaZ0,
                maxPoints: 760,
              });
            }),
          (clouds) => ({
            perFramePointCounts: clouds.map((c) => c.length),
            totalPoints: clouds.reduce((s, c) => s + c.length, 0),
          })
        )
      : frames.map((fr, idx) => {
          if (idx < startIdx || idx > endIdx) return [];
          return extractV3OpticalWeightedCloud(fr.depth, seg.frameMasks[idx], K, {
            filterEnabled: config.useFlyingPixelFilter,
            sigmaZ0: config.sensor.sigmaZ0,
            maxPoints: 760,
          });
        });

    const qStartEnc = frames[startIdx].qEncoder[action.jointIndex];
    const qEndEnc = frames[endIdx].qEncoder[action.jointIndex];
    const deltaQ = qEndEnc - qStartEnc;
    const xiBaseMid = xiBasePerFrame[midIdx];
    const isDeltaTrans = isParallelDelta && action.jointIndex < 3;

    let screwObs: ScrewObservation | null = null;
    let pointCount = 0;
    let ampResidual = 99;

    if (isDeltaTrans) {
      const pStartB = engine.getDeltaPlatformPosBase(frames[startIdx].qEncoder);
      const pEndB = engine.getDeltaPlatformPosBase(frames[endIdx].qEncoder);
      const pMidB = engine.getDeltaPlatformPosBase(frames[midIdx].qEncoder);
      if (pStartB && pEndB && pMidB) {
        const deltaPBase: Vec3 = [
          pEndB[0] - pStartB[0],
          pEndB[1] - pStartB[1],
          pEndB[2] - pStartB[2],
        ];
        const res = !fastMode
          ? traceLogger.traceCall(
              'stage2V3',
              'extractV3ParallelTranslationFromSweep',
              { actionId: action.id, deltaPBase, pMidB, deltaQ },
              () =>
                extractV3ParallelTranslationFromSweep(
                  action,
                  sweepClouds,
                  startIdx,
                  endIdx,
                  deltaPBase,
                  pMidB,
                  deltaQ
                )
            )
          : extractV3ParallelTranslationFromSweep(
              action,
              sweepClouds,
              startIdx,
              endIdx,
              deltaPBase,
              pMidB,
              deltaQ
            );
        screwObs = res.obs;
        pointCount = res.pointCount;
        ampResidual = res.ampErrMm;
      }
    } else {
      const res = !fastMode
        ? traceLogger.traceCall(
            'stage2V3',
            'extractV3ScrewObservationFromSweep',
            { actionId: action.id, xiBaseMid, deltaQ, startIdx, endIdx },
            () =>
              extractV3ScrewObservationFromSweep(
                action,
                sweepClouds,
                startIdx,
                endIdx,
                xiBaseMid,
                deltaQ
              )
          )
        : extractV3ScrewObservationFromSweep(
            action,
            sweepClouds,
            startIdx,
            endIdx,
            xiBaseMid,
            deltaQ
          );
      screwObs = res.obs;
      pointCount = res.pointCount;
      ampResidual = res.ampErrDeg;
    }

    let acceptedProbe = true;
    if (isColdProbe) {
      const gateLog = !fastMode
        ? traceLogger.traceCall(
            'stage2V3',
            'evaluateBlindProbeGates',
            {
              actionId: action.id,
              pointCount,
              ampResidual,
              acceptedAxesCount: acceptedColdAxes.length,
              isDeltaTrans,
            },
            () =>
              evaluateBlindProbeGates(
                screwObs,
                pointCount,
                ampResidual,
                acceptedColdAxes,
                isDeltaTrans
              )
          )
        : evaluateBlindProbeGates(
            screwObs,
            pointCount,
            ampResidual,
            acceptedColdAxes,
            isDeltaTrans
          );
      probeLogs.push(gateLog);
      acceptedProbe = gateLog.accepted;
      if (gateLog.accepted && screwObs) {
        const axisVec: Vec3 = isDeltaTrans
          ? normalize3(screwObs.deltaPCam ?? [0, 0, 1])
          : normalize3([screwObs.xiCam[0], screwObs.xiCam[1], screwObs.xiCam[2]]);
        acceptedColdAxes.push(axisVec);
      }
    }

    if (screwObs) {
      screwObservations.push(screwObs);
    }

    if (!externalInitPose && screwObservations.length >= 3) {
      T_cb_stage2 = !fastMode
        ? traceLogger.traceCall(
            'stage2V3',
            'solveV3GeneralizedScrewClosedForm',
            { obsCount: screwObservations.length, isParallelDelta, stepIdx },
            () => solveV3GeneralizedScrewClosedForm(screwObservations, isParallelDelta),
            (pose) => ({ pose, error: computePoseError(pose, T_cb_gt) })
          )
        : solveV3GeneralizedScrewClosedForm(screwObservations, isParallelDelta);
      const stepErr = computePoseError(T_cb_stage2, T_cb_gt);
      errorHistory = appendErrorHistoryPoint(errorHistory, {
        label: `S${stepIdx + 1}`,
        phase: 'closed_form',
        transErrMm: stepErr.transMm,
        rotErrDeg: stepErr.rotDeg,
      });
    }

    opts?.onLiveUpdate?.({
      phaseText: `[V3] 完成步骤 ${stepIdx + 1}/${totalSteps}: ${action.label}`,
      progress: (stepIdx + 1) / (totalSteps + 1.3),
      rgb: lastRgb,
      depth: lastDepth,
      mask: lastMask,
      syncDiagnostics: lastSyncDiag,
      stage1Pose: T_cb_stage2,
      errorHistory: [...errorHistory],
    });
    await nextAnimationFrame();

    return { acceptedProbe };
  };

  // ===== PHASE A (Stage 2): Self-Validating Blind-Probe Cold Start =====
  // Queue starts with orthogonal major axes, backed up by candidatePool if any probe fails the 3 local gates
  const probeQueue: MotionAction[] = [
    ...coldStart,
    ...candidatePool.filter((c) => !coldStart.some((cs) => cs.id === c.id)),
  ];
  let stepCursor = 0;
  let passedColdProbes = 0;
  const targetColdProbes = Math.min(3, totalSteps);
  const maxColdBudget = Math.min(totalSteps - 1, targetColdProbes + 1);

  for (let p = 0; p < probeQueue.length && passedColdProbes < targetColdProbes && stepCursor < maxColdBudget; p++) {
    if (opts?.shouldCancel?.()) break;
    const cand = probeQueue[p];
    if (usedIds.has(cand.id)) continue;
    const { acceptedProbe } = await executeOneActionV3(cand, stepCursor, true);
    stepCursor++;
    if (acceptedProbe) passedColdProbes++;
  }

  // Solve Stage 2 closed-form pose (or use external init pose if user selected one)
  const solvedStage2 = solveV3GeneralizedScrewClosedForm(screwObservations, isParallelDelta);
  T_cb_stage2 = externalInitPose ? cloneSE3(externalInitPose) : solvedStage2;

  // Build online static Z-Buffer D_max(u,v) from accumulated cold-start frames
  let { dMax: staticZBuf, dMin: staticDMin } = globalEnvelope.getEnvelopes();

  // Initialize accumulated dimensionless FIM I_acc from executed cold-start actions
  I_acc = new Float64Array(36);
  for (let i = 0; i < executedActions.length; i++) {
    const fimRes = computeOcclusionAwareDimensionlessFIM(
      engine,
      executedActions[i],
      T_cb_stage2,
      K,
      config.sensor.sigmaZ0,
      useDualExclusion ? staticZBuf : null,
      config.useVelocityField
    );
    addMat6InPlace(I_acc, fimRes.I);
    totalFimOccluded += fimRes.occludedCount;
    totalFimInFov += fimRes.totalFovCount;
  }

  // ===== PHASE B (Stage 3): Static Z-Buffer Occlusion-Aware Dimensionless FIM Active Exploration =====
  for (let step = stepCursor; step < totalSteps; step++) {
    if (opts?.shouldCancel?.()) break;
    const env = globalEnvelope.getEnvelopes();
    staticZBuf = env.dMax;
    staticDMin = env.dMin;

    const pick = !fastMode
      ? traceLogger.traceCall(
          'fimV3',
          'selectNextActionV3',
          {
            step,
            strategy: config.strategy,
            remainingCandidates: candidatePool.length - usedIds.size,
            useDualExclusion,
            useVelocityField: config.useVelocityField,
          },
          () =>
            selectNextActionV3(
              engine,
              candidatePool,
              usedIds,
              config.strategy,
              I_acc,
              T_cb_stage2!,
              K,
              config.sensor.sigmaZ0,
              useDualExclusion ? staticZBuf : null,
              config.useVelocityField,
              rng
            ),
          (res) =>
            res
              ? {
                  selectedId: res.action.id,
                  label: res.action.label,
                  occludedCount: res.fimRes.occludedCount,
                  totalFovCount: res.fimRes.totalFovCount,
                }
              : null
        )
      : selectNextActionV3(
          engine,
          candidatePool,
          usedIds,
          config.strategy,
          I_acc,
          T_cb_stage2,
          K,
          config.sensor.sigmaZ0,
          useDualExclusion ? staticZBuf : null,
          config.useVelocityField,
          rng
        );
    if (!pick) break;

    await executeOneActionV3(pick.action, step, false);

    if (!externalInitPose && screwObservations.length >= 3) {
      T_cb_stage2 = solveV3GeneralizedScrewClosedForm(screwObservations, isParallelDelta);
    }

    const updatedEnv = globalEnvelope.getEnvelopes();
    staticZBuf = updatedEnv.dMax;
    staticDMin = updatedEnv.dMin;

    const updatedFim = computeOcclusionAwareDimensionlessFIM(
      engine,
      pick.action,
      T_cb_stage2,
      K,
      config.sensor.sigmaZ0,
      useDualExclusion ? staticZBuf : null,
      config.useVelocityField
    );
    addMat6InPlace(I_acc, updatedFim.I);
    totalFimOccluded += updatedFim.occludedCount;
    totalFimInFov += updatedFim.totalFovCount;
  }

  const stage2Pose = solveV3GeneralizedScrewClosedForm(screwObservations, isParallelDelta);
  const stage2Err = computePoseError(stage2Pose, T_cb_gt);
  const initPoseForStage4 = externalInitPose ? cloneSE3(externalInitPose) : cloneSE3(stage2Pose);
  const initErr = computePoseError(initPoseForStage4, T_cb_gt);
  const fimMetrics = evaluateFIMMetrics(I_acc);

  // Build final global 2D M_occ_2D & 3D V_static static exclusion structures across all frames
  const finalEnv = globalEnvelope.getEnvelopes();
  staticZBuf = finalEnv.dMax;
  staticDMin = finalEnv.dMin;
  const globalOcc2D = !fastMode
    ? traceLogger.traceCall(
        'frontendV3',
        'buildDepthConditionedOcc2DBand',
        { width: K.width, height: K.height },
        () => buildDepthConditionedOcc2DBand(staticZBuf, staticDMin, K.width, K.height),
        (res) => ({ bandPixelCount: res.bandPixelCount })
      )
    : buildDepthConditionedOcc2DBand(staticZBuf, staticDMin, K.width, K.height);
  const globalVStatic = new StaticVoxelExclusion3D(V3_STATIC_VOXEL_RADIUS_M);
  globalVStatic.buildFromStaticEnvelope(staticZBuf, K, 2);

  // ===== PHASE C (Stage 4): 3-Component Annealed EM-LM Multi-Body Refinement (+ Optional 7th State delta_t) =====
  let finalPose = config.mode === 'modeB' ? cloneSE3(initPoseForStage4) : cloneSE3(stage2Pose);
  let estimatedDelayMs: number | null = null;
  let classPriors: [number, number, number] = [0.76, 0.14, 0.10];
  let classCounts: [number, number, number] = [0, 0, 0];
  let posteriorOverlay: Uint8Array | null = null;
  let lmTrace = [
    {
      iter: 0,
      stage: 0,
      cost: 0,
      rotErrDeg: config.mode === 'modeB' ? initErr.rotDeg : stage2Err.rotDeg,
      transErrMm: config.mode === 'modeB' ? initErr.transMm : stage2Err.transMm,
      geoCount: 0,
      velCount: 0,
      lambda: 0,
    },
  ];

  if (config.mode === 'modeB' && !opts?.shouldCancel?.()) {
    opts?.onLiveUpdate?.({
      phaseText: '[V3] Stage 4: 5帧 Savitzky-Golay 深度流 + 3分量确定性退火 EM-LM 联合精修...',
      progress: (totalSteps + 0.5) / (totalSteps + 1.3),
      stage1Pose: stage2Pose,
    });
    await nextAnimationFrame();

    // 1. Static pause frames (dual-domain cleaned)
    const staticObs: V3StaticFrameObs[] = [];
    for (let a = 0; a < allActionFrames.length; a++) {
      const af = allActionFrames[a];
      const pauseIndices = [1, FRAMES_PER_ACTION - 2];
      for (const pIdx of pauseIndices) {
        const fr = af[pIdx];
        if (!fr) continue;
        const mask = computeDualDomainForegroundMask(
          fr.depth,
          staticZBuf,
          K,
          useDualExclusion ? globalVStatic : null,
          useDualExclusion ? globalOcc2D : null,
          { useDualExclusion, minComponentPx: 32 }
        );
        if (config.useTimestampSync) {
          for (let i = 0; i < mask.length; i++) {
            if (globalRejectedMask[i]) mask[i] = 0;
          }
        }
        const pts = extractV3OpticalWeightedCloud(fr.depth, mask, K, {
          filterEnabled: config.useFlyingPixelFilter,
          sigmaZ0: config.sensor.sigmaZ0,
          maxPoints: 580,
        });
        if (pts.length < 25) continue;

        const cadSamples = engine.getRobotSurfacePointsBase(fr.qEncoder, undefined, 1);
        const modelGrid = new SpatialHashGrid3D<PointNormalSample>(0.045);
        modelGrid.insertAll(cadSamples);
        staticObs.push({ qEncoder: fr.qEncoder, points: pts, modelGrid });
      }
    }

    // 2. Unified dynamic observations (5-frame Savitzky-Golay velocity + per-frame CAD grid)
    const dynObsAll: V3JointDynamicObs[] = [];
    const dynFrameGridsAll: V3DynamicFrameGrid[] = [];
    for (let a = 0; a < allActionFrames.length; a++) {
      const { obs, frameGrids } = extractSavitzkyGolayVelocityObservations(
        allActionFrames[a],
        allActionMasks[a],
        allActionXiBase[a],
        K,
        a,
        (qEnc) => {
          const cadSamples = engine.getRobotSurfacePointsBase(qEnc, undefined, 1);
          const g = new SpatialHashGrid3D<PointNormalSample>(0.045);
          g.insertAll(cadSamples);
          return g;
        },
        {
          filterEnabled: config.useFlyingPixelFilter,
          sigmaZ0: config.sensor.sigmaZ0,
          maxObsPerAction: 380,
        }
      );
      const gridOffset = dynFrameGridsAll.length;
      for (let g = 0; g < frameGrids.length; g++) dynFrameGridsAll.push(frameGrids[g]);
      for (let i = 0; i < obs.length; i++) {
        obs[i].frameGroupIdx += gridOffset;
        dynObsAll.push(obs[i]);
      }
    }

    // Annealing temperature & correspondence gate schedule based solely on initMode (zero GT leakage)
    let temps: number[] = [16.0, 4.0, 1.0];
    let gates: number[] = [0.13, 0.08, 0.048];
    let iters: number[] = [7, 7, 10];
    if (config.initMode === 'controlled_random') {
      const coarseGate = Math.max(0.16, Math.min(0.45, (config.initTransErrMm / 1000) * 1.35 + 0.08));
      temps = [24.0, 12.0, 4.0, 1.0];
      gates = [coarseGate, Math.max(0.11, coarseGate * 0.56), 0.08, 0.048];
      iters = [9, 8, 7, 10];
    } else if (config.initMode === 'full_random' || config.initMode === 'zero') {
      temps = [32.0, 14.0, 4.0, 1.0];
      gates = [0.46, 0.25, 0.12, 0.052];
      iters = [10, 9, 8, 10];
    }

    const stage4Opts = {
      useVelocityField: config.useVelocityField,
      estimateTimeOffset: config.estimateTimeOffset,
      temperatures: temps,
      gates,
      itersPerStage: iters,
      T_cb_gt_for_log: T_cb_gt,
    };

    let stage4DeltaTSec = 0;

    if (fastMode) {
      const stage4Res = traceLogger.traceCall(
        'stage4V3',
        'runStage4AnnealedEmLM',
        {
          staticFrames: staticObs.length,
          dynObsCount: dynObsAll.length,
          temperatures: temps,
          gates,
          itersPerStage: iters,
          estimateTimeOffset: config.estimateTimeOffset,
          initErr,
        },
        () =>
          runStage4AnnealedEmLM(
            initPoseForStage4,
            staticObs,
            dynObsAll,
            dynFrameGridsAll,
            stage4Opts
          ),
        (res) => ({
          iters: res.trace.length,
          classPriors: res.classPriors,
          classCounts: res.classCounts,
          estimatedDelayMs: res.deltaTSec * 1000,
          finalErr: computePoseError(res.T_est, T_cb_gt),
        })
      );
      finalPose = stage4Res.T_est;
      lmTrace = stage4Res.trace;
      classPriors = stage4Res.classPriors;
      classCounts = stage4Res.classCounts;
      stage4DeltaTSec = stage4Res.deltaTSec;
      estimatedDelayMs = config.estimateTimeOffset ? stage4Res.deltaTSec * 1000 : null;
      for (let i = 0; i < lmTrace.length; i++) {
        const pt = lmTrace[i];
        errorHistory = appendErrorHistoryPoint(errorHistory, {
          label: `EM${pt.iter}`,
          phase: 'em_lm',
          transErrMm: pt.transErrMm,
          rotErrDeg: pt.rotErrDeg,
          stage: pt.stage,
        });
      }
    } else {
      traceLogger.log(
        'stage',
        'stage4V3',
        'iterateStage4AnnealedEmLM',
        `[V3] Stage 4 退火 EM-LM 开始 (${staticObs.length} 静态帧 / ${dynObsAll.length} SG 动态观测)`,
        {
          input: {
            staticFrameCount: staticObs.length,
            staticPointTotal: staticObs.reduce((s, f) => s + f.points.length, 0),
            dynObsCount: dynObsAll.length,
            temperatures: temps,
            gates,
            itersPerStage: iters,
            estimateTimeOffset: config.estimateTimeOffset,
            initPose: initPoseForStage4,
            initErr,
          },
        }
      );
      const totalPlannedIters = iters.reduce((s, n) => s + n, 0);
      const gen = iterateStage4AnnealedEmLM(
        initPoseForStage4,
        staticObs,
        dynObsAll,
        dynFrameGridsAll,
        stage4Opts
      );
      let iterT0 = performance.now();
      let iterRes = gen.next();
      let liveTrace: LMTracePoint[] = [];
      let livePose = initPoseForStage4;
      while (!iterRes.done) {
        if (opts?.shouldCancel?.()) break;
        const step = iterRes.value;
        const iterDur = performance.now() - iterT0;
        livePose = step.T_est;
        liveTrace = step.trace;
        classPriors = step.classPriors;
        stage4DeltaTSec = step.deltaTSec;
        traceLogger.recordCall(
          'stage4V3',
          'stage4EmLM_iter',
          {
            iter: step.point.iter,
            stage: step.point.stage,
            lambda: step.point.lambda,
          },
          {
            cost: step.point.cost,
            geoCount: step.point.geoCount,
            velCount: step.point.velCount,
            classPriors: step.classPriors,
            estimatedDelayMs: step.deltaTSec * 1000,
            transErrMm: step.point.transErrMm,
            rotErrDeg: step.point.rotErrDeg,
          },
          iterDur,
          `EM-LM Iter ${step.point.iter} (T-Stage ${step.point.stage}): ${step.point.transErrMm.toFixed(2)}mm / ${step.point.rotErrDeg.toFixed(3)}°`
        );
        errorHistory = appendErrorHistoryPoint(errorHistory, {
          label: `EM${step.point.iter}`,
          phase: 'em_lm',
          transErrMm: step.point.transErrMm,
          rotErrDeg: step.point.rotErrDeg,
          stage: step.point.stage,
        });
        const frac = Math.min(1, step.point.iter / Math.max(1, totalPlannedIters));
        rgbdScene.setFrustumPoses(T_cb_gt, initPoseForStage4, livePose);
        rgbdScene.renderNow();
        opts?.onLiveUpdate?.({
          phaseText: `[V3] Stage 4 EM-LM Iter ${step.point.iter} · ${step.point.transErrMm.toFixed(2)} mm / ${step.point.rotErrDeg.toFixed(3)}°`,
          progress: (totalSteps + 0.5 + 0.76 * frac) / (totalSteps + 1.3),
          stage1Pose: initPoseForStage4,
          stage2Pose: livePose,
          lmTrace: [...liveTrace],
          errorHistory: [...errorHistory],
        });
        await nextAnimationFrame();
        iterT0 = performance.now();
        iterRes = gen.next();
      }
      if (iterRes.done && iterRes.value) {
        finalPose = iterRes.value.T_est;
        lmTrace = iterRes.value.trace;
        classPriors = iterRes.value.classPriors;
        classCounts = iterRes.value.classCounts;
        stage4DeltaTSec = iterRes.value.deltaTSec;
      } else {
        finalPose = livePose;
        lmTrace = liveTrace;
      }
      estimatedDelayMs = config.estimateTimeOffset ? stage4DeltaTSec * 1000 : null;
    }

    // Build real per-pixel Class I / II / III / M_occ_2D posterior map on last action's mid frame
    if (allActionFrames.length > 0) {
      const lastActIdx = allActionFrames.length - 1;
      const midIdx = Math.floor(FRAMES_PER_ACTION / 2);
      posteriorOverlay = !fastMode
        ? traceLogger.traceCall(
            'stage4V3',
            'buildDensePosteriorOverlay',
            { lastActIdx, midIdx, deltaTSec: stage4DeltaTSec, classPriors },
            () =>
              buildDensePosteriorOverlay(
                allActionFrames[lastActIdx],
                allActionMasks[lastActIdx],
                allActionXiBase[lastActIdx],
                midIdx,
                useDualExclusion ? globalOcc2D : null,
                engine,
                finalPose,
                stage4DeltaTSec,
                classPriors,
                K
              ),
            (ov) => {
              const hist = [0, 0, 0, 0, 0];
              for (let i = 0; i < ov.length; i++) hist[ov[i]] = (hist[ov[i]] ?? 0) + 1;
              return { bgPx: hist[0], classIPx: hist[1], classIIPx: hist[2], classIIIPx: hist[3], occ2DPx: hist[4] };
            }
          )
        : buildDensePosteriorOverlay(
            allActionFrames[lastActIdx],
            allActionMasks[lastActIdx],
            allActionXiBase[lastActIdx],
            midIdx,
            useDualExclusion ? globalOcc2D : null,
            engine,
            finalPose,
            stage4DeltaTSec,
            classPriors,
            K
          );
    }
  } else if (allActionFrames.length > 0 && useDualExclusion) {
    // Mode A still shows 2D front-occlusion band M_occ_2D on the diagnostic depth canvas
    const overlay = new Uint8Array(K.width * K.height);
    for (let i = 0; i < overlay.length; i++) {
      if (globalOcc2D.bandMask[i]) overlay[i] = 4;
      else if (lastMask && lastMask[i]) overlay[i] = 1;
    }
    posteriorOverlay = overlay;
  }

  // Return robot to home posture & update 3 frustums in viewport
  engine.setState(engine.spec.homeQ, 0);
  rgbdScene.syncFromMujoco();
  if (config.enableFlexCable) {
    rgbdScene.updateFlexCable(0, 0);
  }
  rgbdScene.setFrustumPoses(
    T_cb_gt,
    config.mode === 'modeB' ? initPoseForStage4 : stage2Pose,
    config.mode === 'modeB' ? finalPose : null
  );
  rgbdScene.renderNow();

  const finalErr = computePoseError(finalPose, T_cb_gt);
  const elapsedMs = performance.now() - t0;
  const success = finalErr.transMm < 20.0 && finalErr.rotDeg < 2.0;

  const v3Diagnostics: V3Diagnostics = {
    estimatedDelayMs,
    classPriors,
    classCounts,
    zBufferOccludedPct:
      totalFimInFov > 0 ? (totalFimOccluded / totalFimInFov) * 100 : 0,
    occ2DBandPixels: globalOcc2D.bandPixelCount,
    probeLogs,
    posteriorOverlay,
  };

  traceLogger.log(
    'metric',
    'pipelineV3',
    'runSingleCalibrationV3',
    `[V3] 标定完成: ${finalErr.transMm.toFixed(2)}mm / ${finalErr.rotDeg.toFixed(3)}° (${success ? '合格' : '发散'})`,
    {
      durationMs: elapsedMs,
      output: {
        initTransMm: initErr.transMm,
        initRotDeg: initErr.rotDeg,
        stage2TransMm: stage2Err.transMm,
        stage2RotDeg: stage2Err.rotDeg,
        finalTransMm: finalErr.transMm,
        finalRotDeg: finalErr.rotDeg,
        fimLogDet: fimMetrics.logDet,
        fimLambdaMin: fimMetrics.lambdaMin,
        estimatedDelayMs,
        classPriors,
        classCounts,
        zBufferOccludedPct: v3Diagnostics.zBufferOccludedPct,
        occ2DBandPixels: v3Diagnostics.occ2DBandPixels,
        probeLogs,
        success,
        T_cb_final: finalPose,
      },
    }
  );

  opts?.onLiveUpdate?.({
    phaseText: `[V3] 标定完成 (${finalErr.transMm.toFixed(2)}mm / ${finalErr.rotDeg.toFixed(3)}°)`,
    progress: 1,
    rgb: lastRgb ?? undefined,
    depth: lastDepth ?? undefined,
    mask: lastMask ?? undefined,
    v3PosteriorOverlay: posteriorOverlay,
    syncDiagnostics: lastSyncDiag ?? undefined,
    stage1Pose: config.mode === 'modeB' ? initPoseForStage4 : stage2Pose,
    stage2Pose: config.mode === 'modeB' ? finalPose : null,
    lmTrace: [...lmTrace],
    errorHistory: [...errorHistory],
  });

  return {
    algoVersion: 'v3',
    initMode: config.initMode,
    T_cb_gt,
    T_cb_init: initPoseForStage4,
    T_cb_stage1: stage2Pose,
    T_cb_final: finalPose,
    initRotDeg: initErr.rotDeg,
    initTransMm: initErr.transMm,
    stage1RotDeg: stage2Err.rotDeg,
    stage1TransMm: stage2Err.transMm,
    finalRotDeg: finalErr.rotDeg,
    finalTransMm: finalErr.transMm,
    fimLogDet: fimMetrics.logDet,
    fimLambdaMin: fimMetrics.lambdaMin,
    elapsedMs,
    executedActions,
    screwObservations,
    syncDiagnostics: lastSyncDiag,
    lmTrace,
    errorHistory,
    lastRgb,
    lastDepth,
    lastMask,
    v3Diagnostics,
    success,
  };
}

interface V3AblationSpec {
  id: string;
  name: string;
  shortDesc: string;
  useDualExclusion: boolean;
  override: (base: CalibrationConfig) => CalibrationConfig;
}

export const SIX_V3_ABLATION_CONDITIONS: readonly V3AblationSpec[] = [
  {
    id: 'v3_full',
    name: '1. V3 Full (4阶段退火EM完整版)',
    shortDesc: '双域排斥 + GNC-Welsch + Z-Buf FIM + 退火EM-LM',
    useDualExclusion: true,
    override: (base) => ({
      ...base,
      algoVersion: 'v3',
      mode: 'modeB',
      strategy: 'fim',
      useFlyingPixelFilter: true,
      useVelocityField: true,
      useTimestampSync: true,
      enableDistractor: true,
      enableCustomTool: true,
    }),
  },
  {
    id: 'v3_mode_a',
    name: '2. V3 Mode A (无CAD 广义螺旋轴)',
    shortDesc: '仅 Stage 2 盲探针 + GNC-Welsch 闭式解',
    useDualExclusion: true,
    override: (base) => ({
      ...base,
      algoVersion: 'v3',
      mode: 'modeA',
      strategy: 'fim',
      useFlyingPixelFilter: true,
      useVelocityField: false,
      useTimestampSync: true,
      enableDistractor: true,
      enableCustomTool: true,
    }),
  },
  {
    id: 'v3_wo_dual_exclusion',
    name: '3. w/o 2D+3D 双域静态排斥',
    shortDesc: '关闭 V_static 体素排斥与 M_occ_2D 遮挡带',
    useDualExclusion: false,
    override: (base) => ({
      ...base,
      algoVersion: 'v3',
      mode: 'modeB',
      strategy: 'fim',
      useFlyingPixelFilter: true,
      useVelocityField: true,
      useTimestampSync: true,
      enableDistractor: true,
      enableCustomTool: true,
    }),
  },
  {
    id: 'v3_random_planning',
    name: '4. Random 规划 (vs Z-Buf FIM)',
    shortDesc: '随机候选动作替代 Z-Buffer 无量纲 FIM',
    useDualExclusion: true,
    override: (base) => ({
      ...base,
      algoVersion: 'v3',
      mode: 'modeB',
      strategy: 'random',
      useFlyingPixelFilter: true,
      useVelocityField: true,
      useTimestampSync: true,
      enableDistractor: true,
      enableCustomTool: true,
    }),
  },
  {
    id: 'v3_wo_velocity',
    name: '5. w/o 动态速度场 (关Class II复用)',
    shortDesc: '仅静态退火几何项，关闭 Savitzky-Golay 速度场',
    useDualExclusion: true,
    override: (base) => ({
      ...base,
      algoVersion: 'v3',
      mode: 'modeB',
      strategy: 'fim',
      useFlyingPixelFilter: true,
      useVelocityField: false,
      useTimestampSync: true,
      enableDistractor: true,
      enableCustomTool: true,
    }),
  },
  {
    id: 'v3_wo_multifrag_sync',
    name: '6. w/o 多碎片因果同步聚合',
    shortDesc: '存在异步干扰物且关闭多碎片相关性滤波',
    useDualExclusion: true,
    override: (base) => ({
      ...base,
      algoVersion: 'v3',
      mode: 'modeB',
      strategy: 'fim',
      useFlyingPixelFilter: true,
      useVelocityField: true,
      useTimestampSync: false,
      enableDistractor: true,
      enableCustomTool: true,
    }),
  },
];

function computeMeanStd(vals: readonly number[]): { mean: number; std: number } {
  if (vals.length === 0) return { mean: 0, std: 0 };
  const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
  const variance =
    vals.length > 1
      ? vals.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / (vals.length - 1)
      : 0;
  return { mean, std: Math.sqrt(Math.max(0, variance)) };
}

function computeMedian(vals: readonly number[]): number {
  if (vals.length === 0) return 0;
  const sorted = [...vals].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
}

/**
 * Run V3 6-Condition Paired Monte Carlo Ablation Benchmark on identical random GT camera poses.
 */
export async function runPairedAblationBenchmarkV3(
  engine: MujocoPandaEngine,
  rgbdScene: PandaRGBDScene,
  baseConfig: CalibrationConfig,
  opts: {
    baseSeed?: number;
    onProgress: (info: {
      text: string;
      progress: number;
      rows: AblationRow[];
      latestResult?: CalibrationResult;
    }) => void;
    shouldCancel: () => boolean;
  }
): Promise<AblationRow[]> {
  const nTrials = Math.max(3, Math.min(20, baseConfig.mcTrials));
  const poseRng = new SeededRNG(opts.baseSeed ?? 20261126);

  const pairedPoses: SE3[] = [];
  for (let t = 0; t < nTrials; t++) {
    pairedPoses.push(sampleRandomCameraPose(poseRng, engine.spec));
  }

  const totalRuns = SIX_V3_ABLATION_CONDITIONS.length * nTrials;
  let completedRuns = 0;
  const rows: AblationRow[] = [];

  for (let cIdx = 0; cIdx < SIX_V3_ABLATION_CONDITIONS.length; cIdx++) {
    if (opts.shouldCancel()) break;
    const spec = SIX_V3_ABLATION_CONDITIONS[cIdx];
    const condConfig = spec.override(baseConfig);

    const rotDegs: number[] = [];
    const transMms: number[] = [];
    const timesMs: number[] = [];
    const fimLogs: number[] = [];
    let successCnt = 0;

    for (let t = 0; t < nTrials; t++) {
      if (opts.shouldCancel()) break;
      const trialSeed = 10000 + t * 997 + cIdx * 31;

      opts.onProgress({
        text: `[V3 消融] [${cIdx + 1}/6] ${spec.name} · Trial ${t + 1}/${nTrials}`,
        progress: completedRuns / totalRuns,
        rows: [...rows],
      });
      await nextAnimationFrame();

      const res = await runSingleCalibrationV3(
        engine,
        rgbdScene,
        pairedPoses[t],
        condConfig,
        trialSeed,
        {
          shouldCancel: opts.shouldCancel,
          fastMode: true,
          useDualExclusion: spec.useDualExclusion,
        }
      );

      rotDegs.push(res.finalRotDeg);
      transMms.push(res.finalTransMm);
      timesMs.push(res.elapsedMs);
      fimLogs.push(res.fimLogDet);
      if (res.success) successCnt++;
      completedRuns++;

      opts.onProgress({
        text: `[V3 消融] [${cIdx + 1}/6] ${spec.name} · 完成 Trial ${t + 1}/${nTrials} (${res.finalTransMm.toFixed(1)}mm / ${res.finalRotDeg.toFixed(2)}°)`,
        progress: completedRuns / totalRuns,
        rows: [...rows],
        latestResult: res,
      });
      await nextAnimationFrame();
    }

    if (rotDegs.length > 0) {
      const rotStat = computeMeanStd(rotDegs);
      const transStat = computeMeanStd(transMms);
      const timeStat = computeMeanStd(timesMs);
      const fimStat = computeMeanStd(fimLogs);

      const row: AblationRow = {
        id: spec.id,
        name: spec.name,
        shortDesc: spec.shortDesc,
        trials: rotDegs.length,
        rotMeanDeg: rotStat.mean,
        rotStdDeg: rotStat.std,
        transMeanMm: transStat.mean,
        transStdMm: transStat.std,
        transMedianMm: computeMedian(transMms),
        successRate: (successCnt / rotDegs.length) * 100,
        meanTimeMs: timeStat.mean,
        fimLogDetMean: fimStat.mean,
        trialRotDegs: rotDegs,
        trialTransMms: transMms,
      };
      rows.push(row);

      traceLogger.log('metric', 'ablationV3', 'conditionComplete', `[V3 消融] 完成条件 ${cIdx + 1}/6: ${spec.name}`, {
        output: row,
      });

      opts.onProgress({
        text: `[V3 消融] 已完成条件 ${cIdx + 1}/6: ${spec.name}`,
        progress: completedRuns / totalRuns,
        rows: [...rows],
      });
    }
  }

  return rows;
}
