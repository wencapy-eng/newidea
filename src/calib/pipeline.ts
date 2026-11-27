import {
  cloneSE3,
  computePoseError,
  expSO3,
  identitySE3,
  lookAtCV,
  mat3Mul,
  SeededRNG,
  SpatialHashGrid3D,
  type Mat3,
  type PointNormalSample,
  type SE3,
  type Vec3,
  type Vec6,
} from '../math/se3';
import type { MujocoPandaEngine } from '../sim/mujocoEngine';
import type { PandaRGBDScene } from '../sim/rgbdScene';
import { getRobotSpec, type RobotSpec } from '../sim/robotCatalog';
import {
  addMat6InPlace,
  buildColdStartActions,
  buildSafeCandidatePool,
  computeActionFIM,
  evaluateFIMMetrics,
  selectNextAction,
} from './fimPlanner';
import {
  BG_DIFF_THRESHOLD_M,
  computeMaxDepthEnvelope,
  computeOneSidedForegroundMask,
  extractWeightedPointCloud,
  filterMaskSmallComponents,
  segmentActionWithTimestampSync,
} from './frontend';
import {
  extractDynamicVelocityObservations,
  iterateStage2ManifoldLM,
  runStage2ManifoldLM,
  type DynamicVelObs,
  type StaticGeoFrameObs,
} from './manifoldStage2';
import {
  extractParallelTranslationObservationFromSweep,
  extractScrewObservationFromSweep,
  solveParallelHandEyeClosedForm,
  solveScrewHandEyeClosedForm,
} from './screwStage1';
import { appendErrorHistoryPoint } from '../utils/errorHistory';
import { traceLogger } from '../utils/traceLogger';
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
} from './types';

export const FRAMES_PER_ACTION = 16;
export const PRE_PAUSE_FRAMES = 3;
export const SWEEP_FRAMES = 10;
export const POST_PAUSE_FRAMES = 3;
export const FRAME_DT_SEC = 0.065; // ~15.4 Hz capture rate

/** Yield to browser animation frame so UI and 3D canvas remain completely responsive */
export function nextAnimationFrame(): Promise<void> {
  return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

/**
 * Sample a realistic front-hemisphere ground-truth RealSense D435i pose T_cb_gt
 * tailored to the active robot's workspace bounds & viewing center.
 */
export function sampleRandomCameraPose(rng: SeededRNG, spec?: RobotSpec): SE3 {
  const [dMin, dMax] = spec?.cameraDistRange ?? [1.18, 1.48];
  const baseTarget = spec?.cameraTarget ?? [0.08, 0.0, 0.42];
  const dist = rng.uniform(dMin, dMax);
  const az = rng.uniform((-42 * Math.PI) / 180, (42 * Math.PI) / 180);
  const el = rng.uniform((22 * Math.PI) / 180, (41 * Math.PI) / 180);

  const cosEl = Math.cos(el);
  const camPos: Vec3 = [
    baseTarget[0] + dist * cosEl * Math.cos(az),
    baseTarget[1] + dist * cosEl * Math.sin(az),
    Math.max(0.42, baseTarget[2] - 0.16 + dist * Math.sin(el)),
  ];
  const target: Vec3 = [
    baseTarget[0] + rng.uniform(-0.04, 0.04),
    baseTarget[1] + rng.uniform(-0.04, 0.04),
    baseTarget[2] + rng.uniform(-0.04, 0.04),
  ];
  const roll = rng.uniform((-7 * Math.PI) / 180, (7 * Math.PI) / 180);
  return lookAtCV(camPos, target, roll);
}

/** Uniform random unit vector on the 2-sphere S^2 */
function sampleUnitSphere(rng: SeededRNG): Vec3 {
  const z = rng.uniform(-1, 1);
  const phi = rng.uniform(0, 2 * Math.PI);
  const r = Math.sqrt(Math.max(0, 1 - z * z));
  return [r * Math.cos(phi), r * Math.sin(phi), z];
}

/** Exact Shoemake (1992) Haar-uniform random rotation matrix R in SO(3) */
function sampleUniformSO3(rng: SeededRNG): Mat3 {
  const u1 = rng.next();
  const u2 = 2 * Math.PI * rng.next();
  const u3 = 2 * Math.PI * rng.next();
  const s1 = Math.sqrt(Math.max(0, 1 - u1));
  const s2 = Math.sqrt(Math.max(0, u1));
  const qw = s1 * Math.sin(u2);
  const qx = s1 * Math.cos(u2);
  const qy = s2 * Math.sin(u3);
  const qz = s2 * Math.cos(u3);

  const xx = qx * qx, yy = qy * qy, zz = qz * qz;
  const xy = qx * qy, xz = qx * qz, yz = qy * qz;
  const wx = qw * qx, wy = qw * qy, wz = qw * qz;

  return [
    1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
    2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
    2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
  ];
}

/** Human-readable Chinese label for each initial extrinsic mode */
export function getInitModeLabel(
  mode: ExtrinsicInitMode,
  rotErrDeg?: number,
  transErrMm?: number
): string {
  switch (mode) {
    case 'stage1':
      return 'Stage 1 螺旋轴闭式解';
    case 'controlled_random':
      return rotErrDeg !== undefined && transErrMm !== undefined
        ? `受控随机 (${rotErrDeg.toFixed(0)}° / ${transErrMm.toFixed(0)}mm)`
        : '受控随机初始化';
    case 'full_random':
      return '完全随机初始化 (SO(3)×R³)';
    case 'zero':
      return '0 初始化 (ξ⁽⁰⁾=0, I₄)';
  }
}

/**
 * Generate an external initial hand-eye extrinsic T_cb^(0) for benchmarking Stage 2 convergence basins:
 *  - 'zero':              Lie-algebra zero initialization xi^(0) = 0 => T_cb^(0) = I_4 (R = I_3, t = 0)
 *  - 'controlled_random': Exact geodesic rotation perturbation (rotErrDeg) + exact Euclidean translation
 *                         perturbation (transErrMm) along uniform S^2 directions around T_cb_gt
 *  - 'full_random':       Unguided Haar-uniform random R in SO(3) + wide random t in R^3 (zero GT prior)
 */
export function generateInitialHandEyePose(
  mode: Exclude<ExtrinsicInitMode, 'stage1'>,
  T_cb_gt: SE3,
  rng: SeededRNG,
  opts?: { rotErrDeg?: number; transErrMm?: number }
): SE3 {
  if (mode === 'zero') {
    return identitySE3();
  }

  if (mode === 'controlled_random') {
    const rotRad = (Math.max(0, opts?.rotErrDeg ?? 15) * Math.PI) / 180;
    const transM = Math.max(0, opts?.transErrMm ?? 120) / 1000;
    const uRot = sampleUnitSphere(rng);
    const uTrans = sampleUnitSphere(rng);
    const dR = expSO3([uRot[0] * rotRad, uRot[1] * rotRad, uRot[2] * rotRad]);
    const R_init = mat3Mul(T_cb_gt.R, dR);
    const t_init: Vec3 = [
      T_cb_gt.t[0] + uTrans[0] * transM,
      T_cb_gt.t[1] + uTrans[1] * transM,
      T_cb_gt.t[2] + uTrans[2] * transM,
    ];
    return { R: R_init, t: t_init };
  }

  // 'full_random': completely blind Haar-uniform SO(3) rotation + wide 3D translation
  const R_rand = sampleUniformSO3(rng);
  const t_rand: Vec3 = [
    rng.uniform(-1.2, 1.2),
    rng.uniform(-1.2, 1.2),
    rng.uniform(0.25, 2.5),
  ];
  return { R: R_rand, t: t_rand };
}

/**
 * Evaluate continuous smooth single-joint trajectory:
 *  - Pre-pause: 3 frames (t in [0, 2*dt])
 *  - Cosine S-curve sweep: 10 interior frames (t in [2*dt, 13*dt])
 *  - Post-pause: 3 frames (t in [13*dt, 15*dt])
 */
export function evaluateActionTrajectory(
  action: MotionAction,
  tLocalSec: number
): { q: number[]; qdot: number[] } {
  const tStart = (PRE_PAUSE_FRAMES - 1) * FRAME_DT_SEC; // 2 * dt
  const tEnd = (PRE_PAUSE_FRAMES + SWEEP_FRAMES) * FRAME_DT_SEC; // 13 * dt
  const T_sweep = tEnd - tStart; // 11 * dt

  const q = [...action.qBase];
  const qdot = new Array<number>(action.qBase.length).fill(0);
  const dq = action.qEnd - action.qStart;

  if (tLocalSec <= tStart) {
    q[action.jointIndex] = action.qStart;
  } else if (tLocalSec >= tEnd) {
    q[action.jointIndex] = action.qEnd;
  } else {
    const tau = (tLocalSec - tStart) / T_sweep;
    const s = 0.5 * (1 - Math.cos(Math.PI * tau));
    const dsdt = (Math.PI / (2 * T_sweep)) * Math.sin(Math.PI * tau);
    q[action.jointIndex] = action.qStart + dq * s;
    qdot[action.jointIndex] = dq * dsdt;
  }

  return { q, qdot };
}

/** Asynchronous sinusoidal distractor position (uncorrelated with robot S-curve pulses) */
export function evaluateDistractorPos(tGlobalSec: number, phaseOffset: number): number {
  return 0.26 * Math.cos(2 * Math.PI * 0.92 * tGlobalSec + phaseOffset);
}

export interface LiveStepUpdate {
  phaseText: string;
  progress: number; // 0..1
  rgb?: Uint8ClampedArray;
  depth?: Float32Array;
  mask?: Uint8Array;
  v3PosteriorOverlay?: Uint8Array | null;
  syncDiagnostics?: SyncDiagnostics;
  stage1Pose?: SE3 | null;
  stage2Pose?: SE3 | null;
  lmTrace?: LMTracePoint[];
  errorHistory?: ErrorHistoryPoint[];
}

/**
 * Execute one complete real-physics, real-WebGL2 active hand-eye calibration run.
 */
export async function runSingleCalibration(
  engine: MujocoPandaEngine,
  rgbdScene: PandaRGBDScene,
  T_cb_gt: SE3,
  config: CalibrationConfig,
  seed: number,
  opts?: {
    onLiveUpdate?: (upd: LiveStepUpdate) => void;
    shouldCancel?: () => boolean;
    fastMode?: boolean; // Fewer visual yields during Monte Carlo ablation while preserving 100% identical physics/math
    /** Optional pre-sampled external initial extrinsic (matches 3D idle preview when initMode !== 'stage1') */
    externalInitPose?: SE3 | null;
  }
): Promise<CalibrationResult> {
  const t0 = performance.now();
  const rng = new SeededRNG(seed);
  const K = rgbdScene.getIntrinsics();
  const fastMode = opts?.fastMode ?? false;
  const useExternalInit = config.initMode !== 'stage1';

  traceLogger.log(
    'stage',
    'pipelineV2',
    'runSingleCalibration',
    `[V2] 启动 ${engine.spec.shortName} 标定 (${config.mode} / ${config.strategy} / ${config.initMode})`,
    {
      input: {
        robotId: config.robotId,
        mode: config.mode,
        strategy: config.strategy,
        initMode: config.initMode,
        numSteps: config.numSteps,
        seed,
        fastMode,
        T_cb_gt,
        sensor: config.sensor,
      },
    }
  );

  // Generate or reuse external initial pose T_cb^(0) when user selects zero / controlled_random / full_random
  const externalInitPose: SE3 | null = useExternalInit
    ? opts?.externalInitPose
      ? cloneSE3(opts.externalInitPose)
      : traceLogger.traceCall(
          'pipelineV2',
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

  // Apply scene visibility (unknown custom tool, dynamic distractor, V3 stress pillar & cable)
  rgbdScene.updateFrontPillarForCamera(T_cb_gt);
  rgbdScene.setSceneVisibility({
    showCustomTool: config.enableCustomTool,
    showDistractor: config.enableDistractor,
    showFrontPillar: config.enableFrontPillar,
    showFlexCable: config.enableFlexCable,
  });
  rgbdScene.setFrustumPoses(T_cb_gt, externalInitPose, null);

  const coldStart = traceLogger.traceCall(
    'fim',
    'buildColdStartActions',
    { robotId: engine.spec.id, dof: engine.spec.dof },
    () => buildColdStartActions(engine),
    (acts) => ({ count: acts.length, ids: acts.map((a) => a.id) })
  );
  const candidatePool = traceLogger.traceCall(
    'fim',
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
  const globalRejectedMask = new Uint8Array(K.width * K.height);

  let I_acc = new Float64Array(36);
  let T_cb_stage1: SE3 | null = externalInitPose ? cloneSE3(externalInitPose) : null;
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
        phaseText: `初始外参就绪 (${extErr.transMm.toFixed(1)}mm / ${extErr.rotDeg.toFixed(2)}°)`,
        progress: 0.03,
        stage1Pose: T_cb_stage1,
        errorHistory: [...errorHistory],
      });
    }
  }
  let lastSyncDiag: SyncDiagnostics | null = null;
  let lastRgb: Uint8ClampedArray | null = null;
  let lastDepth: Float32Array | null = null;
  let lastMask: Uint8Array | null = null;

  const distractorPhase = rng.uniform(0, 2 * Math.PI);
  let tGlobal = 0;

  // Helper to capture one 16-frame action sequence and process Stage 1 screw observation
  const executeOneAction = async (action: MotionAction, stepIdx: number) => {
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

      // True physical pose at camera shutter timestamp t
      const { q: qTrue } = evaluateActionTrajectory(action, tLocal);
      const distPos = config.enableDistractor
        ? evaluateDistractorPos(tCurGlobal, distractorPhase)
        : 0;

      // Encoder reading delayed by asyncDelayMs: evaluated at (t - asyncDelay)
      const { q: qEncoder, qdot: qdotEncoder } = evaluateActionTrajectory(
        action,
        Math.max(0, tLocal - delaySec)
      );

      // Step MuJoCo forward kinematics & sync Three.js meshes
      engine.setState(qTrue, distPos);
      rgbdScene.syncFromMujoco();
      if (config.enableFlexCable) {
        const qdotNorm = Math.abs(qdotEncoder[action.jointIndex] ?? 0);
        rgbdScene.updateFlexCable(tCurGlobal, qdotNorm);
      }

      // Dual-pass + material G-buffer WebGL2 offscreen capture of virtual RealSense D435i RGB-D
      const { rgb, depth } = rgbdScene.captureVirtualRGBD(T_cb_gt, config.sensor, rng);

      // Record theoretical effective spatial twist at encoder configuration (never peeking at qTrue!)
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
        // Restore true state for viewport render
        engine.setState(qTrue, distPos);
        rgbdScene.syncFromMujoco();
        rgbdScene.renderNow();
        opts?.onLiveUpdate?.({
          phaseText: `步骤 ${stepIdx + 1}/${totalSteps}: ${action.label} (帧 ${f + 1}/${FRAMES_PER_ACTION})`,
          progress: (stepIdx + (f + 1) / FRAMES_PER_ACTION) / (totalSteps + 1.2),
          rgb,
          depth,
          stage1Pose: T_cb_stage1,
        });
        await nextAnimationFrame();
      }
    }

    tGlobal += FRAMES_PER_ACTION * FRAME_DT_SEC + 0.15;

    if (!fastMode) {
      traceLogger.recordCall(
        'sim',
        'captureActionSequence',
        {
          stepIdx,
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
        `采集动作序列 ${action.label} (${frames.length} 帧 RGB-D)`
      );
    }

    // Layer 1 unsupervised foreground segmentation + timestamp sync correlation filtering
    const seg = !fastMode
      ? traceLogger.traceCall(
          'frontend',
          'segmentActionWithTimestampSync',
          {
            actionLabel: action.label,
            frameCount: frames.length,
            useTimestampSync: config.useTimestampSync,
          },
          () => segmentActionWithTimestampSync(frames, action.label, config.useTimestampSync),
          (res) => ({
            diagnostics: res.diagnostics,
            midFrameMaskPx: res.frameMasks[Math.floor(res.frameMasks.length / 2)]?.reduce(
              (acc, v) => acc + (v ? 1 : 0),
              0
            ),
          })
        )
      : segmentActionWithTimestampSync(frames, action.label, config.useTimestampSync);
    lastSyncDiag = seg.diagnostics;

    if (config.useTimestampSync) {
      for (let i = 0; i < globalRejectedMask.length; i++) {
        if (seg.rejectedRegionMask[i]) globalRejectedMask[i] = 1;
      }
    }

    allActionFrames.push(frames);
    allActionMasks.push(seg.frameMasks);
    allActionXiBase.push(xiBasePerFrame);

    // Mid-sweep visual snapshot for diagnostic overlay
    const midIdx = Math.floor(FRAMES_PER_ACTION / 2);
    lastRgb = frames[midIdx].rgb;
    lastDepth = frames[midIdx].depth;
    lastMask = seg.frameMasks[midIdx];

    // Extract weighted point clouds across the sweep frames [PRE_PAUSE-1 .. PRE_PAUSE+SWEEP]
    const startIdx = PRE_PAUSE_FRAMES - 1; // frame 2 (just before motion)
    const endIdx = PRE_PAUSE_FRAMES + SWEEP_FRAMES; // frame 13 (just after motion)
    const sweepClouds = !fastMode
      ? traceLogger.traceCall(
          'frontend',
          'extractWeightedPointCloudBatch',
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
              return extractWeightedPointCloud(fr.depth, seg.frameMasks[idx], K, {
                filterEnabled: config.useFlyingPixelFilter,
                sigmaZ0: config.sensor.sigmaZ0,
                maxPoints: 760,
                pixelStride: 1,
              });
            }),
          (clouds) => ({
            perFramePointCounts: clouds.map((c) => c.length),
            totalPoints: clouds.reduce((s, c) => s + c.length, 0),
          })
        )
      : frames.map((fr, idx) => {
          if (idx < startIdx || idx > endIdx) return [];
          return extractWeightedPointCloud(fr.depth, seg.frameMasks[idx], K, {
            filterEnabled: config.useFlyingPixelFilter,
            sigmaZ0: config.sensor.sigmaZ0,
            maxPoints: 760,
            pixelStride: 1,
          });
        });

    // Encoder deltaQ from static pause endpoints
    const qStartEnc = frames[startIdx].qEncoder[action.jointIndex];
    const qEndEnc = frames[endIdx].qEncoder[action.jointIndex];
    const deltaQ = qEndEnc - qStartEnc;
    const xiBaseMid = xiBasePerFrame[midIdx];
    const frameQEnc = frames.map((fr) => fr.qEncoder[action.jointIndex]);

    const isDeltaTranslationSweep =
      engine.spec.mechanism === 'parallel_delta' && action.jointIndex < 3;

    let screwObs: ScrewObservation | null = null;
    if (isDeltaTranslationSweep) {
      const pStartB = engine.getDeltaPlatformPosBase(frames[startIdx].qEncoder);
      const pEndB = engine.getDeltaPlatformPosBase(frames[endIdx].qEncoder);
      const pMidB = engine.getDeltaPlatformPosBase(frames[midIdx].qEncoder);
      if (pStartB && pEndB && pMidB) {
        const deltaPBase: Vec3 = [
          pEndB[0] - pStartB[0],
          pEndB[1] - pStartB[1],
          pEndB[2] - pStartB[2],
        ];
        screwObs = !fastMode
          ? traceLogger.traceCall(
              'screw',
              'extractParallelTranslationObservationFromSweep',
              { actionId: action.id, deltaPBase, pMidB, deltaQ },
              () =>
                extractParallelTranslationObservationFromSweep(
                  action,
                  sweepClouds,
                  startIdx,
                  endIdx,
                  deltaPBase,
                  pMidB,
                  deltaQ
                )
            )
          : extractParallelTranslationObservationFromSweep(
              action,
              sweepClouds,
              startIdx,
              endIdx,
              deltaPBase,
              pMidB,
              deltaQ
            );
      }
    } else {
      screwObs = !fastMode
        ? traceLogger.traceCall(
            'screw',
            'extractScrewObservationFromSweep',
            { actionId: action.id, xiBaseMid, deltaQ, startIdx, endIdx },
            () =>
              extractScrewObservationFromSweep(
                action,
                sweepClouds,
                startIdx,
                endIdx,
                xiBaseMid,
                deltaQ,
                frameQEnc
              )
          )
        : extractScrewObservationFromSweep(
            action,
            sweepClouds,
            startIdx,
            endIdx,
            xiBaseMid,
            deltaQ,
            frameQEnc
          );
    }

    if (screwObs) {
      screwObservations.push(screwObs);
    }

    const solveStage1 = (obsList: readonly ScrewObservation[]): SE3 =>
      engine.spec.mechanism === 'parallel_delta'
        ? solveParallelHandEyeClosedForm(obsList)
        : solveScrewHandEyeClosedForm(obsList);

    if (!externalInitPose && screwObservations.length >= 3) {
      T_cb_stage1 = !fastMode
        ? traceLogger.traceCall(
            'screw',
            engine.spec.mechanism === 'parallel_delta'
              ? 'solveParallelHandEyeClosedForm'
              : 'solveScrewHandEyeClosedForm',
            { obsCount: screwObservations.length, stepIdx },
            () => solveStage1(screwObservations),
            (pose) => ({ pose, error: computePoseError(pose, T_cb_gt) })
          )
        : solveStage1(screwObservations);
      const stepErr = computePoseError(T_cb_stage1, T_cb_gt);
      errorHistory = appendErrorHistoryPoint(errorHistory, {
        label: `S${stepIdx + 1}`,
        phase: 'closed_form',
        transErrMm: stepErr.transMm,
        rotErrDeg: stepErr.rotDeg,
      });
    }

    opts?.onLiveUpdate?.({
      phaseText: `完成步骤 ${stepIdx + 1}/${totalSteps}: ${action.label}`,
      progress: (stepIdx + 1) / (totalSteps + 1.2),
      rgb: lastRgb,
      depth: lastDepth,
      mask: lastMask,
      syncDiagnostics: lastSyncDiag,
      stage1Pose: T_cb_stage1,
      errorHistory: [...errorHistory],
    });
    await nextAnimationFrame();
  };

  // Phase A: Steps 1..3 Fixed Orthogonal Major-Axis Cold Start
  const coldCount = Math.min(3, coldStart.length, totalSteps);
  for (let i = 0; i < coldCount; i++) {
    if (opts?.shouldCancel?.()) break;
    await executeOneAction(coldStart[i], i);
  }

  const solveStage1All = (obsList: readonly ScrewObservation[]): SE3 =>
    engine.spec.mechanism === 'parallel_delta'
      ? solveParallelHandEyeClosedForm(obsList)
      : solveScrewHandEyeClosedForm(obsList);

  // Solve Stage 1 closed-form pose (used as initializer in 'stage1' mode and as output in Mode A)
  const solvedStage1 = solveStage1All(screwObservations);
  const activePlanningPose = externalInitPose ?? solvedStage1;
  T_cb_stage1 = activePlanningPose;

  // Initialize accumulated FIM I_acc from executed cold-start actions using current initial pose estimate
  I_acc = new Float64Array(36);
  for (let i = 0; i < executedActions.length; i++) {
    const fim = computeActionFIM(
      engine,
      executedActions[i],
      T_cb_stage1,
      K,
      config.sensor.sigmaZ0,
      config.useVelocityField
    );
    addMat6InPlace(I_acc, fim);
  }

  // Phase B: Steps 4..N Online Active Exploration (FIM D/E-Optimal or Random Baseline)
  for (let step = coldCount; step < totalSteps; step++) {
    if (opts?.shouldCancel?.()) break;
    const pick = !fastMode
      ? traceLogger.traceCall(
          'fim',
          'selectNextAction',
          {
            step,
            strategy: config.strategy,
            remainingCandidates: candidatePool.length - usedIds.size,
            useVelocityField: config.useVelocityField,
          },
          () =>
            selectNextAction(
              engine,
              candidatePool,
              usedIds,
              config.strategy,
              I_acc,
              T_cb_stage1!,
              K,
              config.sensor.sigmaZ0,
              config.useVelocityField,
              rng
            ),
          (res) =>
            res
              ? {
                  selectedId: res.action.id,
                  label: res.action.label,
                  actionFimDiagSum:
                    res.actionFIM[0] +
                    res.actionFIM[7] +
                    res.actionFIM[14] +
                    res.actionFIM[21] +
                    res.actionFIM[28] +
                    res.actionFIM[35],
                }
              : null
        )
      : selectNextAction(
          engine,
          candidatePool,
          usedIds,
          config.strategy,
          I_acc,
          T_cb_stage1,
          K,
          config.sensor.sigmaZ0,
          config.useVelocityField,
          rng
        );
    if (!pick) break;

    await executeOneAction(pick.action, step);

    // Update T_cb_stage1 with all collected screw/translation observations & accumulate FIM
    if (!externalInitPose && screwObservations.length >= 3) {
      T_cb_stage1 = solveStage1All(screwObservations);
    }
    const updatedFim = computeActionFIM(
      engine,
      pick.action,
      T_cb_stage1,
      K,
      config.sensor.sigmaZ0,
      config.useVelocityField
    );
    addMat6InPlace(I_acc, updatedFim);
  }

  const stage1Pose = solveStage1All(screwObservations);
  const stage1Err = computePoseError(stage1Pose, T_cb_gt);
  const initPoseForStage2 = externalInitPose ? cloneSE3(externalInitPose) : cloneSE3(stage1Pose);
  const initErr = computePoseError(initPoseForStage2, T_cb_gt);
  const fimMetrics = evaluateFIMMetrics(I_acc);

  // Phase C: Stage 2 SE(3) Manifold Joint Optimization (Mode B only)
  let finalPose = config.mode === 'modeB' ? cloneSE3(initPoseForStage2) : cloneSE3(stage1Pose);
  let lmTrace = [
    {
      iter: 0,
      stage: 0,
      cost: 0,
      rotErrDeg: config.mode === 'modeB' ? initErr.rotDeg : stage1Err.rotDeg,
      transErrMm: config.mode === 'modeB' ? initErr.transMm : stage1Err.transMm,
      geoCount: 0,
      velCount: 0,
      lambda: 0,
    },
  ];

  if (config.mode === 'modeB' && !opts?.shouldCancel?.()) {
    opts?.onLiveUpdate?.({
      phaseText: 'Stage 2: 全局背景剥离 + 静态点到面与动态速度场 SE(3) 流形 LM 联合优化...',
      progress: (totalSteps + 0.4) / (totalSteps + 1.2),
      stage1Pose,
    });
    await nextAnimationFrame();

    // Layer 2 Global Static Max-Depth Envelope D_bg(u, v) = max_{all t} D_t(u, v)
    const flatFrames: RGBDFrame[] = [];
    for (let a = 0; a < allActionFrames.length; a++) {
      const af = allActionFrames[a];
      for (let f = 0; f < af.length; f++) flatFrames.push(af[f]);
    }
    const globalBgDepth = computeMaxDepthEnvelope(flatFrames, K.width, K.height);

    // Build static geometric observations from pause frames (pre-pause frame 1 & post-pause frame 14 of each action)
    const staticObs: StaticGeoFrameObs[] = [];
    for (let a = 0; a < allActionFrames.length; a++) {
      const af = allActionFrames[a];
      const pauseIndices = [1, FRAMES_PER_ACTION - 2];
      for (const pIdx of pauseIndices) {
        const fr = af[pIdx];
        if (!fr) continue;
        const rawMask = computeOneSidedForegroundMask(
          fr.depth,
          globalBgDepth,
          BG_DIFF_THRESHOLD_M
        );
        // When timestamp sync is enabled, remove regions identified as asynchronous distractors
        if (config.useTimestampSync) {
          for (let i = 0; i < rawMask.length; i++) {
            if (globalRejectedMask[i]) rawMask[i] = 0;
          }
        }
        // Strip scattered negative stereo outlier speckles (<40 px) on static background walls/table
        const mask = config.useFlyingPixelFilter
          ? filterMaskSmallComponents(rawMask, K.width, K.height, 40)
          : rawMask;

        const pts = extractWeightedPointCloud(fr.depth, mask, K, {
          filterEnabled: config.useFlyingPixelFilter,
          sigmaZ0: config.sensor.sigmaZ0,
          maxPoints: 600,
          pixelStride: 1,
        });
        if (pts.length < 30) continue;

        // Build base-frame 3D SpatialHashGrid from official Panda CAD surface (strictly excluding custom_tool_*)
        const cadSamples = engine.getRobotSurfacePointsBase(fr.qEncoder, undefined, 1);
        const modelGrid = new SpatialHashGrid3D<PointNormalSample>(0.045);
        modelGrid.insertAll(cadSamples);

        staticObs.push({
          qEncoder: fr.qEncoder,
          points: pts,
          modelGrid,
        });
      }
    }

    // Build dynamic normal velocity field observations across all executed sweeps
    const velObsAll: DynamicVelObs[] = [];
    if (config.useVelocityField) {
      for (let a = 0; a < allActionFrames.length; a++) {
        const obs = extractDynamicVelocityObservations(
          allActionFrames[a],
          allActionMasks[a],
          allActionXiBase[a],
          K,
          {
            filterEnabled: config.useFlyingPixelFilter,
            sigmaZ0: config.sensor.sigmaZ0,
            maxObsPerAction: 420,
          }
        );
        for (let i = 0; i < obs.length; i++) velObsAll.push(obs[i]);
      }
    }

    // Select coarse-to-fine gate schedule based solely on configured initMode (zero GT leakage)
    let stage2Gates: number[] = [0.13, 0.08, 0.045];
    let stage2Iters: number[] = [8, 7, 10];
    if (config.initMode === 'controlled_random') {
      const coarseGate = Math.max(0.16, Math.min(0.45, (config.initTransErrMm / 1000) * 1.35 + 0.08));
      stage2Gates = [coarseGate, Math.max(0.11, coarseGate * 0.56), 0.08, 0.045];
      stage2Iters = [10, 9, 8, 10];
    } else if (config.initMode === 'full_random' || config.initMode === 'zero') {
      stage2Gates = [0.45, 0.25, 0.12, 0.055];
      stage2Iters = [10, 9, 8, 10];
    }

    const stage2Opts = {
      useVelocityField: config.useVelocityField,
      lambdaVel: 0.0025,
      huberDeltaGeo: 0.014,
      huberDeltaVel: 0.045,
      gates: stage2Gates,
      itersPerStage: stage2Iters,
      T_cb_gt_for_log: T_cb_gt,
    };

    if (fastMode) {
      const stage2Res = traceLogger.traceCall(
        'solver',
        'runStage2ManifoldLM',
        {
          staticFrames: staticObs.length,
          velObsCount: velObsAll.length,
          gates: stage2Gates,
          itersPerStage: stage2Iters,
          initErr,
        },
        () => runStage2ManifoldLM(initPoseForStage2, staticObs, velObsAll, stage2Opts),
        (res) => ({
          iters: res.trace.length,
          finalErr: computePoseError(res.T_est, T_cb_gt),
          lastPoint: res.trace[res.trace.length - 1],
        })
      );
      finalPose = stage2Res.T_est;
      lmTrace = stage2Res.trace;
      for (let i = 0; i < lmTrace.length; i++) {
        const pt = lmTrace[i];
        errorHistory = appendErrorHistoryPoint(errorHistory, {
          label: `LM${pt.iter}`,
          phase: 'lm',
          transErrMm: pt.transErrMm,
          rotErrDeg: pt.rotErrDeg,
          stage: pt.stage,
        });
      }
    } else {
      traceLogger.log(
        'stage',
        'solver',
        'iterateStage2ManifoldLM',
        `Stage 2 流形 LM 优化开始 (${staticObs.length} 静态帧 / ${velObsAll.length} 速度观测)`,
        {
          input: {
            staticFrameCount: staticObs.length,
            staticPointTotal: staticObs.reduce((s, f) => s + f.points.length, 0),
            velObsCount: velObsAll.length,
            gates: stage2Gates,
            itersPerStage: stage2Iters,
            initPose: initPoseForStage2,
            initErr,
          },
        }
      );
      const totalPlannedIters = stage2Iters.reduce((s, n) => s + n, 0);
      const gen = iterateStage2ManifoldLM(initPoseForStage2, staticObs, velObsAll, stage2Opts);
      let iterT0 = performance.now();
      let iterRes = gen.next();
      let liveTrace: LMTracePoint[] = [];
      let livePose = initPoseForStage2;
      while (!iterRes.done) {
        if (opts?.shouldCancel?.()) break;
        const step = iterRes.value;
        const iterDur = performance.now() - iterT0;
        livePose = step.T_est;
        liveTrace = step.trace;
        traceLogger.recordCall(
          'solver',
          'stage2LM_iter',
          { iter: step.point.iter, stage: step.point.stage, lambda: step.point.lambda },
          {
            cost: step.point.cost,
            geoCount: step.point.geoCount,
            velCount: step.point.velCount,
            transErrMm: step.point.transErrMm,
            rotErrDeg: step.point.rotErrDeg,
          },
          iterDur,
          `LM Iter ${step.point.iter} (Stage ${step.point.stage}): ${step.point.transErrMm.toFixed(2)}mm / ${step.point.rotErrDeg.toFixed(3)}°`
        );
        errorHistory = appendErrorHistoryPoint(errorHistory, {
          label: `LM${step.point.iter}`,
          phase: 'lm',
          transErrMm: step.point.transErrMm,
          rotErrDeg: step.point.rotErrDeg,
          stage: step.point.stage,
        });
        const frac = Math.min(1, step.point.iter / Math.max(1, totalPlannedIters));
        rgbdScene.setFrustumPoses(T_cb_gt, initPoseForStage2, livePose);
        rgbdScene.renderNow();
        opts?.onLiveUpdate?.({
          phaseText: `Stage 2 LM Iter ${step.point.iter} · ${step.point.transErrMm.toFixed(2)} mm / ${step.point.rotErrDeg.toFixed(3)}°`,
          progress: (totalSteps + 0.4 + 0.78 * frac) / (totalSteps + 1.2),
          stage1Pose: initPoseForStage2,
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
      } else {
        finalPose = livePose;
        lmTrace = liveTrace;
      }
    }
  }

  // Return robot to home posture & update 3 frustums in viewport
  engine.setState(engine.spec.homeQ, 0);
  rgbdScene.syncFromMujoco();
  rgbdScene.setFrustumPoses(
    T_cb_gt,
    config.mode === 'modeB' ? initPoseForStage2 : stage1Pose,
    config.mode === 'modeB' ? finalPose : null
  );
  rgbdScene.renderNow();

  const finalErr = computePoseError(finalPose, T_cb_gt);
  const elapsedMs = performance.now() - t0;
  const success = finalErr.transMm < 20.0 && finalErr.rotDeg < 2.0;

  traceLogger.log(
    'metric',
    'pipelineV2',
    'runSingleCalibration',
    `[V2] 标定完成: ${finalErr.transMm.toFixed(2)}mm / ${finalErr.rotDeg.toFixed(3)}° (${success ? '合格' : '发散'})`,
    {
      durationMs: elapsedMs,
      output: {
        initTransMm: initErr.transMm,
        initRotDeg: initErr.rotDeg,
        stage1TransMm: stage1Err.transMm,
        stage1RotDeg: stage1Err.rotDeg,
        finalTransMm: finalErr.transMm,
        finalRotDeg: finalErr.rotDeg,
        fimLogDet: fimMetrics.logDet,
        fimLambdaMin: fimMetrics.lambdaMin,
        success,
        T_cb_final: finalPose,
      },
    }
  );

  return {
    algoVersion: 'v2',
    initMode: config.initMode,
    T_cb_gt,
    T_cb_init: initPoseForStage2,
    T_cb_stage1: stage1Pose,
    T_cb_final: finalPose,
    initRotDeg: initErr.rotDeg,
    initTransMm: initErr.transMm,
    stage1RotDeg: stage1Err.rotDeg,
    stage1TransMm: stage1Err.transMm,
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
    success,
  };
}

export interface AblationConditionSpec {
  id: string;
  name: string;
  shortDesc: string;
  override: (base: CalibrationConfig) => CalibrationConfig;
}

export const SIX_ABLATION_CONDITIONS: readonly AblationConditionSpec[] = [
  {
    id: 'full_mode_b',
    name: '1. Full Mode B (完整方法)',
    shortDesc: 'FIM + 速度场 + 飞点滤波 + 时间戳同步',
    override: (base) => ({
      ...base,
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
    id: 'mode_a_no_mesh',
    name: '2. Mode A (无 CAD Mesh)',
    shortDesc: '仅 Stage 1 螺旋轴伴随闭式解',
    override: (base) => ({
      ...base,
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
    id: 'wo_flying_filter',
    name: '3. w/o 飞点滤波与物理加权',
    shortDesc: '关闭腐蚀/SOR/物理权重 (均匀权)',
    override: (base) => ({
      ...base,
      mode: 'modeB',
      strategy: 'fim',
      useFlyingPixelFilter: false,
      useVelocityField: true,
      useTimestampSync: true,
      enableDistractor: true,
      enableCustomTool: true,
    }),
  },
  {
    id: 'random_planning',
    name: '4. Random 规划 (vs FIM)',
    shortDesc: '随机候选动作替代 D/E-最优 FIM',
    override: (base) => ({
      ...base,
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
    id: 'wo_velocity_field',
    name: '5. 仅静态点到面 (w/o 速度场)',
    shortDesc: '关闭动态法向速度场残差项',
    override: (base) => ({
      ...base,
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
    id: 'wo_timestamp_sync',
    name: '6. w/o 时间戳同步 (含干扰物)',
    shortDesc: '存在异步动态干扰物且关闭相关性滤波',
    override: (base) => ({
      ...base,
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
 * Run 6-Condition Paired Monte Carlo Ablation Benchmark on identical random GT camera poses.
 */
export async function runPairedAblationBenchmark(
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

  // Sample N paired ground-truth camera poses tailored to active robot, shared across ALL 6 conditions
  const pairedPoses: SE3[] = [];
  for (let t = 0; t < nTrials; t++) {
    pairedPoses.push(sampleRandomCameraPose(poseRng, engine.spec));
  }

  const totalRuns = SIX_ABLATION_CONDITIONS.length * nTrials;
  let completedRuns = 0;
  const rows: AblationRow[] = [];

  for (let cIdx = 0; cIdx < SIX_ABLATION_CONDITIONS.length; cIdx++) {
    if (opts.shouldCancel()) break;
    const spec = SIX_ABLATION_CONDITIONS[cIdx];
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
        text: `消融基准 [${cIdx + 1}/6] ${spec.name} · Trial ${t + 1}/${nTrials}`,
        progress: completedRuns / totalRuns,
        rows: [...rows],
      });
      await nextAnimationFrame();

      const res = await runSingleCalibration(
        engine,
        rgbdScene,
        pairedPoses[t],
        condConfig,
        trialSeed,
        {
          shouldCancel: opts.shouldCancel,
          fastMode: true,
        }
      );

      rotDegs.push(res.finalRotDeg);
      transMms.push(res.finalTransMm);
      timesMs.push(res.elapsedMs);
      fimLogs.push(res.fimLogDet);
      if (res.success) successCnt++;
      completedRuns++;

      opts.onProgress({
        text: `消融基准 [${cIdx + 1}/6] ${spec.name} · 完成 Trial ${t + 1}/${nTrials} (${res.finalTransMm.toFixed(1)}mm / ${res.finalRotDeg.toFixed(2)}°)`,
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

      traceLogger.log('metric', 'ablationV2', 'conditionComplete', `完成条件 ${cIdx + 1}/6: ${spec.name}`, {
        output: row,
      });

      opts.onProgress({
        text: `已完成条件 ${cIdx + 1}/6: ${spec.name}`,
        progress: completedRuns / totalRuns,
        rows: [...rows],
      });
    }
  }

  return rows;
}

export function formatAblationCSV(rows: readonly AblationRow[], config: CalibrationConfig): string {
  const robotSpec = getRobotSpec(config.robotId);
  const initLabel = getInitModeLabel(config.initMode, config.initRotErrDeg, config.initTransErrMm);
  const algoLabel = config.algoVersion === 'v3' ? 'V3.0' : 'V2.0';
  const header = [
    'AlgoVersion',
    'Robot',
    'InitMode',
    'Condition',
    'Description',
    'Trials',
    'RotMean_deg',
    'RotStd_deg',
    'TransMean_mm',
    'TransStd_mm',
    'TransMedian_mm',
    'SuccessRate_pct',
    'FIM_LogDet',
    'MeanTime_ms',
    'Steps',
    'SigmaZ0',
    'EdgeNoise',
    'MaterialNoise',
    'OutlierRate',
    'AsyncDelay_ms',
    'FrontPillar',
    'FlexCable',
  ].join(',');

  const lines = rows.map((r) =>
    [
      `"${algoLabel}"`,
      `"${robotSpec.name}"`,
      `"${initLabel}"`,
      `"${r.name}"`,
      `"${r.shortDesc}"`,
      r.trials,
      r.rotMeanDeg.toFixed(4),
      r.rotStdDeg.toFixed(4),
      r.transMeanMm.toFixed(3),
      r.transStdMm.toFixed(3),
      r.transMedianMm.toFixed(3),
      r.successRate.toFixed(1),
      r.fimLogDetMean.toFixed(3),
      r.meanTimeMs.toFixed(1),
      config.numSteps,
      config.sensor.sigmaZ0.toFixed(4),
      config.sensor.edgeNoiseStrength.toFixed(2),
      config.sensor.materialNoiseStrength.toFixed(2),
      config.sensor.outlierRate.toFixed(4),
      config.sensor.asyncDelayMs,
      config.enableFrontPillar ? 'ON' : 'OFF',
      config.enableFlexCable ? 'ON' : 'OFF',
    ].join(',')
  );

  return [header, ...lines].join('\n');
}

export function formatMarkdownReport(
  config: CalibrationConfig,
  singleResult: CalibrationResult | null,
  ablationRows: readonly AblationRow[]
): string {
  const robotSpec = getRobotSpec(config.robotId);
  const initLabel = getInitModeLabel(config.initMode, config.initRotErrDeg, config.initTransErrMm);
  const algoLabel =
    config.algoVersion === 'v3'
      ? 'V3.0 统一研究框架 (2D+3D双域排斥 + 盲探针GNC-Welsch + Z-Buf无量纲FIM + 3分量退火EM-LM)'
      : 'V2.0 基础框架 (双层背景剥离 + 螺旋轴闭式解 + FIM + Huber LM)';
  const lines: string[] = [];
  lines.push('# 免分割 RGB-D 主动运动手眼标定实验报告');
  lines.push('');
  lines.push('## 1. 仿真与传感器实验配置');
  lines.push(`- **算法核心版本**: ${algoLabel}`);
  lines.push(`- **物理引擎与机器人模型**: ${robotSpec.name} (${robotSpec.badgeText} · ${robotSpec.materialTag} / \`@mujoco/mujoco\` WASM)`);
  lines.push(`- **初始手眼外参初始化方式**: ${initLabel}`);
  lines.push(`- **虚拟传感器**: RealSense D435i (\`${config.sensor.width}×${config.sensor.height}\`, FOV \`${config.sensor.fovDeg}°\`, WebGL2 24-bit 线性深度 + 材质光学 G-Buffer)`);
  lines.push(`- **深度噪声系数**: \`σ₀ = ${config.sensor.sigmaZ0.toFixed(4)}\` (\`σ(z) = σ₀ z²\`)`);
  lines.push(`- **真实材质/边缘强噪声**: 边缘跳变大噪声 \`${config.sensor.edgeNoiseStrength.toFixed(1)}×\` · 金属反光/黑体吸光退化 \`${config.sensor.materialNoiseStrength.toFixed(1)}×\` · 立体匹配离群率 \`${(config.sensor.outlierRate * 100).toFixed(1)}%\``);
  lines.push(`- **相机-编码器异步延迟**: \`${config.sensor.asyncDelayMs} ms\``);
  lines.push(`- **飞点模拟**: ${config.sensor.enableFlyingPixels ? '开启' : '关闭'} | **掠射角/金属/黑体空洞**: ${config.sensor.enableHoles ? '开启' : '关闭'}`);
  lines.push(`- **探索步数**: \`${config.numSteps}\` 步 (含冷启动盲探针)`);
  lines.push(`- **场景压力干扰**: 动态干扰物(${config.enableDistractor ? '开启' : '关闭'}) · 未知非标夹具(${config.enableCustomTool ? '开启' : '关闭'}) · 前侧遮挡立柱(${config.enableFrontPillar ? '开启' : '关闭'}) · 耦合柔性线缆(${config.enableFlexCable ? '开启' : '关闭'})`);
  lines.push('');

  if (singleResult) {
    const initColTitle =
      singleResult.initMode === 'stage1'
        ? config.algoVersion === 'v3'
          ? 'Stage 2 (GNC-Welsch 闭式解)'
          : 'Stage 1 (闭式解)'
        : `初始外参 (${getInitModeLabel(singleResult.initMode)})`;
    lines.push('## 2. 最近单次标定定量结果');
    lines.push(`| 指标 | ${initColTitle} | 最终结果 (` + (config.mode === 'modeB' ? (config.algoVersion === 'v3' ? 'Stage 4 退火 EM-LM' : 'Mode B 流形精修') : 'Mode A') + ') |');
    lines.push('| :--- | :---: | :---: |');
    lines.push(`| 旋转误差 (deg) | ${singleResult.initRotDeg.toFixed(3)}° | **${singleResult.finalRotDeg.toFixed(3)}°** |`);
    lines.push(`| 平移误差 (mm) | ${singleResult.initTransMm.toFixed(2)} mm | **${singleResult.finalTransMm.toFixed(2)} mm** |`);
    lines.push(`| 累积 FIM log det | — | ${singleResult.fimLogDet.toFixed(2)} (λ_min = ${singleResult.fimLambdaMin.toExponential(2)}) |`);
    if (singleResult.v3Diagnostics) {
      const vd = singleResult.v3Diagnostics;
      lines.push(`| V3 EM 类别先验 (I / II / III) | — | ${(vd.classPriors[0] * 100).toFixed(1)}% / ${(vd.classPriors[1] * 100).toFixed(1)}% / ${(vd.classPriors[2] * 100).toFixed(1)}% |`);
      if (vd.estimatedDelayMs !== null) {
        lines.push(`| V3 时间偏置辨识 δt̂ | — | ${vd.estimatedDelayMs.toFixed(1)} ms (真值 ${config.sensor.asyncDelayMs} ms) |`);
      }
      lines.push(`| V3 Z-Buffer 遮挡剔除率 | — | ${vd.zBufferOccludedPct.toFixed(1)}% (M_occ_2D: ${vd.occ2DBandPixels} px) |`);
    }
    lines.push(`| 总计算耗时 | — | ${(singleResult.elapsedMs / 1000).toFixed(2)} s |`);
    lines.push('');
  }

  if (ablationRows.length > 0) {
    lines.push(`## 3. 六组配对 Monte Carlo 消融对比表 (N = ${ablationRows[0].trials} 组同真值位姿)`);
    lines.push('| 实验条件 | 旋转误差 mean±std (°) | 平移误差 mean±std (mm) | 中位平移误差 (mm) | 成功率 (<20mm & <2°) | FIM log det | 平均耗时 (s) |');
    lines.push('| :--- | :---: | :---: | :---: | :---: | :---: | :---: |');
    for (const r of ablationRows) {
      lines.push(
        `| ${r.name} | ${r.rotMeanDeg.toFixed(3)} ± ${r.rotStdDeg.toFixed(3)} | ${r.transMeanMm.toFixed(2)} ± ${r.transStdMm.toFixed(2)} | ${r.transMedianMm.toFixed(2)} | ${r.successRate.toFixed(1)}% | ${r.fimLogDetMean.toFixed(2)} | ${(r.meanTimeMs / 1000).toFixed(2)} |`
      );
    }
    lines.push('');
  }

  return lines.join('\n');
}
