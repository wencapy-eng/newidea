import type { SE3, Vec3, Vec6 } from '../math/se3';
import type { RobotId } from '../sim/robotCatalog';

export type CalibrationMode = 'modeB' | 'modeA';
export type ExplorationStrategy = 'fim' | 'random';
export type ExtrinsicInitMode = 'stage1' | 'controlled_random' | 'full_random' | 'zero';
export type AlgoVersion = 'v2' | 'v3';

export interface SensorConfig {
  width: number;
  height: number;
  fovDeg: number;
  sigmaZ0: number;
  asyncDelayMs: number;
  enableFlyingPixels: boolean;
  enableHoles: boolean;
  /** Depth-discontinuity 2-ring silhouette heavy-tailed edge noise multiplier (0..3, default 1.0) */
  edgeNoiseStrength: number;
  /** Material-dependent specular metallic multipath & dark-IR absorption degradation (0..2.5, default 1.0) */
  materialNoiseStrength: number;
  /** Random stereo correspondence mismatch outlier spike probability (0..0.06, default 0.004) */
  outlierRate: number;
}

export interface CalibrationConfig {
  algoVersion: AlgoVersion;
  robotId: RobotId;
  mode: CalibrationMode;
  strategy: ExplorationStrategy;
  /** Hand-eye extrinsic initialization mode for Stage 2/4 optimization */
  initMode: ExtrinsicInitMode;
  /** Controlled random initialization rotation error magnitude (deg, 5..60) */
  initRotErrDeg: number;
  /** Controlled random initialization translation error magnitude (mm, 20..350) */
  initTransErrMm: number;
  numSteps: number; // 4..8
  mcTrials: number; // 3..20
  useFlyingPixelFilter: boolean;
  useVelocityField: boolean;
  useTimestampSync: boolean;
  enableDistractor: boolean;
  enableCustomTool: boolean;
  /** V3 Class IV stress test: static front occlusion pillar slicing across camera-robot sightline */
  enableFrontPillar: boolean;
  /** V3 Class III stress test: coupled non-rigid whipping flexible cable attached to arm */
  enableFlexCable: boolean;
  /** V3 Stage 4 optional 7th state variable: joint identification of camera-encoder time offset delta_t */
  estimateTimeOffset: boolean;
  sensor: SensorConfig;
}

export interface CameraIntrinsics {
  width: number;
  height: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
}

export interface RGBDFrame {
  width: number;
  height: number;
  /** Top-left origin RGBA uint8 RGB image (length W*H*4) */
  rgb: Uint8ClampedArray;
  /** Top-left origin linear metric depth in meters, 0 = hole/invalid (length W*H) */
  depth: Float32Array;
  /** Physical timestamp in seconds */
  timestamp: number;
  /** True robot joint angles at capture time (N-DoF) - ONLY for physics simulation, never leaked to algorithm */
  qTrue: number[];
  /** Encoder joint angles read at (timestamp - asyncDelay) */
  qEncoder: number[];
  /** Encoder joint velocities read at (timestamp - asyncDelay) */
  qdotEncoder: number[];
  /** True active joint index (0..dof-1) */
  activeJoint: number;
}

export interface WeightedPoint {
  p: Vec3;
  n: Vec3;
  w: number;
  u: number;
  v: number;
  z: number;
}

export interface MotionAction {
  id: string;
  label: string;
  jointIndex: number; // 0..dof-1
  qBase: number[]; // length = robot dof
  qStart: number;
  qEnd: number;
  isColdStart?: boolean;
}

export interface SyncComponentCurve {
  componentId: number;
  pixelCount: number;
  rho: number;
  bestLag: number;
  accepted: boolean;
  /** Normalized per-frame activity curve (length 16) */
  activity: number[];
}

export interface SyncDiagnostics {
  actionLabel: string;
  jointIndex: number;
  /** Normalized encoder |qdot(t)| curve (length 16) */
  encoderSpeed: number[];
  components: SyncComponentCurve[];
  acceptedPixels: number;
  rejectedPixels: number;
}

export interface ScrewObservation {
  action: MotionAction;
  /** 'screw' for revolute joints; 'translation' for Delta parallel closed-loop pure-translation sweeps */
  kind?: 'screw' | 'translation';
  xiCam: Vec6;
  xiBase: Vec6;
  deltaQ: number;
  thetaObs: number;
  pitch: number;
  icpRms: number;
  weight: number;
  dTCam: SE3;
  /** Optional parallel-platform displacement in base frame {B} */
  deltaPBase?: Vec3;
  /** Optional parallel-platform displacement in camera frame {C} */
  deltaPCam?: Vec3;
  /** Optional platform kinematic anchor in base frame {B} at mid-sweep */
  centroidBase?: Vec3;
  /** Optional tracked moving platform centroid in camera frame {C} at mid-sweep */
  centroidCam?: Vec3;
}

export interface LMTracePoint {
  iter: number;
  stage: number;
  cost: number;
  rotErrDeg: number;
  transErrMm: number;
  geoCount: number;
  velCount: number;
  lambda: number;
}

export type ErrorHistoryPhase = 'init' | 'closed_form' | 'lm' | 'em_lm';

export interface ErrorHistoryPoint {
  /** Monotonic step index across the entire single calibration run */
  step: number;
  /** Compact x-axis / tooltip tag (e.g. 'Init', 'S3', 'S4', 'LM0', 'EM12') */
  label: string;
  /** Pipeline stage that produced this pose estimate */
  phase: ErrorHistoryPhase;
  /** Translation error in mm against T_cb_gt (diagnostics only, never used by solver) */
  transErrMm: number;
  /** Rotation error in degrees against T_cb_gt (diagnostics only, never used by solver) */
  rotErrDeg: number;
  /** Optional optimizer inner stage / annealing stage index */
  stage?: number;
}

export interface V3ProbeLogEntry {
  stepLabel: string;
  jointIndex: number;
  accepted: boolean;
  pointCount: number;
  ampResidual: number;
  ampUnit: 'deg' | 'mm';
  sigmaMin: number;
  reason: string;
}

export interface V3Diagnostics {
  /** Estimated camera-encoder time offset delta_t in ms (when 7-DoF time offset estimation is enabled) */
  estimatedDelayMs: number | null;
  /** Converged EM mixture priors [Class I Nominal, Class II Rigid Attachment, Class III Non-Rigid] */
  classPriors: [number, number, number];
  /** Posterior-weighted point counts [Class I, Class II, Class III] in final EM stage */
  classCounts: [number, number, number];
  /** Percentage of FIM surface samples rejected by static Z-Buffer occlusion */
  zBufferOccludedPct: number;
  /** Number of pixels flagged in 2D depth-conditioned front-occlusion band M_occ_2D */
  occ2DBandPixels: number;
  /** Blind-probe cold-start admission log */
  probeLogs: V3ProbeLogEntry[];
  /**
   * Per-pixel class overlay map (length W*H):
   * 0 = background, 1 = Class I (Cyan), 2 = Class II (Amber), 3 = Class III (Magenta), 4 = M_occ_2D (Crimson)
   */
  posteriorOverlay: Uint8Array | null;
}

export interface CalibrationResult {
  algoVersion?: AlgoVersion;
  initMode: ExtrinsicInitMode;
  T_cb_gt: SE3;
  T_cb_init: SE3;
  T_cb_stage1: SE3;
  T_cb_final: SE3;
  initRotDeg: number;
  initTransMm: number;
  stage1RotDeg: number;
  stage1TransMm: number;
  finalRotDeg: number;
  finalTransMm: number;
  fimLogDet: number;
  fimLambdaMin: number;
  elapsedMs: number;
  executedActions: MotionAction[];
  screwObservations: ScrewObservation[];
  syncDiagnostics: SyncDiagnostics | null;
  lmTrace: LMTracePoint[];
  errorHistory?: ErrorHistoryPoint[];
  lastRgb: Uint8ClampedArray | null;
  lastDepth: Float32Array | null;
  lastMask: Uint8Array | null;
  v3Diagnostics?: V3Diagnostics | null;
  success: boolean;
}

export interface AblationRow {
  id: string;
  name: string;
  shortDesc: string;
  trials: number;
  rotMeanDeg: number;
  rotStdDeg: number;
  transMeanMm: number;
  transStdMm: number;
  transMedianMm: number;
  successRate: number; // percentage 0..100
  meanTimeMs: number;
  fimLogDetMean: number;
  trialRotDegs: number[];
  trialTransMms: number[];
}
