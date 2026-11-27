import {
  cross3,
  dot3,
  norm3,
  normalize3,
  type Vec3,
} from '../../math/se3';
import type {
  CameraIntrinsics,
  RGBDFrame,
  SyncComponentCurve,
  SyncDiagnostics,
  WeightedPoint,
} from '../types';

export const V3_BORDER_MARGIN_PX = 15;
export const V3_STATIC_VOXEL_RADIUS_M = 0.018;
export const V3_OCC_EDGE_DILATE_PX = 5;
export const V3_BG_DIFF_THRESH_M = 0.022;
export const V3_SEED_MOTION_THRESH_M = 0.014;
export const V3_SYNC_CORR_THRESH = 0.40;

/**
 * Spike-resistant online depth envelope tracker maintaining robust per-pixel
 * D_max(u,v) (upper background envelope) and D_min(u,v) (lower foreground envelope)
 * across an arbitrary stream of RGB-D frames.
 *
 * Uses top-K / bottom-K order statistics (K ~ 6% of frames, capped at 7) so random
 * positive/negative stereo outlier spikes never corrupt D_max or D_min.
 */
export class OnlineDepthEnvelope {
  readonly width: number;
  readonly height: number;
  private readonly topK: number;
  private readonly topBuf: Float32Array; // length N * topK, descending
  private readonly botBuf: Float32Array; // length N * topK, ascending
  private readonly validCount: Uint16Array;
  private frameCount = 0;

  constructor(width: number, height: number, topK = 6) {
    this.width = width;
    this.height = height;
    this.topK = Math.max(2, Math.min(8, topK));
    const n = width * height;
    this.topBuf = new Float32Array(n * this.topK);
    this.botBuf = new Float32Array(n * this.topK);
    this.botBuf.fill(99.0);
    this.validCount = new Uint16Array(n);
  }

  get totalFrames(): number {
    return this.frameCount;
  }

  addFrame(depth: Float32Array): void {
    this.frameCount++;
    const n = this.width * this.height;
    const K = this.topK;

    for (let i = 0; i < n; i++) {
      const z = depth[i];
      if (z <= 0.12 || z >= 7.5) continue;
      this.validCount[i]++;
      const base = i * K;

      // Insert into descending top-K buffer
      if (z > this.topBuf[base + K - 1]) {
        let pos = K - 1;
        while (pos > 0 && z > this.topBuf[base + pos - 1]) {
          this.topBuf[base + pos] = this.topBuf[base + pos - 1];
          pos--;
        }
        this.topBuf[base + pos] = z;
      }

      // Insert into ascending bottom-K buffer
      if (z < this.botBuf[base + K - 1]) {
        let pos = K - 1;
        while (pos > 0 && z < this.botBuf[base + pos - 1]) {
          this.botBuf[base + pos] = this.botBuf[base + pos - 1];
          pos--;
        }
        this.botBuf[base + pos] = z;
      }
    }
  }

  addFrames(frames: readonly RGBDFrame[]): void {
    for (let i = 0; i < frames.length; i++) {
      this.addFrame(frames[i].depth);
    }
  }

  /** Extract spike-resistant D_max and D_min maps */
  getEnvelopes(): { dMax: Float32Array; dMin: Float32Array } {
    const n = this.width * this.height;
    const K = this.topK;
    const dMax = new Float32Array(n);
    const dMin = new Float32Array(n);

    // Pick rank proportional to total frames (~6% tail rejected)
    const targetRank =
      this.frameCount <= 16
        ? Math.min(1, K - 1)
        : Math.min(K - 1, Math.max(1, Math.round(this.frameCount * 0.06)));

    for (let i = 0; i < n; i++) {
      const cnt = this.validCount[i];
      if (cnt === 0) continue;
      const r = Math.min(targetRank, cnt - 1);
      const base = i * K;
      const maxVal = this.topBuf[base + r];
      const minVal = this.botBuf[base + r];
      dMax[i] = maxVal > 0.12 ? maxVal : this.topBuf[base];
      dMin[i] = minVal < 90.0 ? minVal : this.botBuf[base];
    }

    return { dMax, dMin };
  }
}

/**
 * 3D spatial voxel exclusion grid V_static built by backprojecting D_max(u,v)
 * with Euclidean safety inflation radius r_s (default 18 mm).
 *
 * Excises static base/table contacts & permanent static structures in 3D without
 * hurting robot links sweeping in front of rear backgrounds.
 */
export class StaticVoxelExclusion3D {
  private readonly invCell: number;
  private readonly radiusSq: number;
  private readonly buckets = new Map<number, number[]>(); // packed [x,y,z,...]

  constructor(radiusM = V3_STATIC_VOXEL_RADIUS_M) {
    const cell = Math.max(0.012, radiusM);
    this.invCell = 1 / cell;
    this.radiusSq = radiusM * radiusM;
  }

  private hash(ix: number, iy: number, iz: number): number {
    return ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) | 0;
  }

  buildFromStaticEnvelope(
    dMax: Float32Array,
    K: CameraIntrinsics,
    stride = 2
  ): void {
    this.buckets.clear();
    const W = K.width;
    const H = K.height;
    const invFx = 1 / K.fx;
    const invFy = 1 / K.fy;

    for (let v = 0; v < H; v += stride) {
      const row = v * W;
      const ny = (v + 0.5 - K.cy) * invFy;
      for (let u = 0; u < W; u += stride) {
        const z = dMax[row + u];
        if (z <= 0.14 || z >= 6.5) continue;
        const x = (u + 0.5 - K.cx) * invFx * z;
        const y = ny * z;
        const ix = Math.floor(x * this.invCell);
        const iy = Math.floor(y * this.invCell);
        const iz = Math.floor(z * this.invCell);
        const key = this.hash(ix, iy, iz);
        let arr = this.buckets.get(key);
        if (!arr) {
          arr = [];
          this.buckets.set(key, arr);
        }
        // Keep at most 12 representative static samples per voxel for O(1) lookup
        if (arr.length < 36) {
          arr.push(x, y, z);
        }
      }
    }
  }

  /** Returns true if 3D point qCam lies within safety radius r_s of V_static */
  isNearStatic(x: number, y: number, z: number): boolean {
    const cx = Math.floor(x * this.invCell);
    const cy = Math.floor(y * this.invCell);
    const cz = Math.floor(z * this.invCell);
    const r2 = this.radiusSq;

    for (let dx = -1; dx <= 1; dx++) {
      for (let dy = -1; dy <= 1; dy++) {
        for (let dz = -1; dz <= 1; dz++) {
          const arr = this.buckets.get(this.hash(cx + dx, cy + dy, cz + dz));
          if (!arr) continue;
          for (let i = 0; i < arr.length; i += 3) {
            const ex = arr[i] - x;
            const ey = arr[i + 1] - y;
            const ez = arr[i + 2] - z;
            if (ex * ex + ey * ey + ez * ez < r2) {
              return true;
            }
          }
        }
      }
    }
    return false;
  }
}

export interface Occ2DExclusionMap {
  /** Nearest protruding static occluder depth D_near_edge(u,v) in dilated band, 0 if not in band */
  nearEdgeDepth: Float32Array;
  /** 1 if pixel lies in dilated protruding static edge band (for visualization/stats) */
  bandMask: Uint8Array;
  /** Total pixels inside the dilated static front-occlusion edge band */
  bandPixelCount: number;
}

/**
 * 2D Depth-Conditioned Front-Occlusion Edge Band (M_occ_2D):
 *  1. Detect permanent static pixels (D_max - D_min < tauS) sitting on the NEAR (protruding) side
 *     of a depth discontinuity (|∇D_max| > tauG).
 *  2. Dilate by `dilatePx` (4..6 px), storing the shallowest protruding edge depth D_near_edge(u,v).
 *  3. At runtime, reject pixel (u,v) in the band ONLY when D_t(u,v) >= D_near_edge(u,v) - tauZ,
 *     excising pillar-edge flying pixels while preserving robot links passing IN FRONT of the pillar!
 */
export function buildDepthConditionedOcc2DBand(
  dMax: Float32Array,
  dMin: Float32Array,
  width: number,
  height: number,
  opts?: {
    tauStaticDiff?: number;
    tauGrad?: number;
    dilatePx?: number;
  }
): Occ2DExclusionMap {
  const tauS = opts?.tauStaticDiff ?? 0.028;
  const tauG = opts?.tauGrad ?? 0.065;
  const dilatePx = opts?.dilatePx ?? V3_OCC_EDGE_DILATE_PX;
  const n = width * height;

  const seedEdgeDepth = new Float32Array(n);
  const edgeSeeds: number[] = [];

  for (let v = 1; v < height - 1; v++) {
    const row = v * width;
    for (let u = 1; u < width - 1; u++) {
      const idx = row + u;
      const zMax = dMax[idx];
      const zMin = dMin[idx];
      if (zMax <= 0.14 || zMin <= 0.14) continue;
      // Must be a permanently static pixel across the sequence
      if (zMax - zMin >= tauS) continue;

      // Check if this static pixel is strictly NEARER than at least one 4-neighbor by > tauG
      // (protruding front-occluder silhouette edge)
      const nL = dMax[idx - 1];
      const nR = dMax[idx + 1];
      const nU = dMax[idx - width];
      const nD = dMax[idx + width];
      const protruding =
        (nL > 0.14 && nL - zMax > tauG) ||
        (nR > 0.14 && nR - zMax > tauG) ||
        (nU > 0.14 && nU - zMax > tauG) ||
        (nD > 0.14 && nD - zMax > tauG);

      if (protruding) {
        seedEdgeDepth[idx] = zMax;
        edgeSeeds.push(idx);
      }
    }
  }

  const nearEdgeDepth = new Float32Array(n);
  const bandMask = new Uint8Array(n);
  let bandPixelCount = 0;

  // Dilate protruding static edges by circular radius dilatePx, storing minimum (shallowest) edge depth
  const r2 = dilatePx * dilatePx;
  for (let s = 0; s < edgeSeeds.length; s++) {
    const sIdx = edgeSeeds[s];
    const su = sIdx % width;
    const sv = (sIdx - su) / width;
    const zEdge = seedEdgeDepth[sIdx];

    const v0 = Math.max(0, sv - dilatePx);
    const v1 = Math.min(height - 1, sv + dilatePx);
    const u0 = Math.max(0, su - dilatePx);
    const u1 = Math.min(width - 1, su + dilatePx);

    for (let vv = v0; vv <= v1; vv++) {
      const dv = vv - sv;
      const dv2 = dv * dv;
      const rOff = vv * width;
      for (let uu = u0; uu <= u1; uu++) {
        const du = uu - su;
        if (du * du + dv2 > r2) continue;
        const idx = rOff + uu;
        if (bandMask[idx] === 0) {
          bandMask[idx] = 1;
          nearEdgeDepth[idx] = zEdge;
          bandPixelCount++;
        } else if (zEdge < nearEdgeDepth[idx]) {
          nearEdgeDepth[idx] = zEdge;
        }
      }
    }
  }

  return { nearEdgeDepth, bandMask, bandPixelCount };
}

/** Check whether pixel (u, v) at depth z is rejected by M_occ_2D (border margin OR behind/on static front edge) */
export function isRejectedByOcc2D(
  u: number,
  v: number,
  z: number,
  width: number,
  height: number,
  occ2D: Occ2DExclusionMap | null,
  borderPx = V3_BORDER_MARGIN_PX,
  tauZ = 0.035
): boolean {
  if (u < borderPx || u >= width - borderPx || v < borderPx || v >= height - borderPx) {
    return true;
  }
  if (!occ2D) return false;
  const zEdge = occ2D.nearEdgeDepth[v * width + u];
  if (zEdge > 0 && z >= zEdge - tauZ) {
    return true;
  }
  return false;
}

/** 4-connected component labeling on a binary mask */
function labelConnectedComponents(
  mask: Uint8Array,
  width: number,
  height: number,
  minPixels: number
): { labels: Int32Array; compIds: number[]; compSizes: Map<number, number> } {
  const n = width * height;
  const labels = new Int32Array(n);
  const queue = new Int32Array(n);
  const compIds: number[] = [];
  const compSizes = new Map<number, number>();
  let nextLabel = 1;

  for (let i = 0; i < n; i++) {
    if (mask[i] === 0 || labels[i] !== 0) continue;
    const label = nextLabel++;
    let head = 0;
    let tail = 0;
    queue[tail++] = i;
    labels[i] = label;

    while (head < tail) {
      const cur = queue[head++];
      const x = cur % width;
      const y = (cur - x) / width;
      if (x > 0) {
        const nb = cur - 1;
        if (mask[nb] && labels[nb] === 0) {
          labels[nb] = label;
          queue[tail++] = nb;
        }
      }
      if (x + 1 < width) {
        const nb = cur + 1;
        if (mask[nb] && labels[nb] === 0) {
          labels[nb] = label;
          queue[tail++] = nb;
        }
      }
      if (y > 0) {
        const nb = cur - width;
        if (mask[nb] && labels[nb] === 0) {
          labels[nb] = label;
          queue[tail++] = nb;
        }
      }
      if (y + 1 < height) {
        const nb = cur + width;
        if (mask[nb] && labels[nb] === 0) {
          labels[nb] = label;
          queue[tail++] = nb;
        }
      }
    }

    if (tail >= minPixels) {
      compIds.push(label);
      compSizes.set(label, tail);
    } else {
      for (let q = 0; q < tail; q++) {
        labels[queue[q]] = 0;
      }
    }
  }

  return { labels, compIds, compSizes };
}

/**
 * Compute dual-domain (2D M_occ_2D + 3D V_static) foreground mask for a single frame
 * with optional region-level seed growing to keep link interiors intact while removing static noise.
 */
export function computeDualDomainForegroundMask(
  depth: Float32Array,
  dMax: Float32Array,
  K: CameraIntrinsics,
  vStatic: StaticVoxelExclusion3D | null,
  occ2D: Occ2DExclusionMap | null,
  opts?: {
    prevDepth?: Float32Array | null;
    useDualExclusion?: boolean;
    minComponentPx?: number;
    tauBg?: number;
    tauSeed?: number;
  }
): Uint8Array {
  const W = K.width;
  const H = K.height;
  const n = W * H;
  const useDual = opts?.useDualExclusion ?? true;
  const tauBg = opts?.tauBg ?? V3_BG_DIFF_THRESH_M;
  const tauSeed = opts?.tauSeed ?? V3_SEED_MOTION_THRESH_M;
  const minComp = opts?.minComponentPx ?? 28;
  const prevDepth = opts?.prevDepth ?? null;

  const invFx = 1 / K.fx;
  const invFy = 1 / K.fy;

  const mExist = new Uint8Array(n);
  const mSeed = new Uint8Array(n);

  for (let v = 0; v < H; v++) {
    const row = v * W;
    const ny = (v + 0.5 - K.cy) * invFy;
    for (let u = 0; u < W; u++) {
      const idx = row + u;
      const z = depth[idx];
      const zb = dMax[idx];
      if (z <= 0.14 || zb <= 0.14) continue;
      if (zb - z <= tauBg) continue;

      if (useDual) {
        if (isRejectedByOcc2D(u, v, z, W, H, occ2D)) continue;
        if (vStatic) {
          const x = (u + 0.5 - K.cx) * invFx * z;
          const y = ny * z;
          if (vStatic.isNearStatic(x, y, z)) continue;
        }
      }

      mExist[idx] = 1;
      if (prevDepth) {
        const zp = prevDepth[idx];
        if (zp <= 0.14 || Math.abs(z - zp) > tauSeed) {
          mSeed[idx] = 1;
        }
      }
    }
  }

  // Connected-component cleanup + region-level seed retention
  const { labels, compIds, compSizes } = labelConnectedComponents(mExist, W, H, minComp);
  const out = new Uint8Array(n);

  if (!prevDepth) {
    for (let i = 0; i < n; i++) {
      if (labels[i] > 0) out[i] = 1;
    }
    return out;
  }

  // Count dynamic seeds per component
  const seedCounts = new Map<number, number>();
  for (let i = 0; i < n; i++) {
    const lb = labels[i];
    if (lb > 0 && mSeed[i]) {
      seedCounts.set(lb, (seedCounts.get(lb) ?? 0) + 1);
    }
  }

  const keepLabels = new Set<number>();
  for (let c = 0; c < compIds.length; c++) {
    const id = compIds[c];
    const area = compSizes.get(id) ?? 1;
    const seeds = seedCounts.get(id) ?? 0;
    // Retain whole region if it has meaningful dynamic seed activity (no interior hollowing)
    if (seeds >= 5 || seeds / area >= 0.04) {
      keepLabels.add(id);
    }
  }

  // Fallback: if this frame is a static pause frame where motion is ~0, keep all valid M_exist components
  if (keepLabels.size === 0) {
    for (let i = 0; i < n; i++) {
      if (labels[i] > 0) out[i] = 1;
    }
    return out;
  }

  for (let i = 0; i < n; i++) {
    if (keepLabels.has(labels[i])) out[i] = 1;
  }
  return out;
}

/** Normalized lagged cross-correlation max_{|lag|<=maxLag} corr(a, b) */
function laggedMaxCrossCorrelation(
  a: readonly number[],
  b: readonly number[],
  maxLag = 3
): { rho: number; bestLag: number } {
  const T = Math.min(a.length, b.length);
  if (T < 4) return { rho: 0, bestLag: 0 };

  let bestRho = -1;
  let bestLag = 0;

  for (let lag = -maxLag; lag <= maxLag; lag++) {
    let meanA = 0;
    let meanB = 0;
    let count = 0;
    for (let t = 0; t < T; t++) {
      const tb = t - lag;
      if (tb < 0 || tb >= T) continue;
      meanA += a[t];
      meanB += b[tb];
      count++;
    }
    if (count < 4) continue;
    meanA /= count;
    meanB /= count;

    let num = 0;
    let denA = 0;
    let denB = 0;
    for (let t = 0; t < T; t++) {
      const tb = t - lag;
      if (tb < 0 || tb >= T) continue;
      const da = a[t] - meanA;
      const db = b[tb] - meanB;
      num += da * db;
      denA += da * da;
      denB += db * db;
    }
    const den = Math.sqrt(denA * denB);
    const rho = den > 1e-10 ? num / den : 0;
    if (rho > bestRho) {
      bestRho = rho;
      bestLag = lag;
    }
  }

  return { rho: bestRho, bestLag };
}

/**
 * V3 Stage 1 Multi-Fragment Temporal Causal Sync Aggregation:
 *  - Computes local spike-resistant D_max/D_min & merges with global online envelope if available.
 *  - Applies 2D M_occ_2D + 3D V_static dual-domain static exclusion.
 *  - Drops single-component assumption: extracts ALL connected fragments (even when a front pillar
 *    splits the robot arm into 2-3 fragments), computes lagged cross-correlation rho_l against
 *    encoder speed ||qdot(t)||_2, and AGGREGATES every fragment with rho_l >= tauSync!
 *  - Rejects asynchronous dynamic distractors (Class IV) cleanly.
 */
export function segmentActionV3MultiFragment(
  frames: readonly RGBDFrame[],
  K: CameraIntrinsics,
  actionLabel: string,
  opts: {
    useTimestampSync: boolean;
    useDualExclusion: boolean;
    globalEnvelope?: OnlineDepthEnvelope | null;
    vStatic?: StaticVoxelExclusion3D | null;
    occ2D?: Occ2DExclusionMap | null;
  }
): {
  frameMasks: Uint8Array[];
  rejectedRegionMask: Uint8Array;
  localDMax: Float32Array;
  localOcc2D: Occ2DExclusionMap;
  diagnostics: SyncDiagnostics;
} {
  const T = frames.length;
  const W = K.width;
  const H = K.height;
  const n = W * H;

  // Build local action envelope (fast spike-resistant D_max/D_min)
  const localEnv = new OnlineDepthEnvelope(W, H, 3);
  localEnv.addFrames(frames);
  const localRes = localEnv.getEnvelopes();

  // Merge with global envelope when available so static occluded edges are maximally stable
  let dMax = localRes.dMax;
  let dMin = localRes.dMin;
  if (opts.globalEnvelope && opts.globalEnvelope.totalFrames > T) {
    const gRes = opts.globalEnvelope.getEnvelopes();
    dMax = new Float32Array(n);
    dMin = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      dMax[i] = Math.max(localRes.dMax[i], gRes.dMax[i]);
      const a = localRes.dMin[i];
      const b = gRes.dMin[i];
      dMin[i] = a > 0.12 && b > 0.12 ? Math.min(a, b) : Math.max(a, b);
    }
  }

  const localOcc2D =
    opts.occ2D ?? buildDepthConditionedOcc2DBand(dMax, dMin, W, H);
  const vStatic =
    opts.vStatic ??
    (() => {
      const vs = new StaticVoxelExclusion3D(V3_STATIC_VOXEL_RADIUS_M);
      vs.buildFromStaticEnvelope(dMax, K, 2);
      return vs;
    })();

  // Step 1: Per-frame dual-domain existence masks + union map of active motion regions
  const rawMasks: Uint8Array[] = [];
  const unionActive = new Uint8Array(n);
  const activeCount = new Uint16Array(n);

  for (let t = 0; t < T; t++) {
    const m = computeDualDomainForegroundMask(
      frames[t].depth,
      dMax,
      K,
      opts.useDualExclusion ? vStatic : null,
      opts.useDualExclusion ? localOcc2D : null,
      {
        useDualExclusion: opts.useDualExclusion,
        minComponentPx: 22,
      }
    );
    rawMasks.push(m);
    for (let i = 0; i < n; i++) {
      if (m[i]) {
        unionActive[i] = 1;
        activeCount[i]++;
      }
    }
  }

  // Keep pixels active in >= 2 frames so single-frame outlier blips do not bridge fragments
  for (let i = 0; i < n; i++) {
    if (activeCount[i] < 2) unionActive[i] = 0;
  }

  // Step 2: Label ALL spatial fragments across the action sweep
  const { labels, compIds, compSizes } = labelConnectedComponents(unionActive, W, H, 32);

  // Build encoder speed profile ||qdot(t)||_2
  const encoderRaw = new Array<number>(T).fill(0);
  let maxEnc = 1e-6;
  for (let t = 0; t < T; t++) {
    const qdot = frames[t].qdotEncoder;
    let sumSq = 0;
    for (let j = 0; j < qdot.length; j++) sumSq += qdot[j] * qdot[j];
    const sp = Math.sqrt(sumSq);
    encoderRaw[t] = sp;
    if (sp > maxEnc) maxEnc = sp;
  }
  const encoderNorm = encoderRaw.map((v) => v / maxEnc);

  // Step 3: Compute temporal motion energy E_l(t) for EVERY fragment l
  const compPixels = new Map<number, number[]>();
  for (let c = 0; c < compIds.length; c++) compPixels.set(compIds[c], []);
  for (let i = 0; i < n; i++) {
    const lb = labels[i];
    if (lb > 0) compPixels.get(lb)!.push(i);
  }

  const components: SyncComponentCurve[] = [];
  const acceptedLabels = new Set<number>();
  const rejectedLabels = new Set<number>();

  for (let c = 0; c < compIds.length; c++) {
    const cid = compIds[c];
    const pixList = compPixels.get(cid)!;
    const rawEnergy = new Array<number>(T).fill(0);

    for (let t = 1; t < T; t++) {
      const dCur = frames[t].depth;
      const dPrev = frames[t - 1].depth;
      let sumDiff = 0;
      let cnt = 0;
      for (let p = 0; p < pixList.length; p++) {
        const idx = pixList[p];
        const z1 = dCur[idx];
        const z0 = dPrev[idx];
        if (z1 > 0.14 && z0 > 0.14) {
          const diff = Math.abs(z1 - z0);
          if (diff > 0.005 && diff < 0.65) {
            sumDiff += diff;
            cnt++;
          }
        }
      }
      rawEnergy[t] = cnt > 0 ? sumDiff / Math.max(12, Math.sqrt(pixList.length)) : 0;
    }
    rawEnergy[0] = rawEnergy[1] * 0.25;

    // 3-tap temporal smoothing to suppress single-frame sensor jitter
    const smoothEnergy = new Array<number>(T).fill(0);
    let maxE = 1e-6;
    for (let t = 0; t < T; t++) {
      const a = rawEnergy[Math.max(0, t - 1)];
      const b = rawEnergy[t];
      const cc = rawEnergy[Math.min(T - 1, t + 1)];
      const val = 0.25 * a + 0.5 * b + 0.25 * cc;
      smoothEnergy[t] = val;
      if (val > maxE) maxE = val;
    }
    const normEnergy = smoothEnergy.map((v) => v / maxE);

    const { rho, bestLag } = laggedMaxCrossCorrelation(normEnergy, encoderNorm, 3);
    const accepted = !opts.useTimestampSync || rho >= V3_SYNC_CORR_THRESH;

    if (accepted) {
      acceptedLabels.add(cid);
    } else {
      rejectedLabels.add(cid);
    }

    components.push({
      componentId: cid,
      pixelCount: compSizes.get(cid) ?? pixList.length,
      rho,
      bestLag,
      accepted,
      activity: normEnergy,
    });
  }

  // Fallback: if strict threshold rejected every fragment (e.g., extreme noise), keep top-correlated fragment
  if (opts.useTimestampSync && acceptedLabels.size === 0 && components.length > 0) {
    let bestC = components[0];
    for (let i = 1; i < components.length; i++) {
      if (components[i].rho > bestC.rho) bestC = components[i];
    }
    bestC.accepted = true;
    acceptedLabels.add(bestC.componentId);
    rejectedLabels.delete(bestC.componentId);
  }

  // Step 4: Aggregate ALL accepted fragments into final per-frame masks
  const rejectedRegionMask = new Uint8Array(n);
  let acceptedPixels = 0;
  let rejectedPixels = 0;
  for (let i = 0; i < n; i++) {
    const lb = labels[i];
    if (lb > 0) {
      if (acceptedLabels.has(lb)) {
        acceptedPixels++;
      } else if (rejectedLabels.has(lb)) {
        rejectedRegionMask[i] = 1;
        rejectedPixels++;
      }
    }
  }

  const frameMasks: Uint8Array[] = [];
  for (let t = 0; t < T; t++) {
    const raw = rawMasks[t];
    const filtered = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      if (raw[i] && acceptedLabels.has(labels[i])) {
        filtered[i] = 1;
      }
    }
    frameMasks.push(filtered);
  }

  components.sort((a, b) => b.pixelCount - a.pixelCount);

  return {
    frameMasks,
    rejectedRegionMask,
    localDMax: dMax,
    localOcc2D,
    diagnostics: {
      actionLabel,
      jointIndex: frames[0]?.activeJoint ?? 0,
      encoderSpeed: encoderNorm,
      components: components.slice(0, 6),
      acceptedPixels,
      rejectedPixels,
    },
  };
}

/**
 * Extract pure optical-weighted 3D point cloud from depth + foreground mask:
 *   w_i = [max(0, -(n_i^c)^T (q_i^c / ||q_i^c||))^1.5 / (sigma0^2 * z_i^4)] * exp(-||grad D||^2 / (2 sigma_g^2))
 * Strictly uses only optical geometry (zero semantic/material label leakage).
 */
export function extractV3OpticalWeightedCloud(
  depth: Float32Array,
  mask: Uint8Array,
  K: CameraIntrinsics,
  opts: {
    filterEnabled: boolean;
    sigmaZ0: number;
    maxPoints?: number;
    sigmaGradM?: number;
  }
): WeightedPoint[] {
  const W = K.width;
  const H = K.height;
  const invFx = 1 / K.fx;
  const invFy = 1 / K.fy;
  const sigma0Sq = Math.max(opts.sigmaZ0, 0.0004) ** 2;
  const sigmaG = opts.sigmaGradM ?? 0.024;
  const invTwoSigmaGSq = 1 / (2 * sigmaG * sigmaG);
  const maxPoints = opts.maxPoints ?? 780;

  // Optional 1-px erosion + depth jump edge rejection when filterEnabled
  const candidates: Array<{ u: number; v: number; z: number; gradSq: number }> = [];
  for (let v = 2; v < H - 2; v++) {
    const row = v * W;
    for (let u = 2; u < W - 2; u++) {
      const idx = row + u;
      if (mask[idx] === 0) continue;
      const z = depth[idx];
      if (z <= 0.14 || z >= 5.5) continue;

      const zL = depth[idx - 1];
      const zR = depth[idx + 1];
      const zU = depth[idx - W];
      const zD = depth[idx + W];
      if (zL <= 0.14 || zR <= 0.14 || zU <= 0.14 || zD <= 0.14) continue;

      const gx = 0.5 * (zR - zL);
      const gy = 0.5 * (zD - zU);
      const gradSq = gx * gx + gy * gy;

      if (opts.filterEnabled) {
        // Erode 1-px boundary + suppress hard flying-pixel jumps
        if (!mask[idx - 1] || !mask[idx + 1] || !mask[idx - W] || !mask[idx + W]) continue;
        const maxJump = Math.max(
          Math.abs(z - zL),
          Math.abs(z - zR),
          Math.abs(z - zU),
          Math.abs(z - zD)
        );
        if (maxJump > 0.055) continue;
      }

      candidates.push({ u, v, z, gradSq });
    }
  }

  if (candidates.length === 0) return [];
  const stride = Math.max(1, Math.floor(candidates.length / maxPoints));
  const out: WeightedPoint[] = [];

  for (let i = 0; i < candidates.length; i += stride) {
    const { u, v, z, gradSq } = candidates[i];
    const idx = v * W + u;
    const zL = depth[idx - 1];
    const zR = depth[idx + 1];
    const zU = depth[idx - W];
    const zD = depth[idx + W];

    const p: Vec3 = [
      (u + 0.5 - K.cx) * invFx * z,
      (v + 0.5 - K.cy) * invFy * z,
      z,
    ];
    const pL: Vec3 = [(u - 0.5 - K.cx) * invFx * zL, (v + 0.5 - K.cy) * invFy * zL, zL];
    const pR: Vec3 = [(u + 1.5 - K.cx) * invFx * zR, (v + 0.5 - K.cy) * invFy * zR, zR];
    const pU: Vec3 = [(u + 0.5 - K.cx) * invFx * zU, (v - 0.5 - K.cy) * invFy * zU, zU];
    const pD: Vec3 = [(u + 0.5 - K.cx) * invFx * zD, (v + 1.5 - K.cy) * invFy * zD, zD];

    const du: Vec3 = [pR[0] - pL[0], pR[1] - pL[1], pR[2] - pL[2]];
    const dv: Vec3 = [pD[0] - pU[0], pD[1] - pU[1], pD[2] - pU[2]];
    // Right-handed CV frame (+X right, +Y down, +Z forward): dv x du points toward camera (-Z)
    let n = normalize3(cross3(dv, du));
    if (norm3(n) < 0.5) continue;
    // Flip normal so it always faces the camera (n^T p < 0)
    if (dot3(n, p) > 0) {
      n = [-n[0], -n[1], -n[2]];
    }

    const cosInc = Math.max(0, -dot3(n, p) / Math.max(norm3(p), 1e-6));
    if (opts.filterEnabled && cosInc < 0.20) continue;

    // Pure optical physical precision weight (V3 eq.)
    const w = opts.filterEnabled
      ? (Math.pow(cosInc, 1.5) / (sigma0Sq * z * z * z * z)) * Math.exp(-gradSq * invTwoSigmaGSq)
      : 1.0;

    if (!Number.isFinite(w) || w <= 0) continue;
    out.push({ p, n, w, u, v, z });
  }

  return out;
}
