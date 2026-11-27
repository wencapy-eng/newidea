import {
  cross3,
  dot3,
  norm3,
  normalize3,
  SpatialHashGrid3D,
  type Vec3,
} from '../math/se3';
import type {
  CameraIntrinsics,
  RGBDFrame,
  SyncComponentCurve,
  SyncDiagnostics,
  WeightedPoint,
} from './types';

export const BG_DIFF_THRESHOLD_M = 0.032;
export const SYNC_CORR_THRESHOLD = 0.42;

/** Backproject pixel (u, v) with top-left origin into CV camera frame {C} (+X right, +Y down, +Z forward) */
export function backprojectPixel(
  u: number,
  v: number,
  z: number,
  K: CameraIntrinsics
): Vec3 {
  return [
    ((u + 0.5 - K.cx) / K.fx) * z,
    ((v + 0.5 - K.cy) / K.fy) * z,
    z,
  ];
}

/**
 * Compute per-pixel robust max-depth background envelope D_bg(u, v) over valid depth measurements (z > 0.08m).
 * Uses the K-th largest valid depth where K scales with frame count (K=2 for short 16-frame action sweeps;
 * K=5..7 for long 64..128-frame global sequences) so isolated positive stereo mismatch outlier spikes
 * (up to 2.5% rate) cannot inflate the background envelope while preserving uncovered background pixels.
 */
export function computeMaxDepthEnvelope(
  frames: readonly RGBDFrame[],
  width: number,
  height: number
): Float32Array {
  const T = frames.length;
  const nPix = width * height;
  const K = T <= 24 ? 2 : Math.max(3, Math.min(8, Math.round(T * 0.065)));

  if (K === 2) {
    const max1 = new Float32Array(nPix);
    const max2 = new Float32Array(nPix);
    const counts = new Uint16Array(nPix);
    for (let f = 0; f < T; f++) {
      const d = frames[f].depth;
      for (let i = 0; i < nPix; i++) {
        const z = d[i];
        if (z > 0.08) {
          counts[i]++;
          if (z > max1[i]) {
            max2[i] = max1[i];
            max1[i] = z;
          } else if (z > max2[i]) {
            max2[i] = z;
          }
        }
      }
    }
    const out = new Float32Array(nPix);
    for (let i = 0; i < nPix; i++) {
      out[i] = counts[i] >= 6 && max2[i] > 0.08 ? max2[i] : max1[i];
    }
    return out;
  }

  // General top-K buffer (descending order per pixel: top[i*K + 0] >= ... >= top[i*K + K-1])
  const top = new Float32Array(nPix * K);
  const counts = new Uint16Array(nPix);
  for (let f = 0; f < T; f++) {
    const d = frames[f].depth;
    for (let i = 0; i < nPix; i++) {
      const z = d[i];
      if (z <= 0.08) continue;
      counts[i]++;
      const base = i * K;
      if (z <= top[base + K - 1]) continue;
      // Insert z into sorted descending top-K slot
      let pos = K - 1;
      while (pos > 0 && z > top[base + pos - 1]) {
        top[base + pos] = top[base + pos - 1];
        pos--;
      }
      top[base + pos] = z;
    }
  }

  const out = new Float32Array(nPix);
  for (let i = 0; i < nPix; i++) {
    const c = counts[i];
    if (c === 0) continue;
    const base = i * K;
    // Choose K-th largest when enough valid samples exist; gracefully fall back for sparsely uncovered pixels
    const pickIdx = c >= K + 4 ? K - 1 : c >= 6 ? Math.min(1, K - 1) : 0;
    const val = top[base + pickIdx];
    out[i] = val > 0.08 ? val : top[base];
  }
  return out;
}

/** One-sided foreground mask: D_t(u, v) < D_bg(u, v) - threshold */
export function computeOneSidedForegroundMask(
  depth: Float32Array,
  bgDepth: Float32Array,
  thresholdM = BG_DIFF_THRESHOLD_M
): Uint8Array {
  const n = depth.length;
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const z = depth[i];
    const zb = bgDepth[i];
    if (z > 0.08 && zb > 0.08 && z < zb - thresholdM) {
      mask[i] = 1;
    }
  }
  return mask;
}

/**
 * Compute lagged Pearson cross-correlation max_{lag in [-maxLag, +maxLag]} corr(a[t], b[t + lag])
 * between a connected component's temporal depth activity curve and encoder speed |qdot(t)|.
 */
export function computeMaxLaggedPearson(
  signalA: readonly number[],
  signalB: readonly number[],
  maxLag = 2
): { rho: number; bestLag: number } {
  const T = Math.min(signalA.length, signalB.length);
  if (T < 4) return { rho: 0, bestLag: 0 };

  let bestRho = -2;
  let bestLag = 0;

  for (let lag = -maxLag; lag <= maxLag; lag++) {
    let sumA = 0;
    let sumB = 0;
    let count = 0;
    for (let t = 0; t < T; t++) {
      const tb = t + lag;
      if (tb < 0 || tb >= T) continue;
      sumA += signalA[t];
      sumB += signalB[tb];
      count++;
    }
    if (count < 4) continue;
    const meanA = sumA / count;
    const meanB = sumB / count;
    let num = 0;
    let varA = 0;
    let varB = 0;
    for (let t = 0; t < T; t++) {
      const tb = t + lag;
      if (tb < 0 || tb >= T) continue;
      const da = signalA[t] - meanA;
      const db = signalB[tb] - meanB;
      num += da * db;
      varA += da * da;
      varB += db * db;
    }
    const denom = Math.sqrt(varA * varB);
    const rho = denom > 1e-12 ? num / denom : 0;
    if (rho > bestRho) {
      bestRho = rho;
      bestLag = lag;
    }
  }

  return { rho: bestRho < -1 ? 0 : bestRho, bestLag };
}

/** 2D BFS connected components on binary mask (8-connected) */
function findConnectedComponents(
  mask: Uint8Array,
  width: number,
  height: number,
  minPixels = 35
): Array<{ id: number; pixels: Int32Array }> {
  const n = width * height;
  const visited = new Uint8Array(n);
  const queue = new Int32Array(n);
  const components: Array<{ id: number; pixels: Int32Array }> = [];

  for (let start = 0; start < n; start++) {
    if (mask[start] === 0 || visited[start] === 1) continue;
    let head = 0;
    let tail = 0;
    queue[tail++] = start;
    visited[start] = 1;

    while (head < tail) {
      const cur = queue[head++];
      const x = cur % width;
      const y = (cur - x) / width;

      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        const rowOff = ny * width;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          const nIdx = rowOff + nx;
          if (mask[nIdx] === 1 && visited[nIdx] === 0) {
            visited[nIdx] = 1;
            queue[tail++] = nIdx;
          }
        }
      }
    }

    if (tail >= minPixels) {
      components.push({
        id: components.length + 1,
        pixels: queue.slice(0, tail),
      });
    }
  }

  components.sort((a, b) => b.pixels.length - a.pixels.length);
  return components;
}

/**
 * Strip isolated tiny connected components (< minPixels) from a binary foreground mask
 * so scattered negative background stereo outlier spikes cannot leak into Stage 2 static point clouds.
 */
export function filterMaskSmallComponents(
  mask: Uint8Array,
  width: number,
  height: number,
  minPixels = 40
): Uint8Array {
  const comps = findConnectedComponents(mask, width, height, minPixels);
  const out = new Uint8Array(width * height);
  for (let c = 0; c < comps.length; c++) {
    const pix = comps[c].pixels;
    for (let k = 0; k < pix.length; k++) {
      out[pix[k]] = 1;
    }
  }
  return out;
}

/** Normalize a non-negative time series into [0, 1] for visualization */
function normalizeSeries(arr: readonly number[]): number[] {
  let maxV = 0;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > maxV) maxV = arr[i];
  }
  if (maxV < 1e-12) return arr.map(() => 0);
  return arr.map((v) => v / maxV);
}

export interface ActionForegroundResult {
  localBgDepth: Float32Array;
  /** Per-frame active masks filtered by timestamp sync (if enabled) */
  frameMasks: Uint8Array[];
  /** Union mask of rejected (asynchronous distractor) regions */
  rejectedRegionMask: Uint8Array;
  diagnostics: SyncDiagnostics;
}

/**
 * Layer 1 (Local Action Envelope) + Command-Timestamp Synchronized Correlation Filtering:
 *  - Builds D_bg^(k) = max_{t in T_k} D_t(u, v)
 *  - Computes one-sided foreground masks D_t < D_bg^(k) - 0.032
 *  - Clusters active union mask with 2D BFS
 *  - Computes each component's temporal depth activity curve and lagged (lag in [-2, 2])
 *    Pearson cross-correlation against encoder |qdot(t)|
 *  - Keeps synchronous robot components (rho >= 0.42), rejects asynchronous distractors
 */
export function segmentActionWithTimestampSync(
  frames: readonly RGBDFrame[],
  actionLabel: string,
  useTimestampSync: boolean
): ActionForegroundResult {
  const T = frames.length;
  const W = frames[0].width;
  const H = frames[0].height;
  const nPix = W * H;

  const localBgDepth = computeMaxDepthEnvelope(frames, W, H);
  const rawMasks = frames.map((f) =>
    computeOneSidedForegroundMask(f.depth, localBgDepth, BG_DIFF_THRESHOLD_M)
  );

  // Union mask across all frames in this action
  const unionMask = new Uint8Array(nPix);
  for (let t = 0; t < T; t++) {
    const m = rawMasks[t];
    for (let i = 0; i < nPix; i++) {
      if (m[i]) unionMask[i] = 1;
    }
  }

  // Slight 1px dilation before BFS bridges thin cracks from depth holes while keeping distractor separate
  const bfsMask = new Uint8Array(unionMask);
  for (let y = 1; y < H - 1; y++) {
    const row = y * W;
    for (let x = 1; x < W - 1; x++) {
      const idx = row + x;
      if (unionMask[idx]) continue;
      if (
        unionMask[idx - 1] &&
        unionMask[idx + 1] &&
        unionMask[idx - W] &&
        unionMask[idx + W]
      ) {
        bfsMask[idx] = 1;
      }
    }
  }

  const components = findConnectedComponents(bfsMask, W, H, 45);

  // Encoder speed profile |qdot_active(t)|
  const activeJoint = frames[0].activeJoint;
  const rawEncoderSpeed = frames.map((f) => Math.abs(f.qdotEncoder[activeJoint] ?? 0));
  const encoderSpeed = normalizeSeries(rawEncoderSpeed);

  const allowedRegion = new Uint8Array(nPix);
  const rejectedRegionMask = new Uint8Array(nPix);
  const compCurves: SyncComponentCurve[] = [];

  for (let c = 0; c < components.length; c++) {
    const comp = components[c];
    const pixels = comp.pixels;
    const rawActivity = new Array<number>(T).fill(0);

    // Compute uniform 1-step adjacent depth activity diff(t, t+1) with a 15mm deadzone so
    // static Gaussian/material noise, specular glint hole flicker, and outlier spikes do NOT
    // create artificial boundary spikes at pause frames (F0, F1, F14, F15).
    const stepDiff = new Array<number>(Math.max(0, T - 1)).fill(0);
    for (let t = 0; t < T - 1; t++) {
      const dA = frames[t].depth;
      const dB = frames[t + 1].depth;
      let sumDiff = 0;
      let validCnt = 0;
      for (let k = 0; k < pixels.length; k++) {
        const idx = pixels[k];
        const z1 = dA[idx];
        const z2 = dB[idx];
        if (z1 > 0.08 && z2 > 0.08) {
          const absD = Math.abs(z2 - z1);
          // Reject isolated stereo mismatch spikes (> 0.22m in a single 65ms step)
          if (absD > 0.22) continue;
          // Soft 15mm deadzone suppresses static sensor noise & dark-IR variance floor
          const eff = Math.max(0, Math.min(0.14, absD) - 0.015);
          sumDiff += eff;
          validCnt++;
        }
      }
      stepDiff[t] = validCnt > 0 ? sumDiff / validCnt : 0;
    }

    for (let t = 0; t < T; t++) {
      if (T <= 1) {
        rawActivity[t] = 0;
      } else if (t === 0) {
        rawActivity[t] = stepDiff[0];
      } else if (t === T - 1) {
        rawActivity[t] = stepDiff[T - 2];
      } else {
        rawActivity[t] = 0.5 * (stepDiff[t - 1] + stepDiff[t]);
      }
    }

    const { rho, bestLag } = computeMaxLaggedPearson(rawActivity, rawEncoderSpeed, 2);
    const accepted = useTimestampSync ? rho >= SYNC_CORR_THRESHOLD : true;

    compCurves.push({
      componentId: comp.id,
      pixelCount: pixels.length,
      rho,
      bestLag,
      accepted,
      activity: normalizeSeries(rawActivity),
    });

    const target = accepted ? allowedRegion : rejectedRegionMask;
    for (let k = 0; k < pixels.length; k++) {
      target[pixels[k]] = 1;
    }
  }

  // Fallback safeguard only if sync filter rejected all components due to extreme noise
  let acceptedPixels = 0;
  let rejectedPixels = 0;
  for (let i = 0; i < nPix; i++) {
    if (allowedRegion[i]) acceptedPixels++;
    if (rejectedRegionMask[i]) rejectedPixels++;
  }

  if (useTimestampSync && acceptedPixels === 0 && compCurves.length > 0) {
    // Keep highest-correlation component so solver doesn't receive empty cloud
    let bestIdx = 0;
    for (let c = 1; c < compCurves.length; c++) {
      if (compCurves[c].rho > compCurves[bestIdx].rho) bestIdx = c;
    }
    compCurves[bestIdx].accepted = true;
    const pix = components[bestIdx].pixels;
    for (let k = 0; k < pix.length; k++) {
      allowedRegion[pix[k]] = 1;
      rejectedRegionMask[pix[k]] = 0;
    }
    acceptedPixels = pix.length;
    rejectedPixels = Math.max(0, rejectedPixels - pix.length);
  }

  const frameMasks = rawMasks.map((raw) => {
    const out = new Uint8Array(nPix);
    for (let i = 0; i < nPix; i++) {
      if (raw[i] && allowedRegion[i]) out[i] = 1;
    }
    return out;
  });

  return {
    localBgDepth,
    frameMasks,
    rejectedRegionMask,
    diagnostics: {
      actionLabel,
      jointIndex: activeJoint,
      encoderSpeed,
      components: compCurves,
      acceptedPixels,
      rejectedPixels,
    },
  };
}

/** 2D binary morphological erosion with square radius `r` (default 2px) */
export function erodeMask(
  mask: Uint8Array,
  width: number,
  height: number,
  radius = 2
): Uint8Array {
  const out = new Uint8Array(width * height);
  for (let y = radius; y < height - radius; y++) {
    const rowOff = y * width;
    for (let x = radius; x < width - radius; x++) {
      const idx = rowOff + x;
      if (mask[idx] === 0) continue;
      let keep = 1;
      for (let dy = -radius; dy <= radius && keep; dy++) {
        const nRow = (y + dy) * width;
        for (let dx = -radius; dx <= radius; dx++) {
          if (mask[nRow + x + dx] === 0) {
            keep = 0;
            break;
          }
        }
      }
      out[idx] = keep;
    }
  }
  return out;
}

/**
 * 3x3 morphological interior pinhole fill (2 passes): closes isolated 1-2px specular glint
 * and matte-black IR dropout holes inside the foreground mask BEFORE outer silhouette erosion
 * so 2px boundary erosion strips flying pixels without hollowing out metallic/dark robot links.
 */
export function fillInteriorMaskPinholes(mask: Uint8Array, width: number, height: number): Uint8Array {
  let cur = new Uint8Array(mask);
  for (let pass = 0; pass < 2; pass++) {
    const next = new Uint8Array(cur);
    for (let y = 1; y < height - 1; y++) {
      const row = y * width;
      for (let x = 1; x < width - 1; x++) {
        const idx = row + x;
        if (cur[idx]) continue;
        const cnt =
          cur[idx - width - 1] +
          cur[idx - width] +
          cur[idx - width + 1] +
          cur[idx - 1] +
          cur[idx + 1] +
          cur[idx + width - 1] +
          cur[idx + width] +
          cur[idx + width + 1];
        if (cnt >= 5) {
          next[idx] = 1;
        }
      }
    }
    cur = next;
  }
  return cur;
}

/**
 * Extract oriented, physically weighted 3D point cloud from a depth frame + binary foreground mask:
 *  - Interior pinhole closing + 2px outer silhouette erosion + 3x3 median stereo-outlier despike (when filterEnabled=true)
 *  - Hole- & spike-tolerant local surface normal estimation forced toward camera (n^T q < 0)
 *  - 3D Spatial Hash Grid Statistical Outlier Removal (SOR) (when filterEnabled=true)
 *  - Physical uncertainty weight:
 *      w_i = (max(0, -(n^c)^T q^c / ||q^c||)^gamma / (sigma0^2 * z^4)) * exp(-||grad D||^2 / (2 sigma_g^2))
 *    normalized to (0, 1].
 */
export function extractWeightedPointCloud(
  depth: Float32Array,
  mask: Uint8Array,
  K: CameraIntrinsics,
  opts: {
    filterEnabled: boolean;
    sigmaZ0: number;
    maxPoints?: number;
    pixelStride?: number;
  }
): WeightedPoint[] {
  const W = K.width;
  const H = K.height;
  const stride = opts.pixelStride ?? 1;
  const maxPoints = opts.maxPoints ?? 900;
  const filterOn = opts.filterEnabled;

  // Step 1: Fill isolated interior glint/IR pinholes, then erode 2px outer silhouette (when enabled)
  const closedMask = filterOn ? fillInteriorMaskPinholes(mask, W, H) : mask;
  const workMask = filterOn ? erodeMask(closedMask, W, H, 2) : mask;

  // Step 2: 3x3 median despike + silhouette jump rejection + hole-tolerant normals + physical weights
  const rawPoints: WeightedPoint[] = [];
  const sigma0 = Math.max(opts.sigmaZ0, 0.0004);
  const sigma0Sq = sigma0 * sigma0;
  const sigmaG = 0.018;
  const twoSigmaGSq = 2 * sigmaG * sigmaG;
  const gamma = 1.5;
  const nbVals = new Float64Array(8);

  const isCleanNeighbor = (zn: number, zRef: number): boolean =>
    zn > 0.08 && Math.abs(zn - zRef) <= (filterOn ? 0.048 : 0.15);

  for (let v = 2; v < H - 2; v += stride) {
    const row = v * W;
    for (let u = 2; u < W - 2; u += stride) {
      const idx = row + u;
      if (!workMask[idx]) continue;
      const z = depth[idx];
      if (z <= 0.1 || z >= 5.0) continue;

      // 3x3 valid-neighbor median check rejects isolated stereo mismatch outlier spikes
      // AND silhouette jumps without losing valid interior pixels next to 1px metallic/IR pinholes
      let nbCount = 0;
      let jumpCount = 0;
      for (let dy = -1; dy <= 1; dy++) {
        const nRow = (v + dy) * W;
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          const zn = depth[nRow + u + dx];
          if (zn > 0.08) {
            nbVals[nbCount++] = zn;
            if (Math.abs(zn - z) > 0.05) jumpCount++;
          }
        }
      }
      if (nbCount < 4) continue;

      if (filterOn) {
        // Insertion sort tiny array (length 4..8) to get exact 8-neighbor median
        for (let i = 1; i < nbCount; i++) {
          const val = nbVals[i];
          let j = i - 1;
          while (j >= 0 && nbVals[j] > val) {
            nbVals[j + 1] = nbVals[j];
            j--;
          }
          nbVals[j + 1] = val;
        }
        const med = nbVals[nbCount >> 1];
        // Reject single-pixel stereo mismatch spike or flying pixel off local surface
        if (Math.abs(z - med) > 0.040) continue;
        // Reject pixels sitting directly on a multi-pixel depth discontinuity edge
        if (jumpCount >= 3) continue;
      }

      // Hole- & spike-tolerant tangent estimation (prefer 2px baseline, gracefully fall back to 1px or one-sided)
      const q = backprojectPixel(u, v, z, K);

      const z2L = depth[idx - 2];
      const z2R = depth[idx + 2];
      const z1L = depth[idx - 1];
      const z1R = depth[idx + 1];
      let dPdu: Vec3 | null = null;
      let dzU = 0;

      if (isCleanNeighbor(z2L, z) && isCleanNeighbor(z2R, z)) {
        const pL = backprojectPixel(u - 2, v, z2L, K);
        const pR = backprojectPixel(u + 2, v, z2R, K);
        dPdu = [pR[0] - pL[0], pR[1] - pL[1], pR[2] - pL[2]];
        dzU = (z2R - z2L) * 0.25;
      } else if (isCleanNeighbor(z1L, z) && isCleanNeighbor(z1R, z)) {
        const pL = backprojectPixel(u - 1, v, z1L, K);
        const pR = backprojectPixel(u + 1, v, z1R, K);
        dPdu = [pR[0] - pL[0], pR[1] - pL[1], pR[2] - pL[2]];
        dzU = (z1R - z1L) * 0.5;
      } else if (isCleanNeighbor(z1R, z)) {
        const pR = backprojectPixel(u + 1, v, z1R, K);
        dPdu = [pR[0] - q[0], pR[1] - q[1], pR[2] - q[2]];
        dzU = z1R - z;
      } else if (isCleanNeighbor(z1L, z)) {
        const pL = backprojectPixel(u - 1, v, z1L, K);
        dPdu = [q[0] - pL[0], q[1] - pL[1], q[2] - pL[2]];
        dzU = z - z1L;
      }
      if (!dPdu) continue;

      const z2U = depth[idx - 2 * W];
      const z2D = depth[idx + 2 * W];
      const z1U = depth[idx - W];
      const z1D = depth[idx + W];
      let dPdv: Vec3 | null = null;
      let dzV = 0;

      if (isCleanNeighbor(z2U, z) && isCleanNeighbor(z2D, z)) {
        const pU = backprojectPixel(u, v - 2, z2U, K);
        const pD = backprojectPixel(u, v + 2, z2D, K);
        dPdv = [pD[0] - pU[0], pD[1] - pU[1], pD[2] - pU[2]];
        dzV = (z2D - z2U) * 0.25;
      } else if (isCleanNeighbor(z1U, z) && isCleanNeighbor(z1D, z)) {
        const pU = backprojectPixel(u, v - 1, z1U, K);
        const pD = backprojectPixel(u, v + 1, z1D, K);
        dPdv = [pD[0] - pU[0], pD[1] - pU[1], pD[2] - pU[2]];
        dzV = (z1D - z1U) * 0.5;
      } else if (isCleanNeighbor(z1D, z)) {
        const pD = backprojectPixel(u, v + 1, z1D, K);
        dPdv = [pD[0] - q[0], pD[1] - q[1], pD[2] - q[2]];
        dzV = z1D - z;
      } else if (isCleanNeighbor(z1U, z)) {
        const pU = backprojectPixel(u, v - 1, z1U, K);
        dPdv = [q[0] - pU[0], q[1] - pU[1], q[2] - pU[2]];
        dzV = z - z1U;
      }
      if (!dPdv) continue;

      const gradSq = dzU * dzU + dzV * dzV;
      if (filterOn && Math.sqrt(gradSq) > 0.042) continue;

      let n = normalize3(cross3(dPdv, dPdu));
      if (norm3(n) < 0.5) continue;

      // Enforce camera-facing orientation: n^T q < 0
      if (dot3(n, q) > 0) {
        n = [-n[0], -n[1], -n[2]];
      }

      const qNorm = norm3(q);
      const cosInc = Math.max(0, -dot3(n, q) / Math.max(qNorm, 1e-6));
      if (filterOn && cosInc < 0.16) continue;

      let w = 1.0;
      if (filterOn) {
        const z4 = z * z * z * z;
        w = (Math.pow(cosInc, gamma) / (sigma0Sq * z4)) * Math.exp(-gradSq / twoSigmaGSq);
      }

      rawPoints.push({ p: q, n, w, u, v, z });
    }
  }

  if (rawPoints.length === 0) return [];

  // Pre-cap raw points deterministically before O(N) 3D SOR to keep frame extraction fast
  let workPoints = rawPoints;
  const sorCap = Math.floor(maxPoints * 1.35);
  if (rawPoints.length > sorCap) {
    const capped: WeightedPoint[] = [];
    const step = rawPoints.length / sorCap;
    for (let i = 0; i < sorCap; i++) {
      capped.push(rawPoints[Math.floor(i * step)]);
    }
    workPoints = capped;
  }

  // Step 3: 3D Spatial Hash Grid Statistical Outlier Removal (SOR)
  let filtered = workPoints;
  if (filterOn && workPoints.length > 30) {
    const grid = new SpatialHashGrid3D<WeightedPoint>(0.035);
    grid.insertAll(workPoints);
    const meanDists = new Float64Array(workPoints.length);
    const neighborCounts = new Int32Array(workPoints.length);

    let globalSum = 0;
    let validMeanCnt = 0;

    for (let i = 0; i < workPoints.length; i++) {
      const pt = workPoints[i];
      const nbrs = grid.findWithinRadius(pt.p, 0.038);
      if (nbrs.length <= 3) {
        neighborCounts[i] = nbrs.length;
        meanDists[i] = 1.0;
        continue;
      }
      let dSum = 0;
      let cnt = 0;
      for (let j = 0; j < nbrs.length; j++) {
        const o = nbrs[j];
        if (o === pt) continue;
        const d = Math.hypot(o.p[0] - pt.p[0], o.p[1] - pt.p[1], o.p[2] - pt.p[2]);
        dSum += d;
        cnt++;
      }
      neighborCounts[i] = cnt;
      const md = cnt > 0 ? dSum / cnt : 1.0;
      meanDists[i] = md;
      globalSum += md;
      validMeanCnt++;
    }

    if (validMeanCnt > 15) {
      const mu = globalSum / validMeanCnt;
      let varSum = 0;
      for (let i = 0; i < workPoints.length; i++) {
        if (neighborCounts[i] > 3) {
          const d = meanDists[i] - mu;
          varSum += d * d;
        }
      }
      const std = Math.sqrt(varSum / validMeanCnt);
      const maxAllow = mu + 1.5 * std;
      const sorOut: WeightedPoint[] = [];
      for (let i = 0; i < workPoints.length; i++) {
        if (neighborCounts[i] >= 4 && meanDists[i] <= maxAllow) {
          sorOut.push(workPoints[i]);
        }
      }
      if (sorOut.length >= 20) {
        filtered = sorOut;
      }
    }
  }

  // Normalize weights to (0, 1]
  let maxW = 0;
  for (let i = 0; i < filtered.length; i++) {
    if (filtered[i].w > maxW) maxW = filtered[i].w;
  }
  if (maxW > 1e-18) {
    for (let i = 0; i < filtered.length; i++) {
      filtered[i].w = filterOn ? Math.max(1e-4, filtered[i].w / maxW) : 1.0;
    }
  }

  // Deterministic uniform subsampling to respect maxPoints budget
  if (filtered.length > maxPoints) {
    const out: WeightedPoint[] = [];
    const step = filtered.length / maxPoints;
    for (let i = 0; i < maxPoints; i++) {
      out.push(filtered[Math.floor(i * step)]);
    }
    return out;
  }

  return filtered;
}
