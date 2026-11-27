import * as THREE from 'three';
import { SeededRNG, type PointNormalSample, type Vec3 } from '../math/se3';

export type RobotCadAssetId = 'panda' | 'iiwa14' | 'ur5e' | 'aloha' | 'openarm';

export type RobotMaterialName =
  | 'white'
  | 'off_white'
  | 'black'
  | 'light_blue'
  | 'green'
  | 'orange'
  | 'gray'
  | 'light_gray'
  | 'linkgray'
  | 'jointgray'
  | 'urblue'
  | 'aloha_black'
  | 'aloha_silver'
  | 'matte_black'
  | 'metal_silver'
  | 'default_material';

export type FrankaMaterialName = RobotMaterialName;

/** Optical properties used ONLY by the virtual RealSense D435i physical noise simulator */
export interface MaterialOpticalProps {
  /** Metallic specular reflectance [0..1] -> causes specular multipath bias & glint dropouts */
  metalness: number;
  /** Surface gloss/clearcoat [0..1] -> narrows and intensifies specular glint */
  gloss: number;
  /** Low-albedo IR absorption [0..1] -> attenuates active IR stereo SNR & causes dark dropouts */
  darkIR: number;
}

export interface PandaCadMeshSpec {
  body: string;
  material: string;
  vertexCount: number;
  indexCount: number;
  indexType: 'u16' | 'u32';
  posOffset: number;
  normOffset: number;
  idxOffset: number;
}

export interface PandaCadManifest {
  source: string;
  totalVertices: number;
  totalTriangles: number;
  meshes: PandaCadMeshSpec[];
  bodyBounds: Record<string, [number, number, number, number, number, number]>;
}

export interface DecodedPandaMesh {
  body: string;
  material: string;
  vertexCount: number;
  indexCount: number;
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint16Array | Uint32Array;
}

export interface PandaCadBundle {
  assetId: RobotCadAssetId;
  manifest: PandaCadManifest;
  meshesByBody: Map<string, DecodedPandaMesh[]>;
  bodySurfaceSamples: Map<string, PointNormalSample[]>;
  bodyBounds: Record<string, [number, number, number, number, number, number]>;
}

function hashString(str: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function computeMeshesTotalArea(meshes: DecodedPandaMesh[]): number {
  let totalArea = 0;
  for (let mIdx = 0; mIdx < meshes.length; mIdx++) {
    const { positions, indices } = meshes[mIdx];
    for (let t = 0; t < indices.length; t += 3) {
      const i0 = indices[t] * 3;
      const i1 = indices[t + 1] * 3;
      const i2 = indices[t + 2] * 3;
      const ax = positions[i1] - positions[i0];
      const ay = positions[i1 + 1] - positions[i0 + 1];
      const az = positions[i1 + 2] - positions[i0 + 2];
      const bx = positions[i2] - positions[i0];
      const by = positions[i2 + 1] - positions[i0 + 1];
      const bz = positions[i2 + 2] - positions[i0 + 2];
      const cx = ay * bz - az * by;
      const cy = az * bx - ax * bz;
      const cz = ax * by - ay * bx;
      totalArea += 0.5 * Math.hypot(cx, cy, cz);
    }
  }
  return totalArea;
}

/**
 * Deterministic area-weighted stratified triangle sampling directly on official
 * robot CAD meshes. Produces zero-mismatch (position, smooth normal) pairs
 * for Stage 2 point-to-plane registration and FIM observability prediction.
 */
function sampleBodyCadSurface(
  assetId: string,
  bodyName: string,
  meshes: DecodedPandaMesh[],
  targetCount: number
): PointNormalSample[] {
  let totalTris = 0;
  for (const m of meshes) {
    totalTris += Math.floor(m.indices.length / 3);
  }
  if (totalTris === 0 || targetCount <= 0) return [];

  const cdf = new Float64Array(totalTris);
  const triMeshIdx = new Uint16Array(totalTris);
  const triStartIdx = new Uint32Array(totalTris);

  let cumArea = 0;
  let triCursor = 0;
  for (let mIdx = 0; mIdx < meshes.length; mIdx++) {
    const { positions, indices } = meshes[mIdx];
    const triCount = Math.floor(indices.length / 3);
    for (let t = 0; t < triCount; t++) {
      const base = t * 3;
      const i0 = indices[base] * 3;
      const i1 = indices[base + 1] * 3;
      const i2 = indices[base + 2] * 3;
      const ax = positions[i1] - positions[i0];
      const ay = positions[i1 + 1] - positions[i0 + 1];
      const az = positions[i1 + 2] - positions[i0 + 2];
      const bx = positions[i2] - positions[i0];
      const by = positions[i2 + 1] - positions[i0 + 1];
      const bz = positions[i2 + 2] - positions[i0 + 2];
      const cx = ay * bz - az * by;
      const cy = az * bx - ax * bz;
      const cz = ax * by - ay * bx;
      const area = 0.5 * Math.hypot(cx, cy, cz);
      cumArea += area;
      cdf[triCursor] = cumArea;
      triMeshIdx[triCursor] = mIdx;
      triStartIdx[triCursor] = base;
      triCursor++;
    }
  }

  if (cumArea <= 1e-12) return [];

  const rng = new SeededRNG(hashString(`${assetId}_cad_v1_${bodyName}`));
  const samples: PointNormalSample[] = [];

  for (let k = 0; k < targetCount; k++) {
    const targetArea = ((k + rng.next()) / targetCount) * cumArea;

    let lo = 0;
    let hi = totalTris - 1;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (cdf[mid] < targetArea) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }

    const mesh = meshes[triMeshIdx[lo]];
    const base = triStartIdx[lo];
    const i0 = mesh.indices[base] * 3;
    const i1 = mesh.indices[base + 1] * 3;
    const i2 = mesh.indices[base + 2] * 3;

    const r1 = Math.sqrt(rng.next());
    const r2 = rng.next();
    const w0 = 1 - r1;
    const w1 = r1 * (1 - r2);
    const w2 = r1 * r2;

    const pos = mesh.positions;
    const norm = mesh.normals;

    const px = w0 * pos[i0] + w1 * pos[i1] + w2 * pos[i2];
    const py = w0 * pos[i0 + 1] + w1 * pos[i1 + 1] + w2 * pos[i2 + 1];
    const pz = w0 * pos[i0 + 2] + w1 * pos[i1 + 2] + w2 * pos[i2 + 2];

    let nx = w0 * norm[i0] + w1 * norm[i1] + w2 * norm[i2];
    let ny = w0 * norm[i0 + 1] + w1 * norm[i1 + 1] + w2 * norm[i2 + 1];
    let nz = w0 * norm[i0 + 2] + w1 * norm[i1 + 2] + w2 * norm[i2 + 2];

    const nLen = Math.hypot(nx, ny, nz);
    if (nLen > 1e-6) {
      nx /= nLen;
      ny /= nLen;
      nz /= nLen;
    } else {
      const ax = pos[i1] - pos[i0];
      const ay = pos[i1 + 1] - pos[i0 + 1];
      const az = pos[i1 + 2] - pos[i0 + 2];
      const bx = pos[i2] - pos[i0];
      const by = pos[i2 + 1] - pos[i0 + 1];
      const bz = pos[i2 + 2] - pos[i0 + 2];
      nx = ay * bz - az * by;
      ny = az * bx - ax * bz;
      nz = ax * by - ay * bx;
      const fLen = Math.hypot(nx, ny, nz);
      if (fLen <= 1e-9) continue;
      nx /= fLen;
      ny /= fLen;
      nz /= fLen;
    }

    const p: Vec3 = [px, py, pz];
    const n: Vec3 = [nx, ny, nz];
    samples.push({ p, n });
  }

  return samples;
}

const bundleCache = new Map<RobotCadAssetId, Promise<PandaCadBundle>>();

/**
 * Lazily fetch, decode, and cache an official robot CAD binary bundle
 * (/assets/<assetId>_cad.json + /assets/<assetId>_cad.bin).
 */
export function loadRobotCadAssets(assetId: RobotCadAssetId): Promise<PandaCadBundle> {
  const existing = bundleCache.get(assetId);
  if (existing) return existing;

  const promise = (async () => {
    const jsonUrl = `/assets/${assetId}_cad.json`;
    const binUrl = `/assets/${assetId}_cad.bin`;
    const [manifestRes, binRes] = await Promise.all([fetch(jsonUrl), fetch(binUrl)]);

    if (!manifestRes.ok) {
      throw new Error(`无法加载官方 CAD 清单 (${jsonUrl}): HTTP ${manifestRes.status}`);
    }
    if (!binRes.ok) {
      throw new Error(`无法加载官方 CAD 网格二进制 (${binUrl}): HTTP ${binRes.status}`);
    }

    const [manifest, binBuf] = await Promise.all([
      manifestRes.json() as Promise<PandaCadManifest>,
      binRes.arrayBuffer(),
    ]);

    const meshesByBody = new Map<string, DecodedPandaMesh[]>();

    for (const spec of manifest.meshes) {
      const positions = new Float32Array(
        binBuf.slice(spec.posOffset, spec.posOffset + spec.vertexCount * 3 * 4)
      );

      const rawNorm = new Int16Array(binBuf, spec.normOffset, spec.vertexCount * 3);
      const normals = new Float32Array(spec.vertexCount * 3);
      for (let i = 0; i < spec.vertexCount * 3; i += 3) {
        const x = rawNorm[i] / 32767;
        const y = rawNorm[i + 1] / 32767;
        const z = rawNorm[i + 2] / 32767;
        const len = Math.hypot(x, y, z);
        if (len > 1e-6) {
          normals[i] = x / len;
          normals[i + 1] = y / len;
          normals[i + 2] = z / len;
        } else {
          normals[i] = 0;
          normals[i + 1] = 0;
          normals[i + 2] = 1;
        }
      }

      const indices =
        spec.indexType === 'u16'
          ? new Uint16Array(binBuf.slice(spec.idxOffset, spec.idxOffset + spec.indexCount * 2))
          : new Uint32Array(binBuf.slice(spec.idxOffset, spec.idxOffset + spec.indexCount * 4));

      const decoded: DecodedPandaMesh = {
        body: spec.body,
        material: spec.material,
        vertexCount: spec.vertexCount,
        indexCount: spec.indexCount,
        positions,
        normals,
        indices,
      };

      const arr = meshesByBody.get(spec.body);
      if (arr) {
        arr.push(decoded);
      } else {
        meshesByBody.set(spec.body, [decoded]);
      }
    }

    // Area-proportional surface sampling (~4,200 total points across official robot bodies)
    const bodyAreas = new Map<string, number>();
    let totalRobotArea = 0;
    for (const [bodyName, meshes] of meshesByBody.entries()) {
      const area = computeMeshesTotalArea(meshes);
      bodyAreas.set(bodyName, area);
      totalRobotArea += area;
    }

    const TARGET_TOTAL_SAMPLES = 4200;
    const bodySurfaceSamples = new Map<string, PointNormalSample[]>();
    for (const [bodyName, meshes] of meshesByBody.entries()) {
      const area = bodyAreas.get(bodyName) ?? 0;
      const frac = totalRobotArea > 0 ? area / totalRobotArea : 0;
      const count = Math.max(100, Math.round(frac * TARGET_TOTAL_SAMPLES));
      bodySurfaceSamples.set(bodyName, sampleBodyCadSurface(assetId, bodyName, meshes, count));
    }

    return {
      assetId,
      manifest,
      meshesByBody,
      bodySurfaceSamples,
      bodyBounds: manifest.bodyBounds,
    };
  })().catch((err) => {
    bundleCache.delete(assetId);
    throw err;
  });

  bundleCache.set(assetId, promise);
  return promise;
}

export function loadPandaCadAssets(): Promise<PandaCadBundle> {
  return loadRobotCadAssets('panda');
}

/** Per-material optical signature for realistic RealSense D435i material degradation */
export const MATERIAL_OPTICAL_PROPS: Record<RobotMaterialName, MaterialOpticalProps> = {
  // Franka Panda
  white: { metalness: 0.06, gloss: 0.72, darkIR: 0.04 },
  off_white: { metalness: 0.46, gloss: 0.64, darkIR: 0.10 },
  black: { metalness: 0.34, gloss: 0.58, darkIR: 0.78 },
  light_blue: { metalness: 0.12, gloss: 0.78, darkIR: 0.14 },
  green: { metalness: 0.05, gloss: 0.82, darkIR: 0.10 },
  // KUKA LBR iiwa 14
  orange: { metalness: 0.14, gloss: 0.76, darkIR: 0.08 },
  gray: { metalness: 0.38, gloss: 0.50, darkIR: 0.42 },
  light_gray: { metalness: 0.78, gloss: 0.74, darkIR: 0.08 },
  // Universal Robots UR5e (high-specularity brushed stainless steel tubes!)
  linkgray: { metalness: 0.88, gloss: 0.82, darkIR: 0.05 },
  jointgray: { metalness: 0.44, gloss: 0.54, darkIR: 0.48 },
  urblue: { metalness: 0.14, gloss: 0.70, darkIR: 0.12 },
  // ALOHA ViperX 300s (IR-absorbing matte black anodized!)
  aloha_black: { metalness: 0.22, gloss: 0.34, darkIR: 0.88 },
  aloha_silver: { metalness: 0.68, gloss: 0.72, darkIR: 0.12 },
  // OpenArm v1.0 (matte black carbon + bright CNC aluminum!)
  matte_black: { metalness: 0.18, gloss: 0.38, darkIR: 0.82 },
  metal_silver: { metalness: 0.84, gloss: 0.78, darkIR: 0.06 },
  default_material: { metalness: 0.10, gloss: 0.42, darkIR: 0.22 },
};

export function getMaterialOpticalProps(matName: string): MaterialOpticalProps {
  return (
    MATERIAL_OPTICAL_PROPS[matName as RobotMaterialName] ?? {
      metalness: 0.2,
      gloss: 0.45,
      darkIR: 0.2,
    }
  );
}

/**
 * Authentic multi-robot industrial PBR material palette covering
 * Franka Panda, KUKA LBR iiwa 14, Universal Robots UR5e, ALOHA ViperX 300s, and OpenArm v1.0.
 */
export function createRobotMaterialPalette(): Record<RobotMaterialName, THREE.Material> {
  return {
    // Franka Emika Panda
    white: new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0xdce2ea),
      roughness: 0.34,
      metalness: 0.06,
      clearcoat: 0.28,
      clearcoatRoughness: 0.26,
      envMapIntensity: 0.52,
    }),
    off_white: new THREE.MeshStandardMaterial({
      color: new THREE.Color(0xb6c0cc),
      roughness: 0.36,
      metalness: 0.46,
      envMapIntensity: 0.72,
    }),
    black: new THREE.MeshStandardMaterial({
      color: new THREE.Color(0x1b2028),
      roughness: 0.38,
      metalness: 0.36,
      envMapIntensity: 0.75,
    }),
    light_blue: new THREE.MeshStandardMaterial({
      color: new THREE.Color(0x0a95db),
      roughness: 0.22,
      metalness: 0.12,
      emissive: new THREE.Color(0x0772a8),
      emissiveIntensity: 0.52,
      envMapIntensity: 0.85,
    }),
    green: new THREE.MeshStandardMaterial({
      color: new THREE.Color(0x1ce872),
      roughness: 0.18,
      metalness: 0.05,
      emissive: new THREE.Color(0x10b858),
      emissiveIntensity: 0.75,
      envMapIntensity: 0.85,
    }),

    // KUKA LBR iiwa 14
    orange: new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0xf5660a),
      roughness: 0.26,
      metalness: 0.14,
      clearcoat: 0.38,
      clearcoatRoughness: 0.20,
      envMapIntensity: 0.78,
    }),
    gray: new THREE.MeshStandardMaterial({
      color: new THREE.Color(0x4b5360),
      roughness: 0.44,
      metalness: 0.38,
      envMapIntensity: 0.72,
    }),
    light_gray: new THREE.MeshStandardMaterial({
      color: new THREE.Color(0xc2cbd6),
      roughness: 0.24,
      metalness: 0.78,
      envMapIntensity: 0.95,
    }),

    // Universal Robots UR5e
    linkgray: new THREE.MeshStandardMaterial({
      color: new THREE.Color(0xcfd6de),
      roughness: 0.20,
      metalness: 0.86,
      envMapIntensity: 1.05,
    }),
    jointgray: new THREE.MeshStandardMaterial({
      color: new THREE.Color(0x3b424d),
      roughness: 0.42,
      metalness: 0.44,
      envMapIntensity: 0.78,
    }),
    urblue: new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0x5ea8d4),
      roughness: 0.30,
      metalness: 0.14,
      clearcoat: 0.26,
      clearcoatRoughness: 0.24,
      envMapIntensity: 0.72,
    }),

    // ALOHA ViperX 300s (matte charcoal anodized aluminum)
    aloha_black: new THREE.MeshStandardMaterial({
      color: new THREE.Color(0x232830),
      roughness: 0.56,
      metalness: 0.24,
      envMapIntensity: 0.65,
    }),
    aloha_silver: new THREE.MeshStandardMaterial({
      color: new THREE.Color(0xb8c2ce),
      roughness: 0.28,
      metalness: 0.68,
      envMapIntensity: 0.88,
    }),

    // OpenArm v1.0 (matte black + CNC milled silver)
    matte_black: new THREE.MeshStandardMaterial({
      color: new THREE.Color(0x262c35),
      roughness: 0.54,
      metalness: 0.18,
      envMapIntensity: 0.62,
    }),
    metal_silver: new THREE.MeshStandardMaterial({
      color: new THREE.Color(0xc8d0da),
      roughness: 0.22,
      metalness: 0.82,
      envMapIntensity: 1.0,
    }),
    default_material: new THREE.MeshStandardMaterial({
      color: new THREE.Color(0x7c8694),
      roughness: 0.52,
      metalness: 0.12,
      envMapIntensity: 0.55,
    }),
  };
}

export const createFrankaMaterialPalette = createRobotMaterialPalette;
