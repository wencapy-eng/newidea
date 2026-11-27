import loadMujoco, { type MainModule, type MjData, type MjModel } from '@mujoco/mujoco';
import mujocoWasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
import {
  cross3,
  normalize3,
  type Mat3,
  type PointNormalSample,
  type Vec3,
  type Vec6,
} from '../math/se3';
import { solveDeltaFK } from './deltaKinematics';
import {
  loadRobotCadAssets,
  type DecodedPandaMesh,
  type PandaCadBundle,
} from './pandaCadAssets';
import { getRobotSpec, type RobotId, type RobotSpec } from './robotCatalog';

export type GeomCategory = 'robot' | 'custom_tool' | 'distractor' | 'environment';

export interface GeomMeta {
  id: number;
  name: string;
  bodyId: number;
  bodyName: string;
  type: number; // MuJoCo mjtGeom value (2=sphere, 3=capsule, 5=cylinder, 6=box)
  size: [number, number, number];
  rgba: [number, number, number, number];
  category: GeomCategory;
  /** Lowest joint index (0..dof-1) that moves this geom, or -1 if static / non-robot */
  minDrivingJoint: number;
  /** Set of joint indices (0..dof-1) in the ancestor chain (or closed-loop driven chain) of this geom */
  ancestorJoints: Set<number>;
  /** Local surface samples for CAD point-to-plane registration (empty for non-official-robot geoms) */
  localSurfaceSamples: PointNormalSample[];
  /** Official multi-material CAD meshes attached in body local frame (when robot uses CAD bundle) */
  cadMeshes?: DecodedPandaMesh[];
  /** Local AABB [minX, minY, minZ, maxX, maxY, maxZ] of official CAD mesh for exact safety checks */
  localBounds?: [number, number, number, number, number, number];
}

export interface JointMeta {
  name: string;
  jointId: number;
  qposAdr: number;
  bodyId: number;
}

let mujocoModulePromise: Promise<MainModule> | null = null;

function getMujocoModule(): Promise<MainModule> {
  if (!mujocoModulePromise) {
    mujocoModulePromise = loadMujoco({
      locateFile: (file: string) => (file.endsWith('.wasm') ? mujocoWasmUrl : file),
    }).catch((err) => {
      mujocoModulePromise = null;
      throw err;
    });
  }
  return mujocoModulePromise;
}

/** Generate uniform surface point+normal samples in a MuJoCo geom's local frame (Z-aligned for cylinder/capsule) */
function sampleGeomSurfaceLocal(
  geomType: number,
  size: [number, number, number],
  spacing = 0.015
): PointNormalSample[] {
  const samples: PointNormalSample[] = [];

  if (geomType === 6) {
    // Box: half-extents (hx, hy, hz)
    const [hx, hy, hz] = size;
    const sampleFace = (
      axisFixed: 0 | 1 | 2,
      sign: 1 | -1,
      uAxis: 0 | 1 | 2,
      vAxis: 0 | 1 | 2
    ) => {
      const hu = size[uAxis];
      const hv = size[vAxis];
      const nu = Math.max(2, Math.min(14, Math.ceil((2 * hu) / spacing)));
      const nv = Math.max(2, Math.min(14, Math.ceil((2 * hv) / spacing)));
      const n: Vec3 = [0, 0, 0];
      n[axisFixed] = sign;
      for (let iu = 0; iu < nu; iu++) {
        const u = -hu + ((iu + 0.5) / nu) * (2 * hu);
        for (let iv = 0; iv < nv; iv++) {
          const v = -hv + ((iv + 0.5) / nv) * (2 * hv);
          const p: Vec3 = [0, 0, 0];
          p[axisFixed] = sign * size[axisFixed];
          p[uAxis] = u;
          p[vAxis] = v;
          samples.push({ p, n });
        }
      }
    };
    if (hx > 1e-4 && hy > 1e-4 && hz > 1e-4) {
      sampleFace(0, 1, 1, 2);
      sampleFace(0, -1, 1, 2);
      sampleFace(1, 1, 0, 2);
      sampleFace(1, -1, 0, 2);
      sampleFace(2, 1, 0, 1);
      sampleFace(2, -1, 0, 1);
    }
  } else if (geomType === 5) {
    // Cylinder: size[0] = radius, size[1] = halfHeight, axis along local Z
    const r = size[0];
    const hh = size[1];
    const nTheta = Math.max(12, Math.min(28, Math.ceil((2 * Math.PI * r) / spacing)));
    const nZ = Math.max(3, Math.min(14, Math.ceil((2 * hh) / spacing)));
    for (let iz = 0; iz < nZ; iz++) {
      const z = -hh + ((iz + 0.5) / nZ) * (2 * hh);
      for (let it = 0; it < nTheta; it++) {
        const th = (it / nTheta) * 2 * Math.PI;
        const ct = Math.cos(th);
        const st = Math.sin(th);
        samples.push({ p: [r * ct, r * st, z], n: [ct, st, 0] });
      }
    }
    const nRad = Math.max(2, Math.min(6, Math.ceil(r / spacing)));
    for (const sign of [-1, 1] as const) {
      const z = sign * hh;
      samples.push({ p: [0, 0, z], n: [0, 0, sign] });
      for (let ir = 1; ir <= nRad; ir++) {
        const rr = (ir / (nRad + 0.3)) * r;
        const nt = Math.max(6, Math.ceil((2 * Math.PI * rr) / spacing));
        for (let it = 0; it < nt; it++) {
          const th = (it / nt) * 2 * Math.PI;
          samples.push({ p: [rr * Math.cos(th), rr * Math.sin(th), z], n: [0, 0, sign] });
        }
      }
    }
  } else if (geomType === 3) {
    // Capsule: size[0] = radius, size[1] = halfLength along local Z
    const r = size[0];
    const hl = size[1];
    const nTheta = Math.max(12, Math.min(24, Math.ceil((2 * Math.PI * r) / spacing)));
    const nZ = Math.max(3, Math.min(14, Math.ceil((2 * hl) / spacing)));
    for (let iz = 0; iz < nZ; iz++) {
      const z = -hl + ((iz + 0.5) / nZ) * (2 * hl);
      for (let it = 0; it < nTheta; it++) {
        const th = (it / nTheta) * 2 * Math.PI;
        const ct = Math.cos(th);
        const st = Math.sin(th);
        samples.push({ p: [r * ct, r * st, z], n: [ct, st, 0] });
      }
    }
    const nPhi = Math.max(3, Math.min(6, Math.ceil((0.5 * Math.PI * r) / spacing)));
    for (const sign of [-1, 1] as const) {
      samples.push({ p: [0, 0, sign * (hl + r)], n: [0, 0, sign] });
      for (let ip = 1; ip <= nPhi; ip++) {
        const phi = ((ip - 0.3) / nPhi) * (0.5 * Math.PI);
        const sp = Math.sin(phi);
        const cp = Math.cos(phi);
        const nt = Math.max(6, Math.ceil((2 * Math.PI * r * sp) / spacing));
        for (let it = 0; it < nt; it++) {
          const th = (it / nt) * 2 * Math.PI;
          const nx = sp * Math.cos(th);
          const ny = sp * Math.sin(th);
          const nz = sign * cp;
          samples.push({
            p: [r * nx, r * ny, sign * hl + r * nz],
            n: [nx, ny, nz],
          });
        }
      }
    }
  } else if (geomType === 2) {
    // Sphere
    const r = size[0];
    const nLat = 8;
    const nLon = 16;
    for (let il = 0; il < nLat; il++) {
      const phi = ((il + 0.5) / nLat) * Math.PI;
      const sp = Math.sin(phi);
      const cp = Math.cos(phi);
      for (let io = 0; io < nLon; io++) {
        const th = (io / nLon) * 2 * Math.PI;
        const nx = sp * Math.cos(th);
        const ny = sp * Math.sin(th);
        const nz = cp;
        samples.push({ p: [r * nx, r * ny, r * nz], n: [nx, ny, nz] });
      }
    }
  }

  return samples;
}

export class MujocoPandaEngine {
  readonly mujoco: MainModule;
  readonly model: MjModel;
  readonly data: MjData;
  readonly spec: RobotSpec;
  readonly robotJoints: JointMeta[];
  readonly pandaJoints: JointMeta[]; // Backwards-compatible alias
  readonly fixedJointTargets: Array<{ qposAdr: number; value: number }>;
  readonly distractorJoint: JointMeta;
  readonly geoms: GeomMeta[];
  readonly versionLabel: string;

  /** Delta parallel closed-loop freejoint qpos addresses ([0a, 0b, 1a, 1b, 2a, 2b]) */
  private readonly deltaPlatformFreeAdr: number | null;
  private readonly deltaRodFreeAdrs: number[];

  private constructor(
    mujoco: MainModule,
    model: MjModel,
    data: MjData,
    spec: RobotSpec,
    cadAssets: PandaCadBundle | null
  ) {
    this.mujoco = mujoco;
    this.model = model;
    this.data = data;
    this.spec = spec;

    if (cadAssets) {
      const triK = (cadAssets.manifest.totalTriangles / 1000).toFixed(1);
      this.versionLabel = `MuJoCo WASM 4.0.1 · ${spec.shortName} 官方 CAD (${triK}k 面)`;
    } else {
      this.versionLabel = `MuJoCo WASM 4.0.1 · ${spec.shortName} (${spec.badgeText})`;
    }

    const njnt = model.njnt;
    const jntQposAdr = model.jnt_qposadr as Int32Array;
    const jntBodyId = model.jnt_bodyid as Int32Array;
    const bodyParentId = model.body_parentid as Int32Array;

    const jointByName = new Map<string, JointMeta>();
    for (let j = 0; j < njnt; j++) {
      const name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT.value, j);
      jointByName.set(name, {
        name,
        jointId: j,
        qposAdr: jntQposAdr[j],
        bodyId: jntBodyId[j],
      });
    }

    this.robotJoints = spec.jointNames.map((jName) => {
      const meta = jointByName.get(jName);
      if (!meta) {
        throw new Error(
          `Expected joint "${jName}" for robot "${spec.id}" not found in compiled MJCF model.`
        );
      }
      return meta;
    });
    this.pandaJoints = this.robotJoints;

    this.fixedJointTargets = [];
    if (spec.fixedJointValues) {
      for (const fj of spec.fixedJointValues) {
        const meta = jointByName.get(fj.name);
        if (meta) {
          this.fixedJointTargets.push({ qposAdr: meta.qposAdr, value: fj.value });
        }
      }
    }

    const dJoint = jointByName.get('distractor_slide');
    if (!dJoint) {
      throw new Error('Expected slide joint "distractor_slide" not found in compiled MJCF model.');
    }
    this.distractorJoint = dJoint;

    // Bind Delta closed-loop freejoints when loading the parallel Delta robot
    this.deltaRodFreeAdrs = [];
    if (spec.mechanism === 'parallel_delta') {
      const platFree = jointByName.get('delta_platform_free');
      if (!platFree) {
        throw new Error('Expected Delta freejoint "delta_platform_free" not found.');
      }
      this.deltaPlatformFreeAdr = platFree.qposAdr;
      for (let i = 0; i < 3; i++) {
        for (const side of ['a', 'b']) {
          const rName = `delta_rod_${i}${side}`;
          const rMeta = jointByName.get(`${rName}_free`);
          if (!rMeta) {
            throw new Error(`Expected Delta rod freejoint "${rName}_free" not found.`);
          }
          this.deltaRodFreeAdrs.push(rMeta.qposAdr);
        }
      }
    } else {
      this.deltaPlatformFreeAdr = null;
    }

    // Map bodyId -> active robot joint index (0..dof-1) that directly drives that body
    const bodyToRobotJoint = new Map<number, number>();
    this.robotJoints.forEach((jm, idx) => {
      bodyToRobotJoint.set(jm.bodyId, idx);
    });

    const ngeom = model.ngeom;
    const geomBodyId = model.geom_bodyid as Int32Array;
    const geomType = model.geom_type as Int32Array;
    const geomSize = model.geom_size as Float64Array;
    const geomRgba = model.geom_rgba as Float32Array;

    this.geoms = [];
    for (let g = 0; g < ngeom; g++) {
      const name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM.value, g) || `geom_${g}`;
      const bodyId = geomBodyId[g];
      const bodyName =
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY.value, bodyId) || `body_${bodyId}`;
      const type = geomType[g];
      const size: [number, number, number] = [
        geomSize[g * 3 + 0],
        geomSize[g * 3 + 1],
        geomSize[g * 3 + 2],
      ];
      const rgba: [number, number, number, number] = [
        geomRgba[g * 4 + 0],
        geomRgba[g * 4 + 1],
        geomRgba[g * 4 + 2],
        geomRgba[g * 4 + 3],
      ];

      let category: GeomCategory = 'environment';
      if (name.startsWith('custom_tool_')) {
        category = 'custom_tool';
      } else if (bodyName === 'distractor_body' || name.startsWith('distractor_')) {
        category = 'distractor';
      } else if (spec.robotBodyNames.has(bodyName)) {
        category = 'robot';
      }

      const ancestorJoints = new Set<number>();
      let minDrivingJoint = -1;
      if (category === 'robot' || category === 'custom_tool') {
        if (spec.mechanism === 'parallel_delta') {
          // Closed-loop Delta kinematics:
          //  - delta_platform, delta_wrist, and all 6 rods are coupled to motors 0, 1, 2
          //  - delta_wrist (and custom_tool) are additionally driven by rotary wrist motor 3
          //  - delta_arm_k is driven by motor k
          if (bodyName === 'delta_platform' || bodyName === 'delta_wrist' || bodyName.startsWith('delta_rod_')) {
            ancestorJoints.add(0);
            ancestorJoints.add(1);
            ancestorJoints.add(2);
            minDrivingJoint = 0;
            if (bodyName === 'delta_wrist') {
              ancestorJoints.add(3);
            }
          } else if (bodyName.startsWith('delta_arm_')) {
            const armIdx = Number(bodyName.slice('delta_arm_'.length));
            if (Number.isInteger(armIdx) && armIdx >= 0 && armIdx < 3) {
              ancestorJoints.add(armIdx);
              minDrivingJoint = armIdx;
            }
          }
        } else {
          let curBody = bodyId;
          while (curBody > 0) {
            const jIdx = bodyToRobotJoint.get(curBody);
            if (jIdx !== undefined) {
              ancestorJoints.add(jIdx);
              if (minDrivingJoint === -1 || jIdx < minDrivingJoint) {
                minDrivingJoint = jIdx;
              }
            }
            const parent = bodyParentId[curBody];
            if (parent === curBody) break;
            curBody = parent;
          }
        }
      }

      // Strictly exclude custom_tool_*, distractor, and environment from Stage 2 CAD surface samples!
      const cadMeshes =
        category === 'robot' && cadAssets ? cadAssets.meshesByBody.get(bodyName) : undefined;
      const localBounds =
        category === 'robot' && cadAssets ? cadAssets.bodyBounds[bodyName] : undefined;
      let localSurfaceSamples: PointNormalSample[] = [];
      if (category === 'robot') {
        if (cadAssets) {
          localSurfaceSamples = cadAssets.bodySurfaceSamples.get(bodyName) ?? [];
        } else {
          localSurfaceSamples = sampleGeomSurfaceLocal(type, size, 0.016);
        }
      }

      this.geoms.push({
        id: g,
        name,
        bodyId,
        bodyName,
        type,
        size,
        rgba,
        category,
        minDrivingJoint,
        ancestorJoints,
        localSurfaceSamples,
        cadMeshes,
        localBounds,
      });
    }

    // Initialize at home posture
    this.setState(spec.homeQ, 0);
  }

  static async create(robotId: RobotId = 'panda'): Promise<MujocoPandaEngine> {
    const spec = getRobotSpec(robotId);
    const [mujoco, cadAssets] = await Promise.all([
      getMujocoModule(),
      spec.cadAssetId ? loadRobotCadAssets(spec.cadAssetId) : Promise.resolve(null),
    ]);

    // Write active robot MJCF into Emscripten MEMFS
    try {
      mujoco.FS.mkdir('/working');
    } catch {
      // Directory already exists
    }
    const xmlPath = `/working/${spec.id}_calib.xml`;
    mujoco.FS.writeFile(xmlPath, spec.mjcfXml);
    const model = mujoco.MjModel.from_xml_path(xmlPath);
    const data = new mujoco.MjData(model);

    return new MujocoPandaEngine(mujoco, model, data, spec, cadAssets);
  }

  /** Set robot joint angles + distractor slide position and run real mj_forward */
  setState(q: readonly number[], distractorPos = 0): void {
    const qpos = this.data.qpos as Float64Array;
    const dof = this.robotJoints.length;
    for (let i = 0; i < dof; i++) {
      qpos[this.robotJoints[i].qposAdr] = q[i] ?? this.spec.homeQ[i] ?? 0;
    }
    for (let i = 0; i < this.fixedJointTargets.length; i++) {
      const fj = this.fixedJointTargets[i];
      qpos[fj.qposAdr] = fj.value;
    }

    // For 4-DoF Delta parallel robot, solve exact closed-loop 3-sphere trilateration FK
    // and synchronize moving platform + all 6 parallelogram rods into MuJoCo freejoints
    if (this.spec.mechanism === 'parallel_delta' && this.spec.deltaParams && this.deltaPlatformFreeAdr !== null) {
      const fk = solveDeltaFK(q[0] ?? 0.22, q[1] ?? 0.22, q[2] ?? 0.22, this.spec.deltaParams);
      if (fk) {
        const pAdr = this.deltaPlatformFreeAdr;
        qpos[pAdr + 0] = fk.platformPos[0];
        qpos[pAdr + 1] = fk.platformPos[1];
        qpos[pAdr + 2] = fk.platformPos[2];
        qpos[pAdr + 3] = 1.0; // identity quaternion (pure 3D translation)
        qpos[pAdr + 4] = 0.0;
        qpos[pAdr + 5] = 0.0;
        qpos[pAdr + 6] = 0.0;

        for (let r = 0; r < fk.rods.length; r++) {
          const rod = fk.rods[r];
          const rAdr = this.deltaRodFreeAdrs[r];
          if (rAdr !== undefined) {
            qpos[rAdr + 0] = rod.pos[0];
            qpos[rAdr + 1] = rod.pos[1];
            qpos[rAdr + 2] = rod.pos[2];
            qpos[rAdr + 3] = rod.quat[0];
            qpos[rAdr + 4] = rod.quat[1];
            qpos[rAdr + 5] = rod.quat[2];
            qpos[rAdr + 6] = rod.quat[3];
          }
        }
      }
    }

    qpos[this.distractorJoint.qposAdr] = distractorPos;
    this.mujoco.mj_forward(this.model, this.data);
  }

  /** Live Float64Array views into MuJoCo WASM heap (updated after mj_forward) */
  getGeomXpos(): Float64Array {
    return this.data.geom_xpos as Float64Array;
  }

  getGeomXmat(): Float64Array {
    return this.data.geom_xmat as Float64Array;
  }

  /**
   * Read theoretical unit screw axis xi_k^b = [omega_k^b; -omega_k^b x r_k^b]
   * directly from MuJoCo data.xaxis and data.xanchor at the given joint configuration.
   */
  getJointScrewAxis(jointIndex: number, q?: readonly number[]): {
    xiBase: Vec6;
    omega: Vec3;
    anchor: Vec3;
  } {
    if (q) {
      this.setState(q, 0);
    }
    const jId = this.robotJoints[jointIndex].jointId;
    const xaxis = this.data.xaxis as Float64Array;
    const xanchor = this.data.xanchor as Float64Array;

    const omega = normalize3([
      xaxis[jId * 3 + 0],
      xaxis[jId * 3 + 1],
      xaxis[jId * 3 + 2],
    ]);
    const anchor: Vec3 = [
      xanchor[jId * 3 + 0],
      xanchor[jId * 3 + 1],
      xanchor[jId * 3 + 2],
    ];
    // v = -omega x r
    const wCrossR = cross3(omega, anchor);
    const v: Vec3 = [-wCrossR[0], -wCrossR[1], -wCrossR[2]];
    return {
      xiBase: [omega[0], omega[1], omega[2], v[0], v[1], v[2]],
      omega,
      anchor,
    };
  }

  /**
   * Compute the base-frame spatial twist per unit joint velocity (xi_k^b) for the dominant
   * moving end-effector subchain:
   *  - For all serial robots and Delta's rotary wrist (jointIndex === 3): exact MuJoCo revolute screw axis.
   *  - For Delta's 3 closed-loop parallel motors (jointIndex 0..2): pure-translation platform Jacobian
   *    [0, 0, 0, dPx/dq_k, dPy/dq_k, dPz/dq_k] computed via exact analytical Delta FK central differences.
   */
  getEffectiveActionTwistBase(jointIndex: number, q: readonly number[]): Vec6 {
    if (
      this.spec.mechanism === 'parallel_delta' &&
      this.spec.deltaParams &&
      jointIndex >= 0 &&
      jointIndex < 3
    ) {
      const eps = 1e-4;
      const qPlus = [...q];
      const qMinus = [...q];
      qPlus[jointIndex] += eps;
      qMinus[jointIndex] -= eps;
      const fkPlus = solveDeltaFK(qPlus[0], qPlus[1], qPlus[2], this.spec.deltaParams);
      const fkMinus = solveDeltaFK(qMinus[0], qMinus[1], qMinus[2], this.spec.deltaParams);
      if (fkPlus && fkMinus) {
        const inv2Eps = 1 / (2 * eps);
        return [
          0,
          0,
          0,
          (fkPlus.platformPos[0] - fkMinus.platformPos[0]) * inv2Eps,
          (fkPlus.platformPos[1] - fkMinus.platformPos[1]) * inv2Eps,
          (fkPlus.platformPos[2] - fkMinus.platformPos[2]) * inv2Eps,
        ];
      }
    }
    return this.getJointScrewAxis(jointIndex, q).xiBase;
  }

  /** Exact moving-platform center in base frame {B} for Delta parallel robot */
  getDeltaPlatformPosBase(q: readonly number[]): Vec3 | null {
    if (this.spec.mechanism !== 'parallel_delta' || !this.spec.deltaParams) return null;
    const fk = solveDeltaFK(q[0] ?? 0.22, q[1] ?? 0.22, q[2] ?? 0.22, this.spec.deltaParams);
    return fk ? fk.platformPos : null;
  }

  /**
   * Verify via real MuJoCo FK that all moving robot links stay safely above table z > minZ
   * and within workspace bounds.
   */
  isPoseSafe(q: readonly number[], minZ = 0.06): boolean {
    if (this.spec.mechanism === 'parallel_delta' && this.spec.deltaParams) {
      const fk = solveDeltaFK(q[0] ?? 0.22, q[1] ?? 0.22, q[2] ?? 0.22, this.spec.deltaParams);
      if (!fk) return false;
      if (fk.platformPos[2] < 0.24 || fk.platformPos[2] > 0.72) return false;
    }

    this.setState(q, 0);
    const xpos = this.getGeomXpos();
    const xmat = this.getGeomXmat();
    for (let i = 0; i < this.geoms.length; i++) {
      const g = this.geoms[i];
      if (g.category !== 'robot' && g.category !== 'custom_tool') continue;
      if (this.spec.staticBaseBodies.has(g.bodyName)) continue;

      const px = xpos[g.id * 3 + 0];
      const py = xpos[g.id * 3 + 1];
      const pz = xpos[g.id * 3 + 2];

      if (g.localBounds) {
        const [x0, y0, z0, x1, y1, z1] = g.localBounds;
        const m = g.id * 9;
        const r00 = xmat[m + 0], r01 = xmat[m + 1], r02 = xmat[m + 2];
        const r10 = xmat[m + 3], r11 = xmat[m + 4], r12 = xmat[m + 5];
        const r20 = xmat[m + 6], r21 = xmat[m + 7], r22 = xmat[m + 8];
        const xs = [x0, x1];
        const ys = [y0, y1];
        const zs = [z0, z1];
        for (let ix = 0; ix < 2; ix++) {
          for (let iy = 0; iy < 2; iy++) {
            for (let iz = 0; iz < 2; iz++) {
              const lx = xs[ix], ly = ys[iy], lz = zs[iz];
              const wx = r00 * lx + r01 * ly + r02 * lz + px;
              const wy = r10 * lx + r11 * ly + r12 * lz + py;
              const wz = r20 * lx + r21 * ly + r22 * lz + pz;
              if (wz < minZ || wx < -0.78 || Math.abs(wy) > 0.92) {
                return false;
              }
            }
          }
        }
      } else {
        const clearance = g.type === 6 ? Math.min(g.size[0], g.size[1], g.size[2]) : g.size[0];
        if (pz - clearance < minZ) {
          return false;
        }
        if (px < -0.78 || Math.abs(py) > 0.92) {
          return false;
        }
      }
    }
    return true;
  }

  /**
   * Transform official robot CAD surface samples into base frame {B} at configuration q.
   * Strictly excludes custom_tool_*, distractor, and environment geoms.
   * Optionally filters to geoms strictly downstream/driven by `downstreamOfJoint`.
   */
  getRobotSurfacePointsBase(
    q: readonly number[],
    downstreamOfJoint?: number,
    stride = 1
  ): PointNormalSample[] {
    this.setState(q, 0);
    const xpos = this.getGeomXpos();
    const xmat = this.getGeomXmat();
    const out: PointNormalSample[] = [];

    for (let i = 0; i < this.geoms.length; i++) {
      const g = this.geoms[i];
      if (g.category !== 'robot') continue;
      if (downstreamOfJoint !== undefined && !g.ancestorJoints.has(downstreamOfJoint)) {
        continue;
      }

      const px = xpos[g.id * 3 + 0];
      const py = xpos[g.id * 3 + 1];
      const pz = xpos[g.id * 3 + 2];
      const m = g.id * 9;
      const R: Mat3 = [
        xmat[m + 0], xmat[m + 1], xmat[m + 2],
        xmat[m + 3], xmat[m + 4], xmat[m + 5],
        xmat[m + 6], xmat[m + 7], xmat[m + 8],
      ];

      const local = g.localSurfaceSamples;
      for (let s = 0; s < local.length; s += stride) {
        const lp = local[s].p;
        const ln = local[s].n;
        const p: Vec3 = [
          R[0] * lp[0] + R[1] * lp[1] + R[2] * lp[2] + px,
          R[3] * lp[0] + R[4] * lp[1] + R[5] * lp[2] + py,
          R[6] * lp[0] + R[7] * lp[1] + R[8] * lp[2] + pz,
        ];
        const n: Vec3 = [
          R[0] * ln[0] + R[1] * ln[1] + R[2] * ln[2],
          R[3] * ln[0] + R[4] * ln[1] + R[5] * ln[2],
          R[6] * ln[0] + R[7] * ln[1] + R[8] * ln[2],
        ];
        out.push({ p, n });
      }
    }
    return out;
  }

  /**
   * Transform official robot CAD surface samples into base frame {B} at configuration q,
   * tagged with each geom's bodyId and ancestorJoints for V3 multi-body Stage 4 EM-LM.
   */
  getRobotSurfacePointsBaseWithLink(
    q: readonly number[],
    stride = 1
  ): Array<PointNormalSample & { bodyId: number; ancestorJoints: ReadonlySet<number> }> {
    this.setState(q, 0);
    const xpos = this.getGeomXpos();
    const xmat = this.getGeomXmat();
    const out: Array<PointNormalSample & { bodyId: number; ancestorJoints: ReadonlySet<number> }> = [];

    for (let i = 0; i < this.geoms.length; i++) {
      const g = this.geoms[i];
      if (g.category !== 'robot') continue;

      const px = xpos[g.id * 3 + 0];
      const py = xpos[g.id * 3 + 1];
      const pz = xpos[g.id * 3 + 2];
      const m = g.id * 9;
      const R: Mat3 = [
        xmat[m + 0], xmat[m + 1], xmat[m + 2],
        xmat[m + 3], xmat[m + 4], xmat[m + 5],
        xmat[m + 6], xmat[m + 7], xmat[m + 8],
      ];

      const local = g.localSurfaceSamples;
      for (let s = 0; s < local.length; s += stride) {
        const lp = local[s].p;
        const ln = local[s].n;
        const p: Vec3 = [
          R[0] * lp[0] + R[1] * lp[1] + R[2] * lp[2] + px,
          R[3] * lp[0] + R[4] * lp[1] + R[5] * lp[2] + py,
          R[6] * lp[0] + R[7] * lp[1] + R[8] * lp[2] + pz,
        ];
        const n: Vec3 = [
          R[0] * ln[0] + R[1] * ln[1] + R[2] * ln[2],
          R[3] * ln[0] + R[4] * ln[1] + R[5] * ln[2],
          R[6] * ln[0] + R[7] * ln[1] + R[8] * ln[2],
        ];
        out.push({ p, n, bodyId: g.bodyId, ancestorJoints: g.ancestorJoints });
      }
    }
    return out;
  }

  /**
   * Compute 4 realistic base-frame 3D anchor points along the robot arm (base -> shoulder -> elbow -> wrist/platform)
   * directly from live MuJoCo geom_xpos for attaching a coupled non-rigid flexible conduit/cable.
   */
  getCableControlPointsBase(): Vec3[] {
    const xpos = this.getGeomXpos();
    const robotGeoms = this.geoms.filter((g) => g.category === 'robot');
    if (robotGeoms.length === 0) {
      return [
        [0, -0.06, 0.08],
        [0.04, -0.08, 0.35],
        [0.18, -0.08, 0.55],
        [0.32, -0.04, 0.48],
      ];
    }

    if (this.spec.mechanism === 'parallel_delta') {
      const topG = robotGeoms.find((g) => g.bodyName === 'delta_base') ?? robotGeoms[0];
      const armG = robotGeoms.find((g) => g.bodyName === 'delta_arm_0') ?? robotGeoms[1] ?? topG;
      const platG = robotGeoms.find((g) => g.bodyName === 'delta_platform') ?? robotGeoms[robotGeoms.length - 1];
      const p0: Vec3 = [xpos[topG.id * 3] + 0.14, xpos[topG.id * 3 + 1] - 0.10, xpos[topG.id * 3 + 2] - 0.03];
      const p1: Vec3 = [xpos[armG.id * 3] + 0.06, xpos[armG.id * 3 + 1] - 0.06, xpos[armG.id * 3 + 2] - 0.05];
      const p3: Vec3 = [xpos[platG.id * 3] + 0.04, xpos[platG.id * 3 + 1] - 0.04, xpos[platG.id * 3 + 2] + 0.03];
      const p2: Vec3 = [
        0.45 * p1[0] + 0.55 * p3[0] + 0.08,
        0.45 * p1[1] + 0.55 * p3[1] - 0.07,
        0.45 * p1[2] + 0.55 * p3[2] - 0.06,
      ];
      return [p0, p1, p2, p3];
    }

    // Serial arms: pick representative geoms distributed by ancestor depth (base -> mid -> distal)
    const sorted = [...robotGeoms].sort((a, b) => a.ancestorJoints.size - b.ancestorJoints.size);
    const pickAt = (frac: number): Vec3 => {
      const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(frac * (sorted.length - 1))));
      const g = sorted[idx];
      return [xpos[g.id * 3 + 0], xpos[g.id * 3 + 1], xpos[g.id * 3 + 2]];
    };
    const pBase = pickAt(0.0);
    const pShoulder = pickAt(0.32);
    const pElbow = pickAt(0.65);
    const pWrist = pickAt(0.92);
    return [
      [pBase[0] - 0.04, pBase[1] - 0.07, Math.max(0.06, pBase[2] + 0.04)],
      [pShoulder[0] + 0.02, pShoulder[1] - 0.09, pShoulder[2] + 0.02],
      [pElbow[0] + 0.03, pElbow[1] - 0.08, pElbow[2] - 0.02],
      [pWrist[0], pWrist[1] - 0.04, pWrist[2] + 0.02],
    ];
  }

  dispose(): void {
    this.data.delete();
    this.model.delete();
  }
}
