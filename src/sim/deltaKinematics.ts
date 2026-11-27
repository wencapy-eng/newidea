import { cross3, normalize3, type Vec3 } from '../math/se3';

export interface DeltaGeometryParams {
  /** Top stationary base attachment radius (m) */
  baseRadius: number;
  /** Moving end-effector platform attachment radius (m) */
  platformRadius: number;
  /** Active upper arm length (m) */
  upperLen: number;
  /** Passive parallelogram forearm rod length (m) */
  forearmLen: number;
  /** Height of the stationary top base plane in world frame {B} (m) */
  baseZ: number;
  /** Half lateral separation of each twin parallelogram rod pair (m) */
  rodPairHalfOffset: number;
}

export const DEFAULT_DELTA_PARAMS: DeltaGeometryParams = {
  baseRadius: 0.19,
  platformRadius: 0.068,
  upperLen: 0.25,
  forearmLen: 0.50,
  baseZ: 0.86,
  rodPairHalfOffset: 0.024,
};

export interface DeltaRodPose {
  /** Midpoint position in base frame {B} */
  pos: Vec3;
  /** Unit quaternion [w, x, y, z] aligning local +Z along the rod axis */
  quat: [number, number, number, number];
  /** Exact rod length (m) */
  length: number;
}

export interface DeltaFKResult {
  platformPos: Vec3;
  elbows: [Vec3, Vec3, Vec3];
  /** 6 parallel forearm rods: [leg0_a, leg0_b, leg1_a, leg1_b, leg2_a, leg2_b] */
  rods: [DeltaRodPose, DeltaRodPose, DeltaRodPose, DeltaRodPose, DeltaRodPose, DeltaRodPose];
}

const LEG_ANGLES_RAD = [0, (2 * Math.PI) / 3, (4 * Math.PI) / 3] as const;

/** Build unit quaternion [w, x, y, z] rotating local +Z ([0,0,1]) to unit direction d */
function quatFromZAxis(d: Vec3): [number, number, number, number] {
  const dz = d[2];
  if (dz > 0.999999) return [1, 0, 0, 0];
  if (dz < -0.999999) return [0, 1, 0, 0];
  // cross([0,0,1], d) = [-d[1], d[0], 0]
  const qx = -d[1];
  const qy = d[0];
  const qz = 0;
  const qw = 1 + dz;
  const n = Math.hypot(qw, qx, qy, qz);
  return [qw / n, qx / n, qy / n, qz / n];
}

/**
 * Exact closed-form analytical Forward Kinematics (3-sphere trilateration)
 * for a 3-leg Clavel Delta parallel robot.
 *
 * Angle convention:
 *   q_i = 0 => upper arm points horizontally outward along radial vector u_i
 *   q_i > 0 => upper arm pitches downward toward the workspace
 */
export function solveDeltaFK(
  q0: number,
  q1: number,
  q2: number,
  params: DeltaGeometryParams = DEFAULT_DELTA_PARAMS
): DeltaFKResult | null {
  const qs = [q0, q1, q2];
  const { baseRadius: R, platformRadius: r, upperLen: L, forearmLen: l, baseZ } = params;

  const elbows: Vec3[] = [];
  const centers: Vec3[] = [];

  for (let i = 0; i < 3; i++) {
    const phi = LEG_ANGLES_RAD[i];
    const cp = Math.cos(phi);
    const sp = Math.sin(phi);
    const cq = Math.cos(qs[i]);
    const sq = Math.sin(qs[i]);

    const rElbow = R + L * cq;
    const ex = rElbow * cp;
    const ey = rElbow * sp;
    const ez = baseZ - L * sq;
    elbows.push([ex, ey, ez]);

    // Shifted sphere center C_i = E_i - r * [cos phi, sin phi, 0]
    centers.push([ex - r * cp, ey - r * sp, ez]);
  }

  // Exact orthonormal trilateration of 3 spheres centered at C0, C1, C2 of radius l
  const c0 = centers[0];
  const c1 = centers[1];
  const c2 = centers[2];

  const c10: Vec3 = [c1[0] - c0[0], c1[1] - c0[1], c1[2] - c0[2]];
  const d = Math.hypot(c10[0], c10[1], c10[2]);
  if (d < 1e-8) return null;
  const ex: Vec3 = [c10[0] / d, c10[1] / d, c10[2] / d];

  const c20: Vec3 = [c2[0] - c0[0], c2[1] - c0[1], c2[2] - c0[2]];
  const iVal = ex[0] * c20[0] + ex[1] * c20[1] + ex[2] * c20[2];

  const eyRaw: Vec3 = [
    c20[0] - iVal * ex[0],
    c20[1] - iVal * ex[1],
    c20[2] - iVal * ex[2],
  ];
  const jVal = Math.hypot(eyRaw[0], eyRaw[1], eyRaw[2]);
  if (jVal < 1e-8) return null;
  const ey: Vec3 = [eyRaw[0] / jVal, eyRaw[1] / jVal, eyRaw[2] / jVal];

  let ez = cross3(ex, ey);
  // Ensure ez points downward (-Z) so positive zVal selects the hanging lower platform branch
  if (ez[2] > 0) {
    ez = [-ez[0], -ez[1], -ez[2]];
  }

  // Equal radii l => x = d/2, y = (i^2 + j^2 - 2*i*x) / (2*j)
  const xVal = 0.5 * d;
  const yVal = (iVal * iVal + jVal * jVal - 2 * iVal * xVal) / (2 * jVal);
  const zSq = l * l - xVal * xVal - yVal * yVal;
  if (zSq < 1e-8) return null;
  const zVal = Math.sqrt(zSq);

  const platformPos: Vec3 = [
    c0[0] + xVal * ex[0] + yVal * ey[0] + zVal * ez[0],
    c0[1] + xVal * ex[1] + yVal * ey[1] + zVal * ez[1],
    c0[2] + xVal * ex[2] + yVal * ey[2] + zVal * ez[2],
  ];

  // Compute exact pose for all 6 twin parallelogram rods
  const rods: DeltaRodPose[] = [];
  const off = params.rodPairHalfOffset;

  for (let i = 0; i < 3; i++) {
    const phi = LEG_ANGLES_RAD[i];
    const cp = Math.cos(phi);
    const sp = Math.sin(phi);
    // Tangential unit vector perpendicular to radial direction in XY
    const tx = -sp;
    const ty = cp;

    const e = elbows[i];
    const pAttach: Vec3 = [
      platformPos[0] + r * cp,
      platformPos[1] + r * sp,
      platformPos[2],
    ];

    for (const sgn of [-1, 1] as const) {
      const a: Vec3 = [e[0] + sgn * off * tx, e[1] + sgn * off * ty, e[2]];
      const b: Vec3 = [pAttach[0] + sgn * off * tx, pAttach[1] + sgn * off * ty, pAttach[2]];
      const mid: Vec3 = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), 0.5 * (a[2] + b[2])];
      const dv: Vec3 = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const len = Math.hypot(dv[0], dv[1], dv[2]);
      const dir = normalize3(dv);
      rods.push({
        pos: mid,
        quat: quatFromZAxis(dir),
        length: len,
      });
    }
  }

  return {
    platformPos,
    elbows: [elbows[0], elbows[1], elbows[2]],
    rods: [rods[0], rods[1], rods[2], rods[3], rods[4], rods[5]],
  };
}
