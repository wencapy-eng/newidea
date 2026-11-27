import type { MotionAction } from '../calib/types';
import type { Vec3 } from '../math/se3';
import { DEFAULT_DELTA_PARAMS, type DeltaGeometryParams } from './deltaKinematics';
import type { RobotCadAssetId } from './pandaCadAssets';

export type RobotId =
  | 'panda'
  | 'iiwa14'
  | 'ur5e'
  | 'aloha'
  | 'openarm'
  | 'custom_serial'
  | 'delta_parallel';

export type RobotMechanism = 'serial' | 'parallel_delta';

export interface RobotSpec {
  id: RobotId;
  name: string;
  shortName: string;
  dof: number;
  mechanism: RobotMechanism;
  badgeText: string;
  materialTag: string;
  cadAssetId?: RobotCadAssetId;
  mjcfXml: string;
  jointNames: readonly string[];
  jointLimits: ReadonlyArray<[number, number]>;
  homeQ: readonly number[];
  fixedJointValues?: ReadonlyArray<{ name: string; value: number }>;
  robotBodyNames: ReadonlySet<string>;
  staticBaseBodies: ReadonlySet<string>;
  /** Virtual RealSense D435i random-pose lookAt target and sampling radius [min, max] (m) */
  cameraTarget: Vec3;
  cameraDistRange: [number, number];
  /** Default 3D viewport OrbitControls camera position & target */
  viewCamPos: Vec3;
  viewCamTarget: Vec3;
  coldStartActions: readonly MotionAction[];
  candidateActions: readonly MotionAction[];
  deltaParams?: DeltaGeometryParams;
}

const SHARED_ENVIRONMENT_XML = `
    <!-- Workbench Tabletop (top surface at z = 0.0) -->
    <geom name="table_top" type="box" pos="0.15 0 -0.025" size="0.95 1.15 0.025" rgba="0.18 0.21 0.26 1"/>
    <geom name="table_leg_fl" type="box" pos="0.98 1.02 -0.38" size="0.035 0.035 0.33" rgba="0.11 0.13 0.17 1"/>
    <geom name="table_leg_fr" type="box" pos="0.98 -1.02 -0.38" size="0.035 0.035 0.33" rgba="0.11 0.13 0.17 1"/>
    <geom name="table_leg_bl" type="box" pos="-0.68 1.02 -0.38" size="0.035 0.035 0.33" rgba="0.11 0.13 0.17 1"/>
    <geom name="table_leg_br" type="box" pos="-0.68 -1.02 -0.38" size="0.035 0.035 0.33" rgba="0.11 0.13 0.17 1"/>

    <!-- Lab Enclosure Walls (placed safely behind robot & outside camera sampling volume) -->
    <geom name="wall_back" type="box" pos="-0.86 0 0.82" size="0.03 1.48 0.90" rgba="0.14 0.17 0.22 1"/>
    <geom name="wall_side" type="box" pos="0.15 -1.30 0.82" size="1.08 0.03 0.90" rgba="0.12 0.15 0.20 1"/>

    <!-- 3 Static Tabletop Clutter Props -->
    <geom name="clutter_1_crate" type="box" pos="-0.42 0.58 0.065" size="0.085 0.075 0.065" rgba="0.30 0.36 0.44 1"/>
    <geom name="clutter_1_lid" type="box" pos="-0.42 0.58 0.138" size="0.092 0.082 0.008" rgba="0.46 0.52 0.60 1"/>
    <geom name="clutter_2_canister" type="cylinder" pos="0.46 -0.60 0.085" size="0.055 0.085" rgba="0.24 0.40 0.46 1"/>
    <geom name="clutter_2_cap" type="cylinder" pos="0.46 -0.60 0.178" size="0.044 0.010" rgba="0.62 0.68 0.74 1"/>
    <geom name="clutter_3_fixture_base" type="box" pos="-0.44 -0.48 0.045" size="0.075 0.060 0.045" rgba="0.38 0.30 0.26 1"/>
    <geom name="clutter_3_fixture_post" type="box" pos="-0.48 -0.48 0.135" size="0.024 0.052 0.048" rgba="0.52 0.40 0.34 1"/>

    <!-- Independent Asynchronous Dynamic Distractor -->
    <body name="distractor_body" pos="0.36 0.56 0.105">
      <joint name="distractor_slide" type="slide" axis="0 1 0" range="-0.42 0.42"/>
      <geom name="distractor_block" type="box" pos="0 0 0" size="0.068 0.085 0.095" rgba="0.82 0.28 0.26 1"/>
      <geom name="distractor_fin" type="box" pos="0.055 0 0.045" size="0.022 0.065 0.045" rgba="0.92 0.46 0.34 1"/>
      <geom name="distractor_cap" type="cylinder" pos="0 0 0.112" size="0.045 0.018" rgba="0.22 0.25 0.30 1"/>
    </body>`;

function wrapMjcf(modelName: string, robotBodyXml: string): string {
  return `<mujoco model="${modelName}">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <default>
    <joint damping="1.0" frictionloss="0.1"/>
    <geom contype="0" conaffinity="0"/>
  </default>
  <worldbody>
${SHARED_ENVIRONMENT_XML}
${robotBodyXml}
  </worldbody>
</mujoco>`;
}

/* ============================================================================
 * 1. Franka Emika Panda (7-DoF, Official mujoco_menagerie CAD)
 * ========================================================================== */
const PANDA_SPEC: RobotSpec = {
  id: 'panda',
  name: 'Franka Emika Panda',
  shortName: 'Franka Panda',
  dof: 7,
  mechanism: 'serial',
  badgeText: '7-DoF 串联 · 官方 CAD',
  materialTag: '缎面白清漆 + 炭黑环',
  cadAssetId: 'panda',
  mjcfXml: wrapMjcf(
    'franka_panda_calib',
    `
    <body name="link0" pos="0 0 0">
      <geom name="link0_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
      <body name="link1" pos="0 0 0.333">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
        <geom name="link1_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
        <body name="link2" pos="0 0 0" quat="1 -1 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1" range="-1.7628 1.7628"/>
          <geom name="link2_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
          <body name="link3" pos="0 -0.316 0" quat="1 1 0 0">
            <joint name="joint3" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
            <geom name="link3_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
            <body name="link4" pos="0.0825 0 0" quat="1 1 0 0">
              <joint name="joint4" type="hinge" axis="0 0 1" range="-3.0718 -0.0698"/>
              <geom name="link4_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
              <body name="link5" pos="-0.0825 0.384 0" quat="1 -1 0 0">
                <joint name="joint5" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                <geom name="link5_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                <body name="link6" pos="0 0 0" quat="1 1 0 0">
                  <joint name="joint6" type="hinge" axis="0 0 1" range="-0.0175 3.7525"/>
                  <geom name="link6_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                  <body name="link7" pos="0.088 0 0" quat="1 1 0 0">
                    <joint name="joint7" type="hinge" axis="0 0 1" range="-2.8973 2.8973"/>
                    <geom name="link7_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                    <body name="hand" pos="0 0 0.107" quat="0.9238795 0 0 -0.3826834">
                      <geom name="hand_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                      <body name="left_finger" pos="0 0 0.0584">
                        <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
                        <geom name="left_finger_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                      </body>
                      <body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">
                        <joint name="finger_joint2" type="slide" axis="0 1 0" range="0 0.04"/>
                        <geom name="right_finger_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                      </body>
                      <geom name="custom_tool_mount" type="box" pos="-0.046 0 0.028" size="0.016 0.040 0.022" rgba="0.86 0.56 0.14 1"/>
                      <geom name="custom_tool_arm" type="cylinder" pos="-0.056 0.024 0.088" size="0.014 0.044" rgba="0.92 0.64 0.18 1"/>
                      <geom name="custom_tool_tip" type="box" pos="-0.056 0.024 0.146" size="0.028 0.016 0.020" rgba="0.78 0.44 0.12 1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>`
  ),
  jointNames: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'],
  jointLimits: [
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-2.8973, 2.8973],
    [-3.0718, -0.0698],
    [-2.8973, 2.8973],
    [-0.0175, 3.7525],
    [-2.8973, 2.8973],
  ],
  homeQ: [0.0, -0.35, 0.0, -1.95, 0.0, 1.65, 0.785],
  fixedJointValues: [
    { name: 'finger_joint1', value: 0.032 },
    { name: 'finger_joint2', value: 0.032 },
  ],
  robotBodyNames: new Set([
    'link0', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7',
    'hand', 'left_finger', 'right_finger',
  ]),
  staticBaseBodies: new Set(['link0']),
  cameraTarget: [0.08, 0.0, 0.42],
  cameraDistRange: [1.18, 1.48],
  viewCamPos: [1.48, 1.18, 1.12],
  viewCamTarget: [0.10, 0.0, 0.44],
  coldStartActions: [
    { id: 'cold_j1', label: '冷启动 #1 (Joint1 底座偏航 Z)', jointIndex: 0, qBase: [0.0, -0.32, 0.0, -1.92, 0.0, 1.62, 0.75], qStart: -0.34, qEnd: 0.34, isColdStart: true },
    { id: 'cold_j2', label: '冷启动 #2 (Joint2 肩部俯仰 Y)', jointIndex: 1, qBase: [0.0, -0.35, 0.0, -1.95, 0.0, 1.68, 0.65], qStart: -0.52, qEnd: 0.12, isColdStart: true },
    { id: 'cold_j4', label: '冷启动 #3 (Joint4 正交肘部 X)', jointIndex: 3, qBase: [1.18, -0.26, 0.32, -1.85, -0.15, 1.72, 0.45], qStart: -2.26, qEnd: -1.52, isColdStart: true },
  ],
  candidateActions: [
    { id: 'cand_j1_ext', label: 'J1 伸展扫掠', jointIndex: 0, qBase: [0.0, 0.12, 0.0, -1.55, 0.0, 1.75, 0.78], qStart: -0.38, qEnd: 0.32 },
    { id: 'cand_j1_high', label: 'J1 高位偏航', jointIndex: 0, qBase: [0.0, -0.55, 0.25, -1.65, -0.35, 1.45, 0.5], qStart: 0.36, qEnd: -0.34 },
    { id: 'cand_j2_left', label: 'J2 左侧俯仰', jointIndex: 1, qBase: [0.95, -0.25, 0.0, -1.85, 0.0, 1.75, 0.4], qStart: -0.52, qEnd: 0.14 },
    { id: 'cand_j2_right', label: 'J2 右侧俯仰', jointIndex: 1, qBase: [-0.95, -0.25, 0.0, -1.85, 0.0, 1.75, 0.4], qStart: -0.50, qEnd: 0.14 },
    { id: 'cand_j2_diag', label: 'J2 斜向俯仰', jointIndex: 1, qBase: [0.55, -0.2, -0.4, -1.7, 0.3, 1.95, 0.8], qStart: -0.48, qEnd: 0.16 },
    { id: 'cand_j3_a', label: 'J3 前倾自转', jointIndex: 2, qBase: [0.25, 0.32, 0.0, -1.85, 0.0, 2.05, 0.6], qStart: -0.46, qEnd: 0.42 },
    { id: 'cand_j3_b', label: 'J3 后仰自转', jointIndex: 2, qBase: [-0.55, -0.52, 0.0, -1.75, 0.2, 1.65, 0.7], qStart: -0.44, qEnd: 0.44 },
    { id: 'cand_j4_front', label: 'J4 正面肘屈', jointIndex: 3, qBase: [-0.15, -0.22, 0.0, -1.9, 0.0, 1.85, 0.75], qStart: -2.28, qEnd: -1.52 },
    { id: 'cand_j4_right', label: 'J4 右偏肘屈', jointIndex: 3, qBase: [-1.15, -0.25, -0.35, -1.85, 0.25, 1.78, 0.55], qStart: -2.24, qEnd: -1.50 },
    { id: 'cand_j4_tilt', label: 'J4 倾臂肘屈', jointIndex: 3, qBase: [0.45, 0.15, 0.75, -1.9, 0.0, 1.95, 0.6], qStart: -2.25, qEnd: -1.48 },
    { id: 'cand_j5_a', label: 'J5 前臂旋腕 A', jointIndex: 4, qBase: [0.15, -0.15, 0.0, -1.75, 0.0, 2.25, 0.5], qStart: -0.52, qEnd: 0.52 },
    { id: 'cand_j6_a', label: 'J6 腕部俯仰 A', jointIndex: 5, qBase: [-0.25, -0.18, 0.15, -1.72, 0.2, 1.85, 0.78], qStart: 1.38, qEnd: 2.28 },
  ],
};

/* ============================================================================
 * 2. KUKA LBR iiwa 14 (7-DoF, Official mujoco_menagerie CAD)
 * ========================================================================== */
const IIWA14_SPEC: RobotSpec = {
  id: 'iiwa14',
  name: 'KUKA LBR iiwa 14',
  shortName: 'KUKA iiwa 14',
  dof: 7,
  mechanism: 'serial',
  badgeText: '7-DoF 串联 · 官方 CAD',
  materialTag: '库卡橙亮漆 + 铸铁灰',
  cadAssetId: 'iiwa14',
  mjcfXml: wrapMjcf(
    'kuka_iiwa14_calib',
    `
    <body name="base" pos="0 0 0">
      <geom name="base_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
      <body name="link1" pos="0 0 0.1575">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-2.96706 2.96706"/>
        <geom name="link1_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
        <body name="link2" pos="0 0 0.2025" quat="0 0 1 1">
          <joint name="joint2" type="hinge" axis="0 0 1" range="-2.0944 2.0944"/>
          <geom name="link2_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
          <body name="link3" pos="0 0.2045 0" quat="0 0 1 1">
            <joint name="joint3" type="hinge" axis="0 0 1" range="-2.96706 2.96706"/>
            <geom name="link3_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
            <body name="link4" pos="0 0 0.2155" quat="1 1 0 0">
              <joint name="joint4" type="hinge" axis="0 0 1" range="-2.0944 2.0944"/>
              <geom name="link4_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
              <body name="link5" pos="0 0.1845 0" quat="0 0 1 1">
                <joint name="joint5" type="hinge" axis="0 0 1" range="-2.96706 2.96706"/>
                <geom name="link5_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                <body name="link6" pos="0 0 0.2155" quat="1 1 0 0">
                  <joint name="joint6" type="hinge" axis="0 0 1" range="-2.0944 2.0944"/>
                  <geom name="link6_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                  <body name="link7" pos="0 0.081 0" quat="0 0 1 1">
                    <joint name="joint7" type="hinge" axis="0 0 1" range="-3.05433 3.05433"/>
                    <geom name="link7_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                    <geom name="custom_tool_mount" type="box" pos="0.028 0 0.058" size="0.026 0.036 0.018" rgba="0.86 0.56 0.14 1"/>
                    <geom name="custom_tool_arm" type="cylinder" pos="0.042 0.018 0.112" size="0.014 0.042" rgba="0.92 0.64 0.18 1"/>
                    <geom name="custom_tool_tip" type="box" pos="0.042 0.018 0.164" size="0.026 0.016 0.018" rgba="0.78 0.44 0.12 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>`
  ),
  jointNames: ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7'],
  jointLimits: [
    [-2.967, 2.967],
    [-2.094, 2.094],
    [-2.967, 2.967],
    [-2.094, 2.094],
    [-2.967, 2.967],
    [-2.094, 2.094],
    [-3.054, 3.054],
  ],
  homeQ: [0.0, 0.45, 0.0, -1.45, 0.0, 1.05, 0.35],
  robotBodyNames: new Set(['base', 'link1', 'link2', 'link3', 'link4', 'link5', 'link6', 'link7']),
  staticBaseBodies: new Set(['base']),
  cameraTarget: [0.10, 0.0, 0.50],
  cameraDistRange: [1.28, 1.58],
  viewCamPos: [1.62, 1.28, 1.24],
  viewCamTarget: [0.10, 0.0, 0.50],
  coldStartActions: [
    { id: 'iiwa_cold_j1', label: '冷启动 #1 (Joint1 底座偏航 Z)', jointIndex: 0, qBase: [0.0, 0.48, 0.0, -1.48, 0.0, 1.05, 0.35], qStart: -0.36, qEnd: 0.36, isColdStart: true },
    { id: 'iiwa_cold_j2', label: '冷启动 #2 (Joint2 肩部俯仰 Y)', jointIndex: 1, qBase: [0.0, 0.42, 0.0, -1.42, 0.0, 1.05, 0.35], qStart: 0.08, qEnd: 0.72, isColdStart: true },
    { id: 'iiwa_cold_j4', label: '冷启动 #3 (Joint4 正交肘部 X)', jointIndex: 3, qBase: [1.22, 0.38, 0.25, -1.35, -0.15, 1.02, 0.35], qStart: -1.68, qEnd: -0.96, isColdStart: true },
  ],
  candidateActions: [
    { id: 'iiwa_j1_a', label: 'J1 伸展偏航', jointIndex: 0, qBase: [0.0, 0.65, 0.0, -1.25, 0.0, 0.95, 0.4], qStart: -0.38, qEnd: 0.34 },
    { id: 'iiwa_j1_b', label: 'J1 紧凑偏航', jointIndex: 0, qBase: [0.0, -0.25, 0.3, -1.65, -0.2, 1.15, 0.2], qStart: 0.36, qEnd: -0.36 },
    { id: 'iiwa_j2_l', label: 'J2 左向俯仰', jointIndex: 1, qBase: [0.95, 0.35, 0.0, -1.45, 0.0, 1.05, 0.3], qStart: 0.04, qEnd: 0.68 },
    { id: 'iiwa_j2_r', label: 'J2 右向俯仰', jointIndex: 1, qBase: [-0.95, 0.35, 0.0, -1.45, 0.0, 1.05, 0.3], qStart: 0.05, qEnd: 0.68 },
    { id: 'iiwa_j3_a', label: 'J3 上臂自转 A', jointIndex: 2, qBase: [0.25, 0.55, 0.0, -1.42, 0.0, 1.08, 0.4], qStart: -0.45, qEnd: 0.45 },
    { id: 'iiwa_j3_b', label: 'J3 上臂自转 B', jointIndex: 2, qBase: [-0.65, 0.38, 0.0, -1.52, 0.2, 1.15, 0.2], qStart: 0.44, qEnd: -0.44 },
    { id: 'iiwa_j4_front', label: 'J4 正面肘屈', jointIndex: 3, qBase: [-0.15, 0.45, 0.0, -1.35, 0.0, 1.05, 0.4], qStart: -1.72, qEnd: -0.98 },
    { id: 'iiwa_j4_right', label: 'J4 右偏肘屈', jointIndex: 3, qBase: [-1.12, 0.42, -0.25, -1.35, 0.2, 1.05, 0.3], qStart: -1.70, qEnd: -0.96 },
    { id: 'iiwa_j4_tilt', label: 'J4 斜向肘屈', jointIndex: 3, qBase: [0.55, 0.32, 0.65, -1.35, 0.0, 1.12, 0.5], qStart: -1.68, qEnd: -0.94 },
    { id: 'iiwa_j5_a', label: 'J5 前臂旋腕', jointIndex: 4, qBase: [0.15, 0.42, 0.0, -1.45, 0.0, 1.25, 0.4], qStart: -0.52, qEnd: 0.52 },
    { id: 'iiwa_j6_a', label: 'J6 腕部俯仰', jointIndex: 5, qBase: [-0.25, 0.38, 0.15, -1.42, 0.25, 0.95, 0.4], qStart: 0.45, qEnd: 1.35 },
  ],
};

/* ============================================================================
 * 3. Universal Robots UR5e (6-DoF, Official mujoco_menagerie CAD)
 * ========================================================================== */
const UR5E_SPEC: RobotSpec = {
  id: 'ur5e',
  name: 'Universal Robots UR5e',
  shortName: 'UR5e (金属强反光)',
  dof: 6,
  mechanism: 'serial',
  badgeText: '6-DoF 串联 · 官方 CAD',
  materialTag: '高反光拉丝不锈钢 + UR蓝',
  cadAssetId: 'ur5e',
  mjcfXml: wrapMjcf(
    'ur5e_calib',
    `
    <body name="base" pos="0 0 0" quat="0 0 0 -1">
      <geom name="base_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
      <body name="shoulder_link" pos="0 0 0.163">
        <joint name="shoulder_pan_joint" type="hinge" axis="0 0 1" range="-6.28319 6.28319"/>
        <geom name="shoulder_link_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
        <body name="upper_arm_link" pos="0 0.138 0" quat="1 0 1 0">
          <joint name="shoulder_lift_joint" type="hinge" axis="0 1 0" range="-6.28319 6.28319"/>
          <geom name="upper_arm_link_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
          <body name="forearm_link" pos="0 -0.131 0.425">
            <joint name="elbow_joint" type="hinge" axis="0 1 0" range="-3.1415 3.1415"/>
            <geom name="forearm_link_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
            <body name="wrist_1_link" pos="0 0 0.392" quat="1 0 1 0">
              <joint name="wrist_1_joint" type="hinge" axis="0 1 0" range="-6.28319 6.28319"/>
              <geom name="wrist_1_link_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
              <body name="wrist_2_link" pos="0 0.127 0">
                <joint name="wrist_2_joint" type="hinge" axis="0 0 1" range="-6.28319 6.28319"/>
                <geom name="wrist_2_link_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                <body name="wrist_3_link" pos="0 0 0.1">
                  <joint name="wrist_3_joint" type="hinge" axis="0 1 0" range="-6.28319 6.28319"/>
                  <geom name="wrist_3_link_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                  <geom name="custom_tool_mount" type="box" pos="0 0.115 0.015" size="0.036 0.018 0.026" rgba="0.86 0.56 0.14 1"/>
                  <geom name="custom_tool_arm" type="box" pos="0.022 0.168 0.018" size="0.014 0.044 0.014" rgba="0.92 0.64 0.18 1"/>
                  <geom name="custom_tool_tip" type="box" pos="0.022 0.222 0.018" size="0.024 0.018 0.022" rgba="0.78 0.44 0.12 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>`
  ),
  jointNames: [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
  ],
  jointLimits: [
    [-6.28, 6.28],
    [-6.28, 6.28],
    [-3.14, 3.14],
    [-6.28, 6.28],
    [-6.28, 6.28],
    [-6.28, 6.28],
  ],
  homeQ: [0.35, -1.22, 1.52, -1.85, -1.57, 0.25],
  robotBodyNames: new Set([
    'base', 'shoulder_link', 'upper_arm_link', 'forearm_link',
    'wrist_1_link', 'wrist_2_link', 'wrist_3_link',
  ]),
  staticBaseBodies: new Set(['base']),
  cameraTarget: [0.06, 0.0, 0.40],
  cameraDistRange: [1.22, 1.52],
  viewCamPos: [1.56, 1.24, 1.18],
  viewCamTarget: [0.08, 0.0, 0.40],
  coldStartActions: [
    { id: 'ur_cold_j1', label: '冷启动 #1 (Shoulder Pan Z)', jointIndex: 0, qBase: [0.35, -1.22, 1.52, -1.85, -1.57, 0.25], qStart: -0.05, qEnd: 0.72, isColdStart: true },
    { id: 'ur_cold_j2', label: '冷启动 #2 (Shoulder Lift Y)', jointIndex: 1, qBase: [0.15, -1.22, 1.58, -1.85, -1.57, 0.25], qStart: -1.56, qEnd: -0.92, isColdStart: true },
    { id: 'ur_cold_j3', label: '冷启动 #3 (Elbow 正交轴 X)', jointIndex: 2, qBase: [1.48, -1.22, 1.48, -1.78, -1.57, 0.25], qStart: 1.08, qEnd: 1.82, isColdStart: true },
  ],
  candidateActions: [
    { id: 'ur_j1_a', label: 'J1 高位底座偏航', jointIndex: 0, qBase: [0.0, -1.48, 1.25, -1.45, -1.57, 0.2], qStart: -0.45, qEnd: 0.45 },
    { id: 'ur_j1_b', label: 'J1 伸展底座偏航', jointIndex: 0, qBase: [0.0, -0.98, 1.65, -2.10, -1.25, 0.4], qStart: 0.42, qEnd: -0.38 },
    { id: 'ur_j2_left', label: 'J2 左侧抬臂', jointIndex: 1, qBase: [1.10, -1.20, 1.52, -1.80, -1.57, 0.2], qStart: -1.55, qEnd: -0.92 },
    { id: 'ur_j2_right', label: 'J2 右侧抬臂', jointIndex: 1, qBase: [-0.95, -1.20, 1.52, -1.80, -1.57, 0.2], qStart: -1.55, qEnd: -0.92 },
    { id: 'ur_j3_front', label: 'J3 正面肘部扫掠', jointIndex: 2, qBase: [0.0, -1.25, 1.45, -1.75, -1.57, 0.3], qStart: 1.05, qEnd: 1.82 },
    { id: 'ur_j3_right', label: 'J3 右偏肘部扫掠', jointIndex: 2, qBase: [-1.25, -1.22, 1.45, -1.75, -1.40, 0.3], qStart: 1.05, qEnd: 1.80 },
    { id: 'ur_j3_high', label: 'J3 高位肘部扫掠', jointIndex: 2, qBase: [0.65, -1.52, 1.35, -1.50, -1.20, 0.1], qStart: 0.95, qEnd: 1.72 },
    { id: 'ur_j4_a', label: 'J4 腕1俯仰 A', jointIndex: 3, qBase: [0.25, -1.18, 1.55, -1.85, -1.57, 0.25], qStart: -2.35, qEnd: -1.35 },
    { id: 'ur_j4_b', label: 'J4 腕1俯仰 B', jointIndex: 3, qBase: [1.15, -1.22, 1.50, -1.80, -1.57, 0.25], qStart: -2.30, qEnd: -1.32 },
    { id: 'ur_j5_a', label: 'J5 腕2偏转', jointIndex: 4, qBase: [0.20, -1.18, 1.52, -1.85, -1.57, 0.35], qStart: -2.15, qEnd: -1.05 },
  ],
};

/* ============================================================================
 * 4. ALOHA ViperX 300s (6-DoF, Official mujoco_menagerie CAD)
 * ========================================================================== */
const ALOHA_SPEC: RobotSpec = {
  id: 'aloha',
  name: 'ALOHA ViperX 300s',
  shortName: 'ALOHA VX300s (暗黑吸光)',
  dof: 6,
  mechanism: 'serial',
  badgeText: '6-DoF 串联 · 官方 CAD',
  materialTag: '低反光哑黑阳极 (强红外吸收)',
  cadAssetId: 'aloha',
  mjcfXml: wrapMjcf(
    'aloha_vx300s_calib',
    `
    <body name="base_link" pos="0 0 0">
      <geom name="base_link_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
      <body name="shoulder_link" pos="0 0 0.079">
        <joint name="waist" type="hinge" axis="0 0 1" range="-3.14158 3.14158"/>
        <geom name="shoulder_link_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
        <body name="upper_arm_link" pos="0 0 0.04805">
          <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.85005 1.25664"/>
          <geom name="upper_arm_link_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
          <body name="upper_forearm_link" pos="0.05955 0 0.3">
            <joint name="elbow" type="hinge" axis="0 1 0" range="-1.76278 1.6057"/>
            <geom name="upper_forearm_link_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
            <body name="lower_forearm_link" pos="0.2 0 0">
              <joint name="forearm_roll" type="hinge" axis="1 0 0" range="-3.14158 3.14158"/>
              <geom name="lower_forearm_link_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
              <body name="wrist_link" pos="0.1 0 0">
                <joint name="wrist_angle" type="hinge" axis="0 1 0" range="-1.8675 2.23402"/>
                <geom name="wrist_link_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                <body name="gripper_link" pos="0.069744 0 0">
                  <joint name="wrist_rotate" type="hinge" axis="1 0 0" range="-3.14158 3.14158"/>
                  <body name="gripper_base" pos="0.035 0 0">
                    <geom name="gripper_base_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                    <body name="left_finger_link" pos="0.0191 -0.0141637 0.0211727" quat="1 -1 -1 1">
                      <joint name="left_finger" type="slide" axis="0 0 -1" range="0 0.041"/>
                      <geom name="left_finger_link_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                    </body>
                    <body name="right_finger_link" pos="-0.0191 -0.0141637 0.0211727" quat="1 1 1 1">
                      <joint name="right_finger" type="slide" axis="0 0 -1" range="0 0.041"/>
                      <geom name="right_finger_link_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                    </body>
                    <geom name="custom_tool_mount" type="box" pos="0.012 0.045 -0.010" size="0.016 0.020 0.022" rgba="0.86 0.56 0.14 1"/>
                    <geom name="custom_tool_arm" type="box" pos="0.055 0.056 -0.010" size="0.036 0.012 0.012" rgba="0.92 0.64 0.18 1"/>
                    <geom name="custom_tool_tip" type="box" pos="0.098 0.056 -0.010" size="0.018 0.022 0.016" rgba="0.78 0.44 0.12 1"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>`
  ),
  jointNames: ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate'],
  jointLimits: [
    [-3.14, 3.14],
    [-1.85, 1.25],
    [-1.76, 1.60],
    [-3.14, 3.14],
    [-1.86, 2.23],
    [-3.14, 3.14],
  ],
  homeQ: [0.0, -0.35, 0.45, 0.0, 0.45, 0.0],
  fixedJointValues: [
    { name: 'left_finger', value: 0.032 },
    { name: 'right_finger', value: 0.032 },
  ],
  robotBodyNames: new Set([
    'base_link', 'shoulder_link', 'upper_arm_link', 'upper_forearm_link',
    'lower_forearm_link', 'wrist_link', 'gripper_base', 'left_finger_link', 'right_finger_link',
  ]),
  staticBaseBodies: new Set(['base_link']),
  cameraTarget: [0.12, 0.0, 0.28],
  cameraDistRange: [0.96, 1.24],
  viewCamPos: [1.18, 0.94, 0.88],
  viewCamTarget: [0.12, 0.0, 0.26],
  coldStartActions: [
    { id: 'aloha_cold_j1', label: '冷启动 #1 (Waist 底座偏航 Z)', jointIndex: 0, qBase: [0.0, -0.35, 0.45, 0.0, 0.45, 0.0], qStart: -0.42, qEnd: 0.42, isColdStart: true },
    { id: 'aloha_cold_j2', label: '冷启动 #2 (Shoulder 肩俯仰 Y)', jointIndex: 1, qBase: [0.0, -0.35, 0.42, 0.0, 0.45, 0.0], qStart: -0.72, qEnd: -0.02, isColdStart: true },
    { id: 'aloha_cold_j3', label: '冷启动 #3 (Elbow 正交轴 X)', jointIndex: 2, qBase: [1.28, -0.38, 0.42, 0.0, 0.45, 0.0], qStart: 0.02, qEnd: 0.78, isColdStart: true },
  ],
  candidateActions: [
    { id: 'aloha_j1_a', label: 'J1 高抬偏航', jointIndex: 0, qBase: [0.0, -0.62, 0.25, 0.0, 0.65, 0.0], qStart: -0.45, qEnd: 0.42 },
    { id: 'aloha_j1_b', label: 'J1 伸展偏航', jointIndex: 0, qBase: [0.0, -0.10, 0.55, 0.35, 0.25, 0.0], qStart: 0.42, qEnd: -0.42 },
    { id: 'aloha_j2_l', label: 'J2 左侧俯仰', jointIndex: 1, qBase: [0.95, -0.35, 0.45, 0.0, 0.45, 0.0], qStart: -0.70, qEnd: -0.04 },
    { id: 'aloha_j2_r', label: 'J2 右侧俯仰', jointIndex: 1, qBase: [-0.95, -0.35, 0.45, 0.0, 0.45, 0.0], qStart: -0.70, qEnd: -0.04 },
    { id: 'aloha_j3_front', label: 'J3 正面肘屈', jointIndex: 2, qBase: [-0.15, -0.42, 0.40, 0.0, 0.45, 0.0], qStart: -0.02, qEnd: 0.76 },
    { id: 'aloha_j3_right', label: 'J3 右偏肘屈', jointIndex: 2, qBase: [-1.18, -0.42, 0.40, 0.0, 0.45, 0.0], qStart: -0.02, qEnd: 0.76 },
    { id: 'aloha_j4_a', label: 'J4 前臂滚转 A', jointIndex: 3, qBase: [0.25, -0.32, 0.35, 0.0, 0.75, 0.0], qStart: -0.62, qEnd: 0.62 },
    { id: 'aloha_j4_b', label: 'J4 前臂滚转 B', jointIndex: 3, qBase: [-0.65, -0.45, 0.42, 0.0, 0.85, 0.0], qStart: 0.60, qEnd: -0.60 },
    { id: 'aloha_j5_a', label: 'J5 腕部俯仰', jointIndex: 4, qBase: [0.20, -0.38, 0.35, 0.25, 0.45, 0.0], qStart: -0.25, qEnd: 0.85 },
  ],
};

/* ============================================================================
 * 5. OpenArm v1.0 (7-DoF, Official enactic/openarm_mujoco CAD)
 * ========================================================================== */
const OPENARM_SPEC: RobotSpec = {
  id: 'openarm',
  name: 'OpenArm v1.0 (Enactic)',
  shortName: 'OpenArm v1.0',
  dof: 7,
  mechanism: 'serial',
  badgeText: '7-DoF 串联 · 官方 CAD',
  materialTag: '哑黑碳纤 + CNC亮银铝',
  cadAssetId: 'openarm',
  mjcfXml: wrapMjcf(
    'openarm_v1_calib',
    `
    <body name="openarm_link0" pos="0 0 0">
      <geom name="openarm_link0_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
      <body name="openarm_link1" pos="0 0 0.0625">
        <joint name="openarm_joint1" type="hinge" axis="0 0 1" range="-1.396263 3.490659"/>
        <geom name="openarm_link1_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
        <body name="openarm_link2" pos="0 0 0.06">
          <joint name="openarm_joint2" type="hinge" axis="-1 0 0" range="-1.74533 1.74533"/>
          <geom name="openarm_link2_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
          <body name="openarm_link3" pos="0 0 0.06625">
            <joint name="openarm_joint3" type="hinge" axis="0 0 1" range="-1.570796 1.570796"/>
            <geom name="openarm_link3_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
            <body name="openarm_link4" pos="0 0 0.15375">
              <joint name="openarm_joint4" type="hinge" axis="0 1 0" range="0.0 2.443461"/>
              <geom name="openarm_link4_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
              <body name="openarm_link5" pos="0 0 0.0955">
                <joint name="openarm_joint5" type="hinge" axis="0 0 1" range="-1.570796 1.570796"/>
                <geom name="openarm_link5_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                <body name="openarm_link6" pos="0 0 0.1205">
                  <joint name="openarm_joint6" type="hinge" axis="1 0 0" range="-0.785398 0.785398"/>
                  <geom name="openarm_link6_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                  <body name="openarm_link7" pos="0 0 0">
                    <joint name="openarm_joint7" type="hinge" axis="0 1 0" range="-1.570796 1.570796"/>
                    <geom name="openarm_link7_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                    <body name="openarm_hand" pos="0 0 0.1001">
                      <geom name="openarm_hand_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                      <body name="openarm_right_finger" pos="0 -0.005 0.015">
                        <joint name="openarm_finger_joint1" type="slide" axis="0 -1 0" range="0 0.044"/>
                        <geom name="openarm_right_finger_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                      </body>
                      <body name="openarm_left_finger" pos="0 0.005 0.015">
                        <joint name="openarm_finger_joint2" type="slide" axis="0 1 0" range="0 0.044"/>
                        <geom name="openarm_left_finger_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                      </body>
                      <geom name="custom_tool_mount" type="box" pos="-0.042 0 0.015" size="0.014 0.036 0.018" rgba="0.86 0.56 0.14 1"/>
                      <geom name="custom_tool_arm" type="cylinder" pos="-0.052 0.020 0.068" size="0.012 0.038" rgba="0.92 0.64 0.18 1"/>
                      <geom name="custom_tool_tip" type="box" pos="-0.052 0.020 0.118" size="0.024 0.014 0.016" rgba="0.78 0.44 0.12 1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>`
  ),
  jointNames: [
    'openarm_joint1',
    'openarm_joint2',
    'openarm_joint3',
    'openarm_joint4',
    'openarm_joint5',
    'openarm_joint6',
    'openarm_joint7',
  ],
  jointLimits: [
    [-1.396, 3.49],
    [-1.745, 1.745],
    [-1.57, 1.57],
    [0.0, 2.443],
    [-1.57, 1.57],
    [-0.785, 0.785],
    [-1.57, 1.57],
  ],
  homeQ: [0.15, 0.18, 0.0, 1.25, 0.0, 0.15, 0.25],
  fixedJointValues: [
    { name: 'openarm_finger_joint1', value: 0.036 },
    { name: 'openarm_finger_joint2', value: 0.036 },
  ],
  robotBodyNames: new Set([
    'openarm_link0', 'openarm_link1', 'openarm_link2', 'openarm_link3',
    'openarm_link4', 'openarm_link5', 'openarm_link6', 'openarm_link7',
    'openarm_hand', 'openarm_right_finger', 'openarm_left_finger',
  ]),
  staticBaseBodies: new Set(['openarm_link0']),
  cameraTarget: [0.10, 0.0, 0.36],
  cameraDistRange: [1.05, 1.32],
  viewCamPos: [1.28, 1.02, 0.96],
  viewCamTarget: [0.10, 0.0, 0.36],
  coldStartActions: [
    { id: 'oa_cold_j1', label: '冷启动 #1 (Joint1 底座偏航 Z)', jointIndex: 0, qBase: [0.20, 0.20, 0.0, 1.30, 0.0, 0.15, 0.25], qStart: -0.22, qEnd: 0.56, isColdStart: true },
    { id: 'oa_cold_j2', label: '冷启动 #2 (Joint2 肩外展 X)', jointIndex: 1, qBase: [0.05, 0.10, 0.0, 1.28, 0.0, 0.10, 0.25], qStart: -0.32, qEnd: 0.38, isColdStart: true },
    { id: 'oa_cold_j4', label: '冷启动 #3 (Joint4 正交肘屈 Y)', jointIndex: 3, qBase: [0.05, 0.12, 0.0, 1.25, 0.0, 0.15, 0.25], qStart: 0.78, qEnd: 1.62, isColdStart: true },
  ],
  candidateActions: [
    { id: 'oa_j1_a', label: 'J1 伸展偏航', jointIndex: 0, qBase: [0.2, 0.35, 0.0, 1.05, 0.0, 0.2, 0.3], qStart: -0.30, qEnd: 0.55 },
    { id: 'oa_j1_b', label: 'J1 反折偏航', jointIndex: 0, qBase: [0.2, -0.32, 0.25, 1.48, -0.2, -0.15, 0.2], qStart: 0.65, qEnd: -0.18 },
    { id: 'oa_j2_l', label: 'J2 偏航外展 A', jointIndex: 1, qBase: [1.05, 0.12, 0.0, 1.25, 0.0, 0.15, 0.2], qStart: -0.34, qEnd: 0.36 },
    { id: 'oa_j2_r', label: 'J2 偏航外展 B', jointIndex: 1, qBase: [-0.75, 0.12, 0.0, 1.25, 0.0, 0.15, 0.2], qStart: -0.34, qEnd: 0.36 },
    { id: 'oa_j3_a', label: 'J3 上臂旋滚 A', jointIndex: 2, qBase: [0.15, 0.38, 0.0, 1.35, 0.0, 0.15, 0.2], qStart: -0.52, qEnd: 0.52 },
    { id: 'oa_j3_b', label: 'J3 上臂旋滚 B', jointIndex: 2, qBase: [0.85, -0.35, 0.0, 1.35, 0.2, 0.15, 0.2], qStart: 0.50, qEnd: -0.50 },
    { id: 'oa_j4_a', label: 'J4 偏置肘屈 A', jointIndex: 3, qBase: [1.10, 0.15, 0.25, 1.25, 0.0, 0.15, 0.2], qStart: 0.76, qEnd: 1.62 },
    { id: 'oa_j4_b', label: 'J4 偏置肘屈 B', jointIndex: 3, qBase: [-0.82, -0.18, -0.25, 1.25, 0.2, 0.15, 0.2], qStart: 0.78, qEnd: 1.60 },
    { id: 'oa_j5_a', label: 'J5 前臂旋腕', jointIndex: 4, qBase: [0.20, 0.15, 0.0, 1.35, 0.0, 0.45, 0.55], qStart: -0.58, qEnd: 0.58 },
    { id: 'oa_j6_a', label: 'J6 腕部摆动', jointIndex: 5, qBase: [0.15, 0.18, 0.0, 1.28, 0.25, 0.0, 0.35], qStart: -0.52, qEnd: 0.52 },
  ],
};

/* ============================================================================
 * 6. Custom 6-DoF Skew-Axis Asymmetric Serial Mechanism (HexaSkew-6)
 *    Non-orthogonal 25°/30°/45° skewed axes + multi-material titanium/copper
 * ========================================================================== */
const CUSTOM_SERIAL_SPEC: RobotSpec = {
  id: 'custom_serial',
  name: '自建 6-DoF 异形串联机构 (HexaSkew-6)',
  shortName: '自建 6-DoF 异形串联',
  dof: 6,
  mechanism: 'serial',
  badgeText: '6-DoF 非正交自建串联',
  materialTag: '钛银桁架 + 铜金谐波鼓',
  mjcfXml: wrapMjcf(
    'custom_hexaskew6_calib',
    `
    <body name="cs_base" pos="0 0 0">
      <geom name="cs_base_plate" type="cylinder" pos="0 0 0.025" size="0.115 0.025" rgba="0.16 0.19 0.24 1"/>
      <geom name="cs_base_column" type="cylinder" pos="0 0 0.095" size="0.072 0.055" rgba="0.24 0.28 0.35 1"/>
      <geom name="cs_base_bus" type="box" pos="-0.072 0 0.075" size="0.042 0.056 0.048" rgba="0.68 0.42 0.18 1"/>

      <body name="cs_link1" pos="0 0 0.175">
        <joint name="cs_j1" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
        <geom name="cs_l1_drum" type="cylinder" pos="0 0 0.025" size="0.068 0.042" rgba="0.76 0.48 0.20 1"/>
        <geom name="cs_l1_yoke" type="box" pos="0.036 0 0.095" size="0.054 0.048 0.062" rgba="0.72 0.77 0.84 1"/>
        <geom name="cs_l1_fin" type="box" pos="-0.032 0.038 0.088" size="0.022 0.028 0.050" rgba="0.22 0.26 0.32 1"/>

        <!-- Joint 2: 25° canted non-orthogonal shoulder axis! -->
        <body name="cs_link2" pos="0.062 0 0.135">
          <joint name="cs_j2" type="hinge" axis="0 0.9063 0.4226" range="-1.4 1.4"/>
          <geom name="cs_l2_hub" type="cylinder" pos="0 0 0" quat="0.7071 0.7071 0 0" size="0.058 0.056" rgba="0.76 0.48 0.20 1"/>
          <geom name="cs_l2_spar_a" type="box" pos="0.022 0.032 0.145" size="0.020 0.016 0.135" rgba="0.78 0.83 0.89 1"/>
          <geom name="cs_l2_spar_b" type="box" pos="0.022 -0.032 0.145" size="0.020 0.016 0.135" rgba="0.78 0.83 0.89 1"/>
          <geom name="cs_l2_brace" type="box" pos="0.036 0 0.155" size="0.026 0.044 0.038" rgba="0.20 0.24 0.30 1"/>

          <!-- Joint 3: 30° skewed elbow axis! -->
          <body name="cs_link3" pos="0.024 0 0.285">
            <joint name="cs_j3" type="hinge" axis="0.5 0.8660 0" range="-2.1 2.1"/>
            <geom name="cs_l3_drum" type="cylinder" pos="0 0 0" quat="0.7071 0.7071 0 0" size="0.052 0.050" rgba="0.76 0.48 0.20 1"/>
            <geom name="cs_l3_boom" type="capsule" fromto="0 0 0 0.21 0.04 0.10" size="0.038" rgba="0.74 0.79 0.86 1"/>
            <geom name="cs_l3_rib" type="box" pos="0.11 0.048 0.068" size="0.072 0.022 0.026" rgba="0.22 0.26 0.34 1"/>

            <!-- Joint 4: 45° diagonal forearm roll axis! -->
            <body name="cs_link4" pos="0.21 0.04 0.10">
              <joint name="cs_j4" type="hinge" axis="0.7071 0 0.7071" range="-2.6 2.6"/>
              <geom name="cs_l4_housing" type="cylinder" pos="0.05 0 0.05" quat="0.9239 0 0.3827 0" size="0.044 0.058" rgba="0.20 0.25 0.32 1"/>
              <geom name="cs_l4_bridge" type="box" pos="0.105 -0.018 0.105" size="0.048 0.032 0.036" rgba="0.80 0.85 0.90 1"/>

              <!-- Joint 5: Offset wrist pitch -->
              <body name="cs_link5" pos="0.135 0 0.135">
                <joint name="cs_j5" type="hinge" axis="0 1 0" range="-1.8 1.8"/>
                <geom name="cs_l5_drum" type="cylinder" pos="0 0 0" quat="0.7071 0.7071 0 0" size="0.038 0.042" rgba="0.76 0.48 0.20 1"/>
                <geom name="cs_l5_fork" type="box" pos="0.058 0 -0.014" size="0.046 0.028 0.024" rgba="0.75 0.80 0.87 1"/>

                <!-- Joint 6: Asymmetric twin-prong end effector -->
                <body name="cs_link6" pos="0.108 0 -0.014">
                  <joint name="cs_j6" type="hinge" axis="1 0 0" range="-2.8 2.8"/>
                  <geom name="cs_l6_flange" type="cylinder" pos="0.018 0 0" quat="0.7071 0 0.7071 0" size="0.036 0.018" rgba="0.22 0.26 0.34 1"/>
                  <geom name="cs_l6_prong_top" type="box" pos="0.068 0 0.032" size="0.042 0.014 0.010" rgba="0.82 0.86 0.92 1"/>
                  <geom name="cs_l6_prong_bot" type="box" pos="0.062 0.018 -0.028" size="0.036 0.016 0.010" rgba="0.18 0.62 0.78 1"/>

                  <!-- Toggleable unknown non-standard tool (excluded from Stage 2 CAD) -->
                  <geom name="custom_tool_mount" type="box" pos="0.032 -0.044 0.005" size="0.022 0.016 0.026" rgba="0.86 0.56 0.14 1"/>
                  <geom name="custom_tool_arm" type="box" pos="0.082 -0.056 0.010" size="0.042 0.012 0.012" rgba="0.92 0.64 0.18 1"/>
                  <geom name="custom_tool_tip" type="box" pos="0.130 -0.056 0.010" size="0.018 0.024 0.018" rgba="0.78 0.44 0.12 1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>`
  ),
  jointNames: ['cs_j1', 'cs_j2', 'cs_j3', 'cs_j4', 'cs_j5', 'cs_j6'],
  jointLimits: [
    [-2.8, 2.8],
    [-1.4, 1.4],
    [-2.1, 2.1],
    [-2.6, 2.6],
    [-1.8, 1.8],
    [-2.8, 2.8],
  ],
  homeQ: [0.0, -0.25, 0.72, 0.15, 0.55, 0.0],
  robotBodyNames: new Set([
    'cs_base', 'cs_link1', 'cs_link2', 'cs_link3', 'cs_link4', 'cs_link5', 'cs_link6',
  ]),
  staticBaseBodies: new Set(['cs_base']),
  cameraTarget: [0.14, 0.0, 0.45],
  cameraDistRange: [1.18, 1.48],
  viewCamPos: [1.48, 1.18, 1.12],
  viewCamTarget: [0.14, 0.0, 0.44],
  coldStartActions: [
    { id: 'cs_cold_j1', label: '冷启动 #1 (J1 底座 Z)', jointIndex: 0, qBase: [0.0, -0.25, 0.72, 0.15, 0.55, 0.0], qStart: -0.38, qEnd: 0.38, isColdStart: true },
    { id: 'cs_cold_j2', label: '冷启动 #2 (J2 25°倾肩轴)', jointIndex: 1, qBase: [0.0, -0.25, 0.72, 0.15, 0.55, 0.0], qStart: -0.58, qEnd: 0.12, isColdStart: true },
    { id: 'cs_cold_j3', label: '冷启动 #3 (J3 30°斜肘轴)', jointIndex: 2, qBase: [1.15, -0.25, 0.72, 0.15, 0.55, 0.0], qStart: 0.28, qEnd: 1.05, isColdStart: true },
  ],
  candidateActions: [
    { id: 'cs_j1_a', label: 'J1 伸展偏航', jointIndex: 0, qBase: [0.0, 0.05, 0.55, 0.2, 0.45, 0.0], qStart: -0.40, qEnd: 0.38 },
    { id: 'cs_j1_b', label: 'J1 后收偏航', jointIndex: 0, qBase: [0.0, -0.48, 0.85, -0.3, 0.65, 0.2], qStart: 0.40, qEnd: -0.38 },
    { id: 'cs_j2_l', label: 'J2 左侧斜俯仰', jointIndex: 1, qBase: [0.95, -0.25, 0.72, 0.15, 0.55, 0.0], qStart: -0.56, qEnd: 0.12 },
    { id: 'cs_j2_r', label: 'J2 右侧斜俯仰', jointIndex: 1, qBase: [-0.95, -0.25, 0.72, 0.15, 0.55, 0.0], qStart: -0.56, qEnd: 0.12 },
    { id: 'cs_j3_front', label: 'J3 正面斜肘屈', jointIndex: 2, qBase: [-0.2, -0.28, 0.68, 0.15, 0.55, 0.0], qStart: 0.25, qEnd: 1.02 },
    { id: 'cs_j3_right', label: 'J3 右偏斜肘屈', jointIndex: 2, qBase: [-1.05, -0.28, 0.68, 0.15, 0.55, 0.0], qStart: 0.25, qEnd: 1.02 },
    { id: 'cs_j4_a', label: 'J4 45°斜滚转 A', jointIndex: 3, qBase: [0.15, -0.22, 0.68, 0.0, 0.72, 0.0], qStart: -0.55, qEnd: 0.55 },
    { id: 'cs_j4_b', label: 'J4 45°斜滚转 B', jointIndex: 3, qBase: [-0.75, -0.32, 0.75, 0.0, 0.78, 0.0], qStart: 0.55, qEnd: -0.55 },
    { id: 'cs_j5_a', label: 'J5 腕部俯仰', jointIndex: 4, qBase: [0.2, -0.25, 0.68, 0.25, 0.55, 0.0], qStart: 0.05, qEnd: 0.98 },
  ],
};

/* ============================================================================
 * 7. 4-DoF Delta Parallel Robot (Closed-Loop 3P-R Mechanism)
 *    Overhead 3-pillar gantry + 3 upper arms + 6 carbon parallel rods + platform + Z-wrist
 * ========================================================================== */
function buildDeltaParallelMjcf(p: DeltaGeometryParams): string {
  const halfForearm = (p.forearmLen * 0.5).toFixed(4);
  const legAngles = [0, 120, 240];

  // 3 overhead gantry pillars placed at 60°, 180°, 300° (between the 3 active arms)
  const pillarRadiusTop = 0.26;
  const pillarRadiusBot = 0.39;
  const pillarAngles = [60, 180, 300];
  const pillarGeoms = pillarAngles
    .map((deg, idx) => {
      const rad = (deg * Math.PI) / 180;
      const x0 = (pillarRadiusBot * Math.cos(rad)).toFixed(4);
      const y0 = (pillarRadiusBot * Math.sin(rad)).toFixed(4);
      const x1 = (pillarRadiusTop * Math.cos(rad)).toFixed(4);
      const y1 = (pillarRadiusTop * Math.sin(rad)).toFixed(4);
      return `      <geom name="delta_pillar_${idx}" type="capsule" fromto="${x0} ${y0} 0.018 ${x1} ${y1} ${(p.baseZ + 0.015).toFixed(4)}" size="0.020" rgba="0.22 0.26 0.34 1"/>
      <geom name="delta_foot_${idx}" type="cylinder" pos="${x0} ${y0} 0.014" size="0.044 0.014" rgba="0.16 0.19 0.25 1"/>`;
    })
    .join('\n');

  // 3 active upper revolute arms at 0°, 120°, 240°
  // In local arm body rotated by phi around Z: +X points radially outward, +Y is tangential
  // Rotating around +Y by positive angle q pitches +X downward (-Z): exactly matching solveDeltaFK!
  const upperArmBodies = legAngles
    .map((deg, idx) => {
      const rad = (deg * Math.PI) / 180;
      const bx = (p.baseRadius * Math.cos(rad)).toFixed(4);
      const by = (p.baseRadius * Math.sin(rad)).toFixed(4);
      const qw = Math.cos(rad / 2).toFixed(6);
      const qz = Math.sin(rad / 2).toFixed(6);
      const L = p.upperLen;
      return `      <body name="delta_arm_${idx}" pos="${bx} ${by} ${p.baseZ.toFixed(4)}" quat="${qw} 0 0 ${qz}">
        <joint name="delta_m${idx + 1}" type="hinge" axis="0 1 0" range="-0.45 0.95"/>
        <geom name="delta_motor_${idx}" type="cylinder" pos="0 0 0" quat="0.7071 0.7071 0 0" size="0.042 0.046" rgba="0.18 0.22 0.28 1"/>
        <geom name="delta_arm_beam_${idx}" type="box" pos="${(L * 0.5).toFixed(4)} 0 0" size="${(L * 0.5).toFixed(4)} 0.022 0.018" rgba="0.80 0.85 0.91 1"/>
        <geom name="delta_arm_stripe_${idx}" type="box" pos="${(L * 0.52).toFixed(4)} 0 0.019" size="${(L * 0.38).toFixed(4)} 0.012 0.005" rgba="0.08 0.62 0.85 1"/>
        <geom name="delta_elbow_bar_${idx}" type="cylinder" pos="${L.toFixed(4)} 0 0" quat="0.7071 0.7071 0 0" size="0.016 0.034" rgba="0.22 0.26 0.34 1"/>
      </body>`;
    })
    .join('\n');

  // 6 passive carbon-fiber forearm rods (free joints synced to exact closed-loop Delta FK)
  const rodBodies: string[] = [];
  for (let i = 0; i < 3; i++) {
    for (const side of ['a', 'b']) {
      const rname = `delta_rod_${i}${side}`;
      rodBodies.push(`    <body name="${rname}" pos="0 0 0.6">
      <freejoint name="${rname}_free"/>
      <geom name="${rname}_tube" type="cylinder" pos="0 0 0" size="0.009 ${halfForearm}" rgba="0.18 0.21 0.27 1"/>
    </body>`);
    }
  }

  return wrapMjcf(
    'delta_parallel_calib',
    `
    <!-- Stationary Overhead Delta Gantry & Hexagonal Servo Housing -->
    <body name="delta_gantry" pos="0 0 0">
${pillarGeoms}
      <geom name="delta_top_plate" type="cylinder" pos="0 0 ${(p.baseZ + 0.022).toFixed(4)}" size="0.275 0.022" rgba="0.74 0.79 0.86 1"/>
      <geom name="delta_top_dome" type="cylinder" pos="0 0 ${(p.baseZ + 0.062).toFixed(4)}" size="0.165 0.024" rgba="0.18 0.22 0.28 1"/>
${upperArmBodies}
    </body>

    <!-- 6 Closed-Loop Carbon-Fiber Parallelogram Rods -->
${rodBodies.join('\n')}

    <!-- Moving End-Effector Platform (exact closed-loop Delta FK translation) + 4th Rotary Z-Wrist -->
    <body name="delta_platform" pos="0 0 0.46">
      <freejoint name="delta_platform_free"/>
      <geom name="delta_plat_hub" type="cylinder" pos="0 0 0" size="0.078 0.020" rgba="0.78 0.83 0.90 1"/>
      <geom name="delta_plat_ring" type="cylinder" pos="0 0 -0.024" size="0.056 0.012" rgba="0.08 0.62 0.85 1"/>
      <geom name="delta_plat_lug_a" type="box" pos="0.054 0 0.016" size="0.032 0.028 0.016" rgba="0.24 0.28 0.35 1"/>
      <geom name="delta_plat_lug_b" type="box" pos="-0.032 0.048 0.016" size="0.026 0.028 0.016" rgba="0.24 0.28 0.35 1"/>
      <geom name="delta_plat_lug_c" type="box" pos="-0.032 -0.048 0.016" size="0.026 0.028 0.016" rgba="0.24 0.28 0.35 1"/>

      <body name="delta_wrist" pos="0 0 -0.036">
        <joint name="delta_wrist_z" type="hinge" axis="0 0 1" range="-2.8 2.8"/>
        <geom name="delta_wrist_col" type="cylinder" pos="0 0 -0.028" size="0.036 0.028" rgba="0.22 0.26 0.34 1"/>
        <geom name="delta_wrist_head" type="box" pos="0.018 0 -0.068" size="0.048 0.030 0.018" rgba="0.82 0.86 0.92 1"/>
        <geom name="delta_wrist_jaw_l" type="box" pos="0.018 0.026 -0.104" size="0.014 0.008 0.024" rgba="0.20 0.24 0.30 1"/>
        <geom name="delta_wrist_jaw_r" type="box" pos="0.018 -0.026 -0.104" size="0.014 0.008 0.024" rgba="0.20 0.24 0.30 1"/>

        <!-- Unknown Non-Standard Tool (excluded from Stage 2 CAD) -->
        <geom name="custom_tool_mount" type="box" pos="-0.046 0 -0.064" size="0.018 0.024 0.018" rgba="0.86 0.56 0.14 1"/>
        <geom name="custom_tool_arm" type="cylinder" pos="-0.060 0.012 -0.108" size="0.013 0.032" rgba="0.92 0.64 0.18 1"/>
        <geom name="custom_tool_tip" type="box" pos="-0.060 0.012 -0.146" size="0.024 0.016 0.014" rgba="0.78 0.44 0.12 1"/>
      </body>
    </body>`
  );
}

const DELTA_PARALLEL_SPEC: RobotSpec = {
  id: 'delta_parallel',
  name: '4-DoF Delta 并联机械臂 (闭链 3P-R)',
  shortName: '4-DoF Delta 并联',
  dof: 4,
  mechanism: 'parallel_delta',
  badgeText: '4-DoF 闭链并联 · Delta',
  materialTag: '航空铝架 + 碳纤并联杆',
  mjcfXml: buildDeltaParallelMjcf(DEFAULT_DELTA_PARAMS),
  jointNames: ['delta_m1', 'delta_m2', 'delta_m3', 'delta_wrist_z'],
  jointLimits: [
    [-0.35, 0.85],
    [-0.35, 0.85],
    [-0.35, 0.85],
    [-2.8, 2.8],
  ],
  homeQ: [0.22, 0.22, 0.22, 0.0],
  robotBodyNames: new Set([
    'delta_gantry',
    'delta_arm_0',
    'delta_arm_1',
    'delta_arm_2',
    'delta_rod_0a',
    'delta_rod_0b',
    'delta_rod_1a',
    'delta_rod_1b',
    'delta_rod_2a',
    'delta_rod_2b',
    'delta_platform',
    'delta_wrist',
  ]),
  staticBaseBodies: new Set(['delta_gantry']),
  cameraTarget: [0.02, 0.0, 0.44],
  cameraDistRange: [1.22, 1.48],
  viewCamPos: [1.56, 1.24, 1.18],
  viewCamTarget: [0.04, 0.0, 0.46],
  deltaParams: DEFAULT_DELTA_PARAMS,
  coldStartActions: [
    { id: 'delta_cold_m1', label: '冷启动 #1 (主动臂 M1 闭链平移)', jointIndex: 0, qBase: [0.22, 0.20, 0.20, 0.0], qStart: -0.08, qEnd: 0.56, isColdStart: true },
    { id: 'delta_cold_m2', label: '冷启动 #2 (主动臂 M2 闭链平移)', jointIndex: 1, qBase: [0.20, 0.22, 0.20, 0.0], qStart: -0.08, qEnd: 0.56, isColdStart: true },
    { id: 'delta_cold_m3', label: '冷启动 #3 (主动臂 M3 闭链平移)', jointIndex: 2, qBase: [0.20, 0.20, 0.22, 0.0], qStart: -0.08, qEnd: 0.56, isColdStart: true },
  ],
  candidateActions: [
    { id: 'delta_m1_hi', label: 'M1 高位闭链扫掠', jointIndex: 0, qBase: [0.15, 0.02, 0.05, 0.3], qStart: -0.10, qEnd: 0.54 },
    { id: 'delta_m1_lo', label: 'M1 低位闭链扫掠', jointIndex: 0, qBase: [0.28, 0.42, 0.36, -0.3], qStart: 0.58, qEnd: -0.05 },
    { id: 'delta_m2_hi', label: 'M2 高位闭链扫掠', jointIndex: 1, qBase: [0.04, 0.15, 0.02, -0.4], qStart: -0.10, qEnd: 0.54 },
    { id: 'delta_m2_lo', label: 'M2 低位闭链扫掠', jointIndex: 1, qBase: [0.40, 0.28, 0.38, 0.4], qStart: 0.58, qEnd: -0.05 },
    { id: 'delta_m3_hi', label: 'M3 高位闭链扫掠', jointIndex: 2, qBase: [0.02, 0.05, 0.15, 0.2], qStart: -0.10, qEnd: 0.54 },
    { id: 'delta_m3_lo', label: 'M3 低位闭链扫掠', jointIndex: 2, qBase: [0.38, 0.40, 0.28, -0.2], qStart: 0.58, qEnd: -0.05 },
    { id: 'delta_wrist_a', label: 'J4 末端平台旋腕 A', jointIndex: 3, qBase: [0.24, 0.24, 0.24, 0.0], qStart: -0.78, qEnd: 0.78 },
    { id: 'delta_wrist_b', label: 'J4 偏置平台旋腕 B', jointIndex: 3, qBase: [0.42, 0.08, 0.22, 0.0], qStart: 0.75, qEnd: -0.75 },
  ],
};

export const ROBOT_CATALOG: Record<RobotId, RobotSpec> = {
  panda: PANDA_SPEC,
  iiwa14: IIWA14_SPEC,
  ur5e: UR5E_SPEC,
  aloha: ALOHA_SPEC,
  openarm: OPENARM_SPEC,
  custom_serial: CUSTOM_SERIAL_SPEC,
  delta_parallel: DELTA_PARALLEL_SPEC,
};

export const ROBOT_LIST: readonly RobotSpec[] = [
  PANDA_SPEC,
  IIWA14_SPEC,
  UR5E_SPEC,
  ALOHA_SPEC,
  OPENARM_SPEC,
  CUSTOM_SERIAL_SPEC,
  DELTA_PARALLEL_SPEC,
];

export function getRobotSpec(id: RobotId): RobotSpec {
  return ROBOT_CATALOG[id] ?? PANDA_SPEC;
}
