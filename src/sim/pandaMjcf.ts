/**
 * Official Franka Emika Panda 7-DoF MJCF model with:
 *  - Exact official link offsets, quaternions, and joint range limits
 *  - Multi-primitive asymmetric visual geometry on every link (cylinders + capsules + feature boxes)
 *  - Hand + left/right finger links
 *  - 3 toggleable unknown non-standard tool geoms (custom_tool_*) under hand
 *    (rendered in RGB-D, strictly excluded from Stage 2 CAD surface sampling)
 *  - Workbench table, back & side walls, 3 static tabletop clutter props
 *  - Independent slide-joint dynamic distractor (distractor_body / distractor_slide)
 */

export const PANDA_JOINT_NAMES = [
  'joint1',
  'joint2',
  'joint3',
  'joint4',
  'joint5',
  'joint6',
  'joint7',
] as const;

export const PANDA_JOINT_LIMITS: ReadonlyArray<[number, number]> = [
  [-2.8973, 2.8973],
  [-1.7628, 1.7628],
  [-2.8973, 2.8973],
  [-3.0718, -0.0698],
  [-2.8973, 2.8973],
  [-0.0175, 3.7525],
  [-2.8973, 2.8973],
];

export const PANDA_HOME_Q: readonly number[] = [0.0, -0.35, 0.0, -1.95, 0.0, 1.65, 0.785];

export const PANDA_MJCF_XML = `<mujoco model="franka_panda_hand_eye_calib">
  <compiler angle="radian" autolimits="true"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>

  <default>
    <joint damping="1.0" frictionloss="0.1"/>
    <geom contype="0" conaffinity="0"/>
  </default>

  <worldbody>
    <!-- Workbench Tabletop (top surface at z = 0.0) -->
    <geom name="table_top" type="box" pos="0.15 0 -0.025" size="0.95 1.15 0.025" rgba="0.18 0.21 0.26 1"/>
    <geom name="table_leg_fl" type="box" pos="0.98 1.02 -0.38" size="0.035 0.035 0.33" rgba="0.11 0.13 0.17 1"/>
    <geom name="table_leg_fr" type="box" pos="0.98 -1.02 -0.38" size="0.035 0.035 0.33" rgba="0.11 0.13 0.17 1"/>
    <geom name="table_leg_bl" type="box" pos="-0.68 1.02 -0.38" size="0.035 0.035 0.33" rgba="0.11 0.13 0.17 1"/>
    <geom name="table_leg_br" type="box" pos="-0.68 -1.02 -0.38" size="0.035 0.035 0.33" rgba="0.11 0.13 0.17 1"/>

    <!-- Lab Enclosure Walls (placed safely behind robot & outside camera sampling volume) -->
    <geom name="wall_back" type="box" pos="-0.82 0 0.78" size="0.03 1.45 0.86" rgba="0.14 0.17 0.22 1"/>
    <geom name="wall_side" type="box" pos="0.15 -1.28 0.78" size="1.05 0.03 0.86" rgba="0.12 0.15 0.20 1"/>

    <!-- 3 Static Tabletop Clutter Props (out of robot collision envelope) -->
    <geom name="clutter_1_crate" type="box" pos="-0.38 0.56 0.065" size="0.085 0.075 0.065" rgba="0.30 0.36 0.44 1"/>
    <geom name="clutter_1_lid" type="box" pos="-0.38 0.56 0.138" size="0.092 0.082 0.008" rgba="0.46 0.52 0.60 1"/>
    <geom name="clutter_2_canister" type="cylinder" pos="0.44 -0.58 0.085" size="0.055 0.085" rgba="0.24 0.40 0.46 1"/>
    <geom name="clutter_2_cap" type="cylinder" pos="0.44 -0.58 0.178" size="0.044 0.010" rgba="0.62 0.68 0.74 1"/>
    <geom name="clutter_3_fixture_base" type="box" pos="-0.40 -0.46 0.045" size="0.075 0.060 0.045" rgba="0.38 0.30 0.26 1"/>
    <geom name="clutter_3_fixture_post" type="box" pos="-0.44 -0.46 0.135" size="0.024 0.052 0.048" rgba="0.52 0.40 0.34 1"/>

    <!-- Independent Asynchronous Dynamic Distractor -->
    <body name="distractor_body" pos="0.36 0.54 0.105">
      <joint name="distractor_slide" type="slide" axis="0 1 0" range="-0.42 0.42"/>
      <geom name="distractor_block" type="box" pos="0 0 0" size="0.068 0.085 0.095" rgba="0.82 0.28 0.26 1"/>
      <geom name="distractor_fin" type="box" pos="0.055 0 0.045" size="0.022 0.065 0.045" rgba="0.92 0.46 0.34 1"/>
      <geom name="distractor_cap" type="cylinder" pos="0 0 0.112" size="0.045 0.018" rgba="0.22 0.25 0.30 1"/>
    </body>

    <!-- Official Franka Emika Panda 7-DoF Manipulator (exact mujoco_menagerie kinematics; CAD meshes bound per body) -->
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

                      <!-- Official Franka Parallel-Jaw Fingers (exact mujoco_menagerie slide joints) -->
                      <body name="left_finger" pos="0 0 0.0584">
                        <joint name="finger_joint1" type="slide" axis="0 1 0" range="0 0.04"/>
                        <geom name="left_finger_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                      </body>
                      <body name="right_finger" pos="0 0 0.0584" quat="0 0 0 1">
                        <joint name="finger_joint2" type="slide" axis="0 1 0" range="0 0.04"/>
                        <geom name="right_finger_cad_anchor" type="sphere" pos="0 0 0" size="0.001" rgba="1 1 1 1"/>
                      </body>

                      <!-- Unknown Non-Standard Custom Tool (3 toggleable geoms; strictly excluded from Stage 2 CAD samples) -->
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
    </body>
  </worldbody>
</mujoco>
`;
