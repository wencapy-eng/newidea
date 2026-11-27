import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import type { CameraIntrinsics, SensorConfig } from '../calib/types';
import { cameraCenterInBase, type SE3, type SeededRNG } from '../math/se3';
import type { GeomMeta, MujocoPandaEngine } from './mujocoEngine';
import {
  createRobotMaterialPalette,
  getMaterialOpticalProps,
  type MaterialOpticalProps,
  type RobotMaterialName,
} from './pandaCadAssets';

const DEPTH_MAX_RANGE_M = 8.0;

/**
 * Custom GLSL ShaderMaterial for Pass 2 exact linear eye-space depth:
 *  - z = -mvPos.z (meters) packed into 24-bit RGB (0..8m -> ~0.48 um resolution)
 *  - A channel stores abs(dot(viewNormal, normalize(-mvPos))) strictly for physical
 *    grazing-angle & metallic hole simulation (NEVER passed to calibration algorithms)
 */
const DEPTH_VERTEX_SHADER = `
  varying vec3 vMvPos;
  varying vec3 vViewNormal;
  void main() {
    vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
    vMvPos = mvPos.xyz;
    vViewNormal = normalize(normalMatrix * normal);
    gl_Position = projectionMatrix * mvPos;
  }
`;

const DEPTH_FRAGMENT_SHADER = `
  precision highp float;
  varying vec3 vMvPos;
  varying vec3 vViewNormal;
  uniform float uMaxRange;
  void main() {
    float z = -vMvPos.z;
    if (z <= 0.0 || z >= uMaxRange) {
      gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
      return;
    }
    float normZ = clamp(z / uMaxRange, 0.0, 1.0);
    float val = floor(normZ * 16777215.0 + 0.5);
    float r = floor(val / 65536.0);
    float g = floor((val - r * 65536.0) / 256.0);
    float b = val - r * 65536.0 - g * 256.0;
    float cosInc = abs(dot(normalize(vViewNormal), normalize(-vMvPos)));
    gl_FragColor = vec4(r / 255.0, g / 255.0, b / 255.0, clamp(cosInc, 0.0, 1.0));
  }
`;

/**
 * Pass 3 Material Optical G-Buffer shader: encodes (metalness, gloss, darkIR) in exact linear [0..1] RGB
 * so virtual RealSense D435i noise reflects real per-pixel surface optics (brushed steel, matte black IR
 * absorption, painted lacquer) without leaking any material labels to calibration algorithms.
 */
const OPTICAL_VERTEX_SHADER = `
  void main() {
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

const OPTICAL_FRAGMENT_SHADER = `
  precision highp float;
  uniform vec3 uOptical;
  void main() {
    gl_FragColor = vec4(uOptical, 1.0);
  }
`;

function createOpticalMaterial(props: MaterialOpticalProps): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    uniforms: {
      uOptical: { value: new THREE.Vector3(props.metalness, props.gloss, props.darkIR) },
    },
    vertexShader: OPTICAL_VERTEX_SHADER,
    fragmentShader: OPTICAL_FRAGMENT_SHADER,
    side: THREE.FrontSide,
  });
}

function detectSoftwareGl(): boolean {
  try {
    const probeCanvas = document.createElement('canvas');
    const gl = probeCanvas.getContext('webgl2');
    if (!gl) return true;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
    const gpu = String(gl.getParameter(param) || '');
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
  } catch {
    return true;
  }
}

/** Build a Three.js BufferGeometry matching a MuJoCo geom with Z-axis alignment */
function createGeomGeometry(g: GeomMeta): THREE.BufferGeometry {
  if (g.type === 6) {
    return new THREE.BoxGeometry(g.size[0] * 2, g.size[1] * 2, g.size[2] * 2);
  }
  if (g.type === 5) {
    // MuJoCo cylinder is along local Z; Three.js CylinderGeometry is along Y -> rotateX(PI/2)
    const geo = new THREE.CylinderGeometry(g.size[0], g.size[0], g.size[1] * 2, 24);
    geo.rotateX(Math.PI / 2);
    return geo;
  }
  if (g.type === 3) {
    // MuJoCo capsule is along local Z; Three.js CapsuleGeometry is along Y -> rotateX(PI/2)
    const geo = new THREE.CapsuleGeometry(g.size[0], g.size[1] * 2, 10, 20);
    geo.rotateX(Math.PI / 2);
    return geo;
  }
  if (g.type === 2) {
    return new THREE.SphereGeometry(g.size[0], 20, 16);
  }
  return new THREE.BoxGeometry(0.05, 0.05, 0.05);
}

/** Build a crisp wireframe + translucent pyramid camera frustum oriented in Three.js camera local space (-Z forward, +Y up) */
function createCameraFrustumGroup(colorHex: number, scale = 0.19): THREE.Group {
  const group = new THREE.Group();
  group.matrixAutoUpdate = false;

  const w = scale * 0.72;
  const h = scale * 0.54;
  const d = scale;

  const positions = new Float32Array([
    0, 0, 0, -w, h, -d,
    0, 0, 0, w, h, -d,
    0, 0, 0, w, -h, -d,
    0, 0, 0, -w, -h, -d,
    -w, h, -d, w, h, -d,
    w, h, -d, w, -h, -d,
    w, -h, -d, -w, -h, -d,
    -w, -h, -d, -w, h, -d,
    -w * 0.25, h, -d, 0, h * 1.32, -d,
    0, h * 1.32, -d, w * 0.25, h, -d,
  ]);
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const lineMat = new THREE.LineBasicMaterial({ color: colorHex });
  group.add(new THREE.LineSegments(lineGeo, lineMat));

  const bodyGeo = new THREE.BoxGeometry(0.090, 0.025, 0.026);
  const bodyMat = new THREE.MeshStandardMaterial({
    color: colorHex,
    roughness: 0.35,
    metalness: 0.45,
  });
  const bodyMesh = new THREE.Mesh(bodyGeo, bodyMat);
  bodyMesh.position.set(0, 0, 0.013);
  group.add(bodyMesh);

  const planeGeo = new THREE.PlaneGeometry(w * 2, h * 2);
  const planeMat = new THREE.MeshBasicMaterial({
    color: colorHex,
    transparent: true,
    opacity: 0.12,
    side: THREE.DoubleSide,
    depthWrite: false,
  });
  const planeMesh = new THREE.Mesh(planeGeo, planeMat);
  planeMesh.position.set(0, 0, -d);
  group.add(planeMesh);

  const axisLen = 0.075;
  const axPos = new Float32Array([
    0, 0, 0, axisLen, 0, 0,
    0, 0, 0, 0, -axisLen, 0,
    0, 0, 0, 0, 0, -axisLen,
  ]);
  const axCols = new Float32Array([
    1, 0.3, 0.3, 1, 0.3, 0.3,
    0.2, 0.9, 0.4, 0.2, 0.9, 0.4,
    0.25, 0.65, 1, 0.25, 0.65, 1,
  ]);
  const axGeo = new THREE.BufferGeometry();
  axGeo.setAttribute('position', new THREE.BufferAttribute(axPos, 3));
  axGeo.setAttribute('color', new THREE.BufferAttribute(axCols, 3));
  group.add(new THREE.LineSegments(axGeo, new THREE.LineBasicMaterial({ vertexColors: true })));

  return group;
}

/** Convert CV extrinsic T_cb to Three.js world matrix M_cam_world = [R_cb^T * diag(1,-1,-1), -R_cb^T * t_cb] */
export function cvExtrinsicToThreeCameraMatrix(T_cb: SE3, outMat = new THREE.Matrix4()): THREE.Matrix4 {
  const R = T_cb.R;
  const p = cameraCenterInBase(T_cb);
  outMat.set(
    R[0], -R[3], -R[6], p[0],
    R[1], -R[4], -R[7], p[1],
    R[2], -R[5], -R[8], p[2],
    0, 0, 0, 1
  );
  return outMat;
}

export function computeCameraIntrinsics(width: number, height: number, fovDeg: number): CameraIntrinsics {
  const fovRad = (fovDeg * Math.PI) / 180;
  const fy = height / (2 * Math.tan(fovRad / 2));
  const fx = fy;
  return {
    width,
    height,
    fx,
    fy,
    cx: width / 2,
    cy: height / 2,
  };
}

interface RenderableMeshBinding {
  mesh: THREE.Mesh;
  pbrMat: THREE.Material;
  opticalMat: THREE.ShaderMaterial;
}

export class PandaRGBDScene {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly viewCam: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;
  readonly virtualCam: THREE.PerspectiveCamera;
  readonly isSoftwareGl: boolean;

  private readonly engine: MujocoPandaEngine;
  private readonly geomObjects: THREE.Object3D[] = [];
  private readonly meshBindings: RenderableMeshBinding[] = [];
  private readonly robotMaterials: Record<RobotMaterialName, THREE.Material>;
  private readonly opticalMatCache = new Map<string, THREE.ShaderMaterial>();
  private readonly helpersGroup: THREE.Group;
  private readonly frustumGT: THREE.Group;
  private readonly frustumStage1: THREE.Group;
  private readonly frustumStage2: THREE.Group;
  /** V3 Class IV stress test: static aluminum stanchion slicing across camera-robot sightline */
  private readonly frontPillarGroup: THREE.Group;
  /** V3 Class III stress test: coupled non-rigid whipping flexible cable attached to robot arm */
  private readonly flexCableGroup: THREE.Group;
  private readonly flexCableMesh: THREE.Mesh;

  private readonly rgbRT: THREE.WebGLRenderTarget;
  private readonly depthRT: THREE.WebGLRenderTarget;
  private readonly matRT: THREE.WebGLRenderTarget;
  private readonly depthShaderMat: THREE.ShaderMaterial;
  private readonly rgbReadBuf: Uint8Array;
  private readonly depthReadBuf: Uint8Array;
  private readonly matReadBuf: Uint8Array;

  private readonly _tmpMat4 = new THREE.Matrix4();
  private readonly width: number;
  private readonly height: number;
  private readonly fovDeg: number;

  private renderBusy = false;
  private renderDirty = false;
  private frameId = 0;
  private disposed = false;
  private resizeObserver: ResizeObserver;
  private container: HTMLElement;

  constructor(
    container: HTMLElement,
    engine: MujocoPandaEngine,
    sensorWidth = 320,
    sensorHeight = 240,
    fovDeg = 58
  ) {
    this.container = container;
    this.engine = engine;
    this.width = sensorWidth;
    this.height = sensorHeight;
    this.fovDeg = fovDeg;
    this.isSoftwareGl = detectSoftwareGl();

    this.renderer = new THREE.WebGLRenderer({
      antialias: !this.isSoftwareGl,
      powerPreference: 'high-performance',
      preserveDrawingBuffer: false,
    });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    this.renderer.setPixelRatio(this.isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = !this.isSoftwareGl;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    const rect = container.getBoundingClientRect();
    const initW = Math.max(320, Math.floor(rect.width || 720));
    const initH = Math.max(260, Math.floor(rect.height || 440));
    this.renderer.setSize(initW, initH);
    this.renderer.domElement.style.display = 'block';
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = '100%';
    container.innerHTML = '';
    container.appendChild(this.renderer.domElement);

    // Scene setup: right-handed Z-up matching MuJoCo base frame {B}
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x0b0f17);
    this.scene.fog = new THREE.Fog(0x0b0f17, 3.8, 8.8);

    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();

    // Studio 3-point lighting with crisp sculptural contrast (Z-up)
    const hemi = new THREE.HemisphereLight(0xb8c7dc, 0x162032, 0.52);
    hemi.position.set(0, 0, 3.5);
    this.scene.add(hemi);

    const keyLight = new THREE.DirectionalLight(0xfff5e8, 1.55);
    keyLight.position.set(1.9, 1.4, 2.6);
    if (!this.isSoftwareGl) {
      keyLight.castShadow = true;
      keyLight.shadow.mapSize.set(1024, 1024);
      keyLight.shadow.camera.near = 0.5;
      keyLight.shadow.camera.far = 7.0;
      keyLight.shadow.camera.left = -1.4;
      keyLight.shadow.camera.right = 1.4;
      keyLight.shadow.camera.top = 1.4;
      keyLight.shadow.camera.bottom = -1.4;
      keyLight.shadow.bias = -0.0005;
      keyLight.shadow.normalBias = 0.02;
    }
    this.scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x7dd3fc, 0.72);
    rimLight.position.set(-1.5, -1.6, 1.8);
    this.scene.add(rimLight);

    // Observation camera: strictly Z-up, framed for active robot!
    const [vcx, vcy, vcz] = engine.spec.viewCamPos;
    const [vtx, vty, vtz] = engine.spec.viewCamTarget;
    this.viewCam = new THREE.PerspectiveCamera(44, initW / initH, 0.05, 25);
    this.viewCam.up.set(0, 0, 1);
    this.viewCam.position.set(vcx, vcy, vcz);

    this.controls = new OrbitControls(this.viewCam, this.renderer.domElement);
    this.controls.target.set(vtx, vty, vtz);
    this.controls.enableDamping = !this.isSoftwareGl;
    this.controls.dampingFactor = 0.08;
    this.controls.maxDistance = 5.5;
    this.controls.minDistance = 0.40;
    this.controls.update();
    this.controls.addEventListener('change', () => this.requestRender());

    // Build shared multi-robot PBR material palette and Three.js meshes directly from MuJoCo model geoms
    this.robotMaterials = createRobotMaterialPalette();
    this.buildMujocoMeshes();

    // Build V3 Class IV static front occlusion pillar & Class III dynamic whipping flexible cable
    // (registered in meshBindings so all 3 WebGL2 passes render them physically, NEVER in CAD samples)
    this.frontPillarGroup = this.buildFrontOcclusionPillar();
    this.frontPillarGroup.visible = false;
    this.scene.add(this.frontPillarGroup);

    const cableBuilt = this.buildFlexCable();
    this.flexCableGroup = cableBuilt.group;
    this.flexCableMesh = cableBuilt.mesh;
    this.flexCableGroup.visible = false;
    this.scene.add(this.flexCableGroup);

    // Helpers group (hidden during virtual RealSense D435i RGB-D rendering!)
    this.helpersGroup = new THREE.Group();
    this.scene.add(this.helpersGroup);

    const grid = new THREE.GridHelper(2.0, 20, 0x26354d, 0x162030);
    grid.rotation.x = Math.PI / 2;
    grid.position.set(0.15, 0, 0.0015);
    this.helpersGroup.add(grid);

    const baseAxes = new THREE.AxesHelper(0.24);
    baseAxes.position.set(0, 0, 0.003);
    this.helpersGroup.add(baseAxes);

    // 3 Color-coded Camera Frustums: Green=GT, Yellow=Stage 1, Cyan=Stage 2
    this.frustumGT = createCameraFrustumGroup(0x10b981, 0.20);
    this.frustumStage1 = createCameraFrustumGroup(0xf59e0b, 0.19);
    this.frustumStage2 = createCameraFrustumGroup(0x06b6d4, 0.18);
    this.frustumStage1.visible = false;
    this.frustumStage2.visible = false;
    this.helpersGroup.add(this.frustumGT, this.frustumStage1, this.frustumStage2);

    // Virtual RealSense D435i camera & offscreen render targets
    this.virtualCam = new THREE.PerspectiveCamera(
      fovDeg,
      sensorWidth / sensorHeight,
      0.08,
      DEPTH_MAX_RANGE_M + 1.0
    );
    this.virtualCam.matrixAutoUpdate = false;

    this.rgbRT = new THREE.WebGLRenderTarget(sensorWidth, sensorHeight, {
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
      colorSpace: THREE.SRGBColorSpace,
    });

    this.depthRT = new THREE.WebGLRenderTarget(sensorWidth, sensorHeight, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
      colorSpace: THREE.NoColorSpace,
    });

    this.matRT = new THREE.WebGLRenderTarget(sensorWidth, sensorHeight, {
      minFilter: THREE.NearestFilter,
      magFilter: THREE.NearestFilter,
      format: THREE.RGBAFormat,
      type: THREE.UnsignedByteType,
      depthBuffer: true,
      colorSpace: THREE.NoColorSpace,
    });

    this.depthShaderMat = new THREE.ShaderMaterial({
      uniforms: {
        uMaxRange: { value: DEPTH_MAX_RANGE_M },
      },
      vertexShader: DEPTH_VERTEX_SHADER,
      fragmentShader: DEPTH_FRAGMENT_SHADER,
      side: THREE.FrontSide,
    });

    this.rgbReadBuf = new Uint8Array(sensorWidth * sensorHeight * 4);
    this.depthReadBuf = new Uint8Array(sensorWidth * sensorHeight * 4);
    this.matReadBuf = new Uint8Array(sensorWidth * sensorHeight * 4);

    // Sync initial geom transforms from MuJoCo
    this.syncFromMujoco();

    // Responsive container resize
    this.resizeObserver = new ResizeObserver((entries) => {
      if (this.disposed) return;
      for (const entry of entries) {
        const w = Math.max(240, Math.floor(entry.contentRect.width));
        const h = Math.max(200, Math.floor(entry.contentRect.height));
        this.viewCam.aspect = w / h;
        this.viewCam.updateProjectionMatrix();
        this.renderer.setSize(w, h);
        this.requestRender();
      }
    });
    this.resizeObserver.observe(container);

    if (!this.isSoftwareGl) {
      this.renderer.compile(this.scene, this.viewCam);
    }
    this.requestRender();
  }

  private getCachedOpticalMaterial(key: string, props: MaterialOpticalProps): THREE.ShaderMaterial {
    const hit = this.opticalMatCache.get(key);
    if (hit) return hit;
    const mat = createOpticalMaterial(props);
    this.opticalMatCache.set(key, mat);
    return mat;
  }

  private buildMujocoMeshes(): void {
    for (let i = 0; i < this.engine.geoms.length; i++) {
      const g = this.engine.geoms[i];

      if (g.cadMeshes && g.cadMeshes.length > 0) {
        // Official multi-material robot CAD meshes bound directly in body local frame
        const group = new THREE.Group();
        group.matrixAutoUpdate = false;
        group.name = g.name;
        group.userData = { geomMeta: g, isSharedRobotCadGroup: true };

        for (let mIdx = 0; mIdx < g.cadMeshes.length; mIdx++) {
          const cad = g.cadMeshes[mIdx];
          const geo = new THREE.BufferGeometry();
          geo.setAttribute('position', new THREE.BufferAttribute(cad.positions, 3));
          geo.setAttribute('normal', new THREE.BufferAttribute(cad.normals, 3));
          geo.setIndex(new THREE.BufferAttribute(cad.indices, 1));
          geo.computeBoundingSphere();

          const mat =
            this.robotMaterials[cad.material as RobotMaterialName] ?? this.robotMaterials.white;
          const optProps = getMaterialOpticalProps(cad.material);
          const optMat = this.getCachedOpticalMaterial(`cad_${cad.material}`, optProps);

          const mesh = new THREE.Mesh(geo, mat);
          mesh.castShadow = !this.isSoftwareGl;
          mesh.receiveShadow = !this.isSoftwareGl;
          mesh.name = `${g.bodyName}_${cad.material}`;
          group.add(mesh);
          this.meshBindings.push({ mesh, pbrMat: mat, opticalMat: optMat });
        }

        this.scene.add(group);
        this.geomObjects.push(group);
        continue;
      }

      const geo = createGeomGeometry(g);
      const color = new THREE.Color().setRGB(
        g.rgba[0],
        g.rgba[1],
        g.rgba[2],
        THREE.SRGBColorSpace
      );

      const lum = g.rgba[0] * 0.299 + g.rgba[1] * 0.587 + g.rgba[2] * 0.114;
      let roughness = 0.55;
      let metalness = 0.15;
      let envMapIntensity = 0.75;
      let optProps: MaterialOpticalProps = { metalness: 0.12, gloss: 0.35, darkIR: 0.22 };

      if (g.category === 'robot') {
        const isDarkCarbon = lum < 0.28;
        const isBrightMetal = lum > 0.64 && Math.abs(g.rgba[0] - g.rgba[2]) < 0.15;
        if (isBrightMetal) {
          roughness = 0.22;
          metalness = 0.80;
          envMapIntensity = 1.0;
          optProps = { metalness: 0.82, gloss: 0.78, darkIR: 0.06 };
        } else if (isDarkCarbon) {
          roughness = 0.44;
          metalness = 0.28;
          envMapIntensity = 0.72;
          optProps = { metalness: 0.24, gloss: 0.42, darkIR: 0.78 };
        } else {
          roughness = 0.32;
          metalness = 0.35;
          envMapIntensity = 0.86;
          optProps = { metalness: 0.36, gloss: 0.65, darkIR: 0.18 };
        }
      } else if (g.category === 'custom_tool') {
        roughness = 0.28;
        metalness = 0.52;
        envMapIntensity = 0.95;
        optProps = { metalness: 0.55, gloss: 0.68, darkIR: 0.15 };
      } else if (g.category === 'distractor') {
        roughness = 0.46;
        metalness = 0.12;
        envMapIntensity = 0.75;
        optProps = { metalness: 0.14, gloss: 0.42, darkIR: 0.24 };
      } else if (g.name === 'table_top') {
        roughness = 0.86;
        metalness = 0.05;
        envMapIntensity = 0.35;
        optProps = { metalness: 0.05, gloss: 0.18, darkIR: 0.32 };
      } else {
        roughness = 0.90;
        metalness = 0.04;
        envMapIntensity = 0.28;
        optProps = { metalness: 0.04, gloss: 0.15, darkIR: 0.28 };
      }

      const mat = new THREE.MeshStandardMaterial({
        color,
        roughness,
        metalness,
        envMapIntensity,
      });

      const optKey = `prim_${optProps.metalness.toFixed(2)}_${optProps.gloss.toFixed(2)}_${optProps.darkIR.toFixed(2)}`;
      const optMat = this.getCachedOpticalMaterial(optKey, optProps);

      const mesh = new THREE.Mesh(geo, mat);
      mesh.matrixAutoUpdate = false;
      mesh.castShadow = !this.isSoftwareGl && g.category !== 'environment';
      mesh.receiveShadow = !this.isSoftwareGl;
      mesh.name = g.name;
      mesh.userData = { geomMeta: g };

      this.scene.add(mesh);
      this.geomObjects.push(mesh);
      this.meshBindings.push({ mesh, pbrMat: mat, opticalMat: optMat });
    }
  }

  /** Sync all Three.js geom meshes/CAD groups directly from MuJoCo geom_xpos & geom_xmat (Z-up, zero swap) */
  syncFromMujoco(): void {
    const xpos = this.engine.getGeomXpos();
    const xmat = this.engine.getGeomXmat();

    for (let i = 0; i < this.geomObjects.length; i++) {
      const obj = this.geomObjects[i];
      const m = i * 9;
      const p = i * 3;
      obj.matrix.set(
        xmat[m + 0], xmat[m + 1], xmat[m + 2], xpos[p + 0],
        xmat[m + 3], xmat[m + 4], xmat[m + 5], xpos[p + 1],
        xmat[m + 6], xmat[m + 7], xmat[m + 8], xpos[p + 2],
        0, 0, 0, 1
      );
      obj.matrixWorldNeedsUpdate = true;
    }
  }

  private buildFrontOcclusionPillar(): THREE.Group {
    const group = new THREE.Group();
    group.name = 'v3_front_occlusion_pillar';

    // Extruded brushed aluminum 42x42mm vertical stanchion (height 1.12m)
    const aluMat = new THREE.MeshStandardMaterial({
      color: 0x9aa4b2,
      roughness: 0.34,
      metalness: 0.72,
      envMapIntensity: 0.9,
    });
    const aluOpt = this.getCachedOpticalMaterial('v3_pillar_alu', {
      metalness: 0.74,
      gloss: 0.65,
      darkIR: 0.12,
    });

    const postGeo = new THREE.BoxGeometry(0.044, 0.044, 1.12);
    const postMesh = new THREE.Mesh(postGeo, aluMat);
    postMesh.position.set(0, 0, 0.56);
    postMesh.castShadow = !this.isSoftwareGl;
    postMesh.receiveShadow = !this.isSoftwareGl;
    group.add(postMesh);
    this.meshBindings.push({ mesh: postMesh, pbrMat: aluMat, opticalMat: aluOpt });

    // Bolted steel base flange + anodized safety collar
    const collarMat = new THREE.MeshStandardMaterial({
      color: 0xd97706,
      roughness: 0.42,
      metalness: 0.25,
      envMapIntensity: 0.7,
    });
    const collarOpt = this.getCachedOpticalMaterial('v3_pillar_collar', {
      metalness: 0.25,
      gloss: 0.45,
      darkIR: 0.20,
    });

    const footGeo = new THREE.BoxGeometry(0.11, 0.11, 0.024);
    const footMesh = new THREE.Mesh(footGeo, collarMat);
    footMesh.position.set(0, 0, 0.012);
    group.add(footMesh);
    this.meshBindings.push({ mesh: footMesh, pbrMat: collarMat, opticalMat: collarOpt });

    const capGeo = new THREE.BoxGeometry(0.056, 0.056, 0.045);
    const capMesh = new THREE.Mesh(capGeo, collarMat);
    capMesh.position.set(0, 0, 0.62);
    group.add(capMesh);
    this.meshBindings.push({ mesh: capMesh, pbrMat: collarMat, opticalMat: collarOpt });

    // Default initial placement on camera-robot sightline
    group.position.set(0.62, 0.08, 0);
    return group;
  }

  private buildFlexCable(): { group: THREE.Group; mesh: THREE.Mesh } {
    const group = new THREE.Group();
    group.name = 'v3_coupled_flex_cable';

    const cableMat = new THREE.MeshStandardMaterial({
      color: 0x262b36,
      roughness: 0.72,
      metalness: 0.08,
      envMapIntensity: 0.45,
    });
    const cableOpt = this.getCachedOpticalMaterial('v3_flex_cable', {
      metalness: 0.08,
      gloss: 0.25,
      darkIR: 0.72,
    });

    const initCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, -0.08, 0.08),
      new THREE.Vector3(0.08, -0.14, 0.34),
      new THREE.Vector3(0.22, -0.14, 0.50),
      new THREE.Vector3(0.32, -0.06, 0.44),
    ]);
    const geo = new THREE.TubeGeometry(initCurve, 24, 0.0155, 8, false);
    const mesh = new THREE.Mesh(geo, cableMat);
    mesh.castShadow = !this.isSoftwareGl;
    mesh.receiveShadow = !this.isSoftwareGl;
    group.add(mesh);
    this.meshBindings.push({ mesh, pbrMat: cableMat, opticalMat: cableOpt });

    return { group, mesh };
  }

  /**
   * Position the static front occlusion stanchion along the camera-to-robot sightline
   * (offset slightly laterally so it visibly slices ~15-20% across the arm in the RGB-D FOV).
   */
  updateFrontPillarForCamera(T_cb: SE3): void {
    const camPos = cameraCenterInBase(T_cb);
    const target = this.engine.spec.cameraTarget;
    const dx = camPos[0] - target[0];
    const dy = camPos[1] - target[1];
    const horizLen = Math.hypot(dx, dy);
    if (horizLen < 0.25) return;

    // Place pillar 46% of the way from robot target toward camera, offset laterally by +0.095 m
    const ux = dx / horizLen;
    const uy = dy / horizLen;
    const latX = -uy;
    const latY = ux;
    const frac = 0.46;
    const px = target[0] + dx * frac + latX * 0.095;
    const py = target[1] + dy * frac + latY * 0.095;
    this.frontPillarGroup.position.set(px, py, 0);
    this.frontPillarGroup.rotation.z = Math.atan2(dy, dx);
    this.frontPillarGroup.updateMatrixWorld(true);
  }

  /**
   * Update the coupled non-rigid flexible conduit/cable geometry from live MuJoCo link positions,
   * injecting realistic elastic slack + non-rigid sinusoidal whipping deflection perpendicular to the arm.
   */
  updateFlexCable(timeSec: number, jointSpeedNorm = 0): void {
    if (!this.flexCableGroup.visible) return;
    const anchors = this.engine.getCableControlPointsBase();
    if (anchors.length < 4) return;

    // Non-rigid elastic whipping wave (coupled with arm motion + independent bending harmonic)
    const amp = 0.055 + 0.055 * Math.min(1.5, jointSpeedNorm);
    const wave1 = Math.sin(2 * Math.PI * 1.65 * timeSec + 0.4) * amp;
    const wave2 = Math.cos(2 * Math.PI * 2.15 * timeSec - 0.7) * (amp * 0.85);
    const sagZ = -0.075 - 0.025 * Math.sin(2 * Math.PI * 1.1 * timeSec);

    const p0 = new THREE.Vector3(anchors[0][0], anchors[0][1], anchors[0][2]);
    const p1 = new THREE.Vector3(
      anchors[1][0] + wave1 * 0.55,
      anchors[1][1] - 0.08 + wave1,
      Math.max(0.05, anchors[1][2] + sagZ * 0.6)
    );
    const p2 = new THREE.Vector3(
      anchors[2][0] + wave2 * 0.65,
      anchors[2][1] - 0.11 + wave2,
      Math.max(0.06, anchors[2][2] + sagZ)
    );
    const p3 = new THREE.Vector3(anchors[3][0], anchors[3][1], anchors[3][2]);

    const curve = new THREE.CatmullRomCurve3([p0, p1, p2, p3]);
    const nextGeo = new THREE.TubeGeometry(curve, 22, 0.0155, 8, false);
    this.flexCableMesh.geometry.dispose();
    this.flexCableMesh.geometry = nextGeo;
  }

  setSceneVisibility(opts: {
    showCustomTool: boolean;
    showDistractor: boolean;
    showFrontPillar?: boolean;
    showFlexCable?: boolean;
  }): void {
    for (let i = 0; i < this.geomObjects.length; i++) {
      const g = this.engine.geoms[i];
      if (g.category === 'custom_tool') {
        this.geomObjects[i].visible = opts.showCustomTool;
      } else if (g.category === 'distractor') {
        this.geomObjects[i].visible = opts.showDistractor;
      }
    }
    if (opts.showFrontPillar !== undefined) {
      this.frontPillarGroup.visible = opts.showFrontPillar;
    }
    if (opts.showFlexCable !== undefined) {
      this.flexCableGroup.visible = opts.showFlexCable;
      if (opts.showFlexCable) {
        this.updateFlexCable(0, 0);
      }
    }
    this.requestRender();
  }

  setFrustumPoses(gt: SE3, stage1: SE3 | null, stage2: SE3 | null): void {
    cvExtrinsicToThreeCameraMatrix(gt, this.frustumGT.matrix);
    this.frustumGT.matrixWorldNeedsUpdate = true;
    this.frustumGT.visible = true;

    if (stage1) {
      cvExtrinsicToThreeCameraMatrix(stage1, this.frustumStage1.matrix);
      this.frustumStage1.matrixWorldNeedsUpdate = true;
      this.frustumStage1.visible = true;
    } else {
      this.frustumStage1.visible = false;
    }

    if (stage2) {
      cvExtrinsicToThreeCameraMatrix(stage2, this.frustumStage2.matrix);
      this.frustumStage2.matrixWorldNeedsUpdate = true;
      this.frustumStage2.visible = true;
    } else {
      this.frustumStage2.visible = false;
    }

    this.requestRender();
  }

  resetViewCamera(): void {
    const [vcx, vcy, vcz] = this.engine.spec.viewCamPos;
    const [vtx, vty, vtz] = this.engine.spec.viewCamTarget;
    this.viewCam.position.set(vcx, vcy, vcz);
    this.controls.target.set(vtx, vty, vtz);
    this.controls.update();
    this.requestRender();
  }

  getIntrinsics(): CameraIntrinsics {
    return computeCameraIntrinsics(this.width, this.height, this.fovDeg);
  }

  /**
   * 3-Pass WebGL2 offscreen capture of virtual RealSense D435i RGB + exact metric Depth
   * with physically grounded material-aware and depth-discontinuity noise degradation:
   *  - Pass 1: Real PBR RGB image
   *  - Pass 2: 24-bit exact linear depth + surface incidence cosine
   *  - Pass 3: Per-pixel material optical G-buffer (metalness, gloss, darkIR) — used ONLY inside sensor noise model
   */
  captureVirtualRGBD(
    T_cb_gt: SE3,
    sensorCfg: SensorConfig,
    rng: SeededRNG
  ): { rgb: Uint8ClampedArray; depth: Float32Array; cleanDepth: Float32Array } {
    const W = this.width;
    const H = this.height;

    cvExtrinsicToThreeCameraMatrix(T_cb_gt, this._tmpMat4);
    this.virtualCam.matrix.copy(this._tmpMat4);
    this.virtualCam.matrixWorld.copy(this._tmpMat4);
    this.virtualCam.matrixWorldInverse.copy(this._tmpMat4).invert();

    const prevHelpersVis = this.helpersGroup.visible;
    const prevFog = this.scene.fog;
    const prevToneMapping = this.renderer.toneMapping;
    const prevClearColor = new THREE.Color();
    const prevClearAlpha = this.renderer.getClearAlpha();
    this.renderer.getClearColor(prevClearColor);

    this.helpersGroup.visible = false;
    this.scene.fog = null;

    // Pass 1: Real RGB capture
    this.renderer.setRenderTarget(this.rgbRT);
    this.renderer.setClearColor(0x0b0f17, 1);
    this.renderer.clear();
    this.renderer.render(this.scene, this.virtualCam);
    this.renderer.readRenderTargetPixels(this.rgbRT, 0, 0, W, H, this.rgbReadBuf);

    // Pass 2: Exact linear 24-bit depth + incidence cosine (A channel)
    this.scene.overrideMaterial = this.depthShaderMat;
    this.renderer.toneMapping = THREE.NoToneMapping;
    this.renderer.setRenderTarget(this.depthRT);
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.clear();
    this.renderer.render(this.scene, this.virtualCam);
    this.renderer.readRenderTargetPixels(this.depthRT, 0, 0, W, H, this.depthReadBuf);
    this.scene.overrideMaterial = null;

    // Pass 3: Per-pixel Material Optical G-buffer (metalness, gloss, darkIR)
    const matStrength = Math.max(0, sensorCfg.materialNoiseStrength ?? 1.0);
    if (matStrength > 0.01) {
      for (let i = 0; i < this.meshBindings.length; i++) {
        this.meshBindings[i].mesh.material = this.meshBindings[i].opticalMat;
      }
      this.renderer.setRenderTarget(this.matRT);
      this.renderer.setClearColor(0x000000, 0);
      this.renderer.clear();
      this.renderer.render(this.scene, this.virtualCam);
      this.renderer.readRenderTargetPixels(this.matRT, 0, 0, W, H, this.matReadBuf);
      for (let i = 0; i < this.meshBindings.length; i++) {
        this.meshBindings[i].mesh.material = this.meshBindings[i].pbrMat;
      }
    }

    // Restore renderer & scene state
    this.renderer.toneMapping = prevToneMapping;
    this.renderer.setRenderTarget(null);
    this.renderer.setClearColor(prevClearColor, prevClearAlpha);
    this.scene.fog = prevFog;
    this.helpersGroup.visible = prevHelpersVis;

    // Flip Y axis (OpenGL bottom-left -> CV image top-left: glRow = H - 1 - v)
    const rgb = new Uint8ClampedArray(W * H * 4);
    const cleanDepth = new Float32Array(W * H);
    const cosIncBuf = new Float32Array(W * H);
    const metalBuf = new Float32Array(W * H);
    const glossBuf = new Float32Array(W * H);
    const darkIRBuf = new Float32Array(W * H);

    for (let v = 0; v < H; v++) {
      const glRow = H - 1 - v;
      const srcRowOff = glRow * W * 4;
      const dstRowOff = v * W * 4;
      const depthRowOff = v * W;
      for (let u = 0; u < W; u++) {
        const sIdx = srcRowOff + u * 4;
        const dIdx = dstRowOff + u * 4;
        rgb[dIdx + 0] = this.rgbReadBuf[sIdx + 0];
        rgb[dIdx + 1] = this.rgbReadBuf[sIdx + 1];
        rgb[dIdx + 2] = this.rgbReadBuf[sIdx + 2];
        rgb[dIdx + 3] = 255;

        const r = this.depthReadBuf[sIdx + 0];
        const g = this.depthReadBuf[sIdx + 1];
        const b = this.depthReadBuf[sIdx + 2];
        const a = this.depthReadBuf[sIdx + 3];
        const packed = (r << 16) | (g << 8) | b;
        cleanDepth[depthRowOff + u] = packed === 0 ? 0 : (packed / 16777215.0) * DEPTH_MAX_RANGE_M;
        cosIncBuf[depthRowOff + u] = a / 255.0;

        if (matStrength > 0.01) {
          metalBuf[depthRowOff + u] = this.matReadBuf[sIdx + 0] / 255.0;
          glossBuf[depthRowOff + u] = this.matReadBuf[sIdx + 1] / 255.0;
          darkIRBuf[depthRowOff + u] = this.matReadBuf[sIdx + 2] / 255.0;
        }
      }
    }

    // Precompute 1-ring & 2-ring depth-discontinuity silhouette map for realistic heavy edge noise
    const edgeStrength = Math.max(0, sensorCfg.edgeNoiseStrength ?? 1.0);
    const edgeRing = new Uint8Array(W * H); // 1 = immediate 1px jump edge, 2 = 2px halo around jump
    const edgeJumpOtherZ = new Float32Array(W * H);

    if (sensorCfg.enableFlyingPixels || edgeStrength > 0.01) {
      for (let v = 1; v < H - 1; v++) {
        const row = v * W;
        for (let u = 1; u < W - 1; u++) {
          const idx = row + u;
          const z0 = cleanDepth[idx];
          if (z0 <= 0.08) continue;
          let maxDiff = 0;
          let otherZ = z0;
          const nbs = [cleanDepth[idx - 1], cleanDepth[idx + 1], cleanDepth[idx - W], cleanDepth[idx + W]];
          for (let k = 0; k < 4; k++) {
            const zn = nbs[k];
            if (zn > 0.08) {
              const diff = Math.abs(zn - z0);
              if (diff > maxDiff) {
                maxDiff = diff;
                otherZ = zn;
              }
            }
          }
          if (maxDiff > 0.055) {
            edgeRing[idx] = 1;
            edgeJumpOtherZ[idx] = otherZ;
          }
        }
      }
      // Mark 2nd-ring silhouette halo (real D435i stereo matching window spills disparity error 2px inward!)
      if (edgeStrength > 0.35) {
        for (let v = 2; v < H - 2; v++) {
          const row = v * W;
          for (let u = 2; u < W - 2; u++) {
            const idx = row + u;
            if (edgeRing[idx] !== 0 || cleanDepth[idx] <= 0.08) continue;
            if (
              edgeRing[idx - 1] === 1 ||
              edgeRing[idx + 1] === 1 ||
              edgeRing[idx - W] === 1 ||
              edgeRing[idx + W] === 1
            ) {
              edgeRing[idx] = 2;
              edgeJumpOtherZ[idx] =
                edgeJumpOtherZ[idx - 1] ||
                edgeJumpOtherZ[idx + 1] ||
                edgeJumpOtherZ[idx - W] ||
                edgeJumpOtherZ[idx + W];
            }
          }
        }
      }
    }

    // Apply physical RealSense D435i degradation model
    const depth = new Float32Array(cleanDepth);
    const cosGrazingThresh = Math.cos((78 * Math.PI) / 180); // ~0.2079
    const sigmaZ0 = sensorCfg.sigmaZ0;
    const outlierRate = Math.max(0, Math.min(0.08, sensorCfg.outlierRate ?? 0.004));

    for (let v = 0; v < H; v++) {
      for (let u = 0; u < W; u++) {
        const idx = v * W + u;
        const z0 = cleanDepth[idx];
        if (z0 <= 0.08) {
          depth[idx] = 0;
          continue;
        }

        const cosInc = cosIncBuf[idx];
        const metal = metalBuf[idx];
        const gloss = glossBuf[idx];
        const darkIR = darkIRBuf[idx];

        // 1. Grazing-angle (>78 deg), metallic specular glint & matte-black IR-absorption dropouts
        if (sensorCfg.enableHoles) {
          if (cosInc < cosGrazingThresh) {
            const dropProb = 0.65 + 0.35 * (1 - cosInc / cosGrazingThresh);
            if (rng.next() < dropProb) {
              depth[idx] = 0;
              continue;
            }
          }

          // Metallic specular glint saturation dropout (strongest near perpendicular incidence on brushed steel/silver)
          if (matStrength > 0.01 && metal > 0.45) {
            const glintFactor = Math.pow(Math.max(0, cosInc), 4.0) * metal * gloss;
            const metalHoleProb = Math.min(0.32, glintFactor * 0.11 * matStrength);
            if (rng.next() < metalHoleProb) {
              depth[idx] = 0;
              continue;
            }
          }

          // Low-albedo matte-black active-IR absorption dropout (ALOHA / OpenArm carbon links)
          if (matStrength > 0.01 && darkIR > 0.55) {
            const darkHoleProb = Math.min(
              0.30,
              (darkIR - 0.45) * (0.040 + 0.070 * (1 - cosInc)) * matStrength
            );
            if (rng.next() < darkHoleProb) {
              depth[idx] = 0;
              continue;
            }
          }

          // Baseline subtle specular dropout
          const lum =
            (rgb[idx * 4] * 0.299 + rgb[idx * 4 + 1] * 0.587 + rgb[idx * 4 + 2] * 0.114) / 255;
          const specProb = lum > 0.72 ? 0.014 : 0.003;
          if (rng.next() < specProb) {
            depth[idx] = 0;
            continue;
          }
        }

        let zVal = z0;

        // 2. Depth-discontinuity flying pixels + heavy-tailed 1px/2px edge disparity noise
        const ring = edgeRing[idx];
        if (ring === 1) {
          if (sensorCfg.enableFlyingPixels) {
            const fpProb = Math.min(0.92, 0.52 + 0.16 * edgeStrength);
            if (rng.next() < fpProb) {
              const alpha = rng.uniform(0.14, 0.86);
              zVal = z0 * (1 - alpha) + edgeJumpOtherZ[idx] * alpha;
            }
          }
          if (edgeStrength > 0.05) {
            // Heavy-tailed silhouette edge jitter (12..36 mm scaled by edgeStrength)
            const edgeSigma = (0.012 + 0.010 * edgeStrength) * z0;
            zVal += rng.gaussian(0, edgeSigma);
          }
        } else if (ring === 2 && edgeStrength > 0.35) {
          // 2nd-ring stereo matching window bleed & elevated silhouette jitter
          if (sensorCfg.enableFlyingPixels && rng.next() < Math.min(0.45, 0.12 * edgeStrength)) {
            const alpha = rng.uniform(0.08, 0.38);
            zVal = z0 * (1 - alpha) + edgeJumpOtherZ[idx] * alpha;
          }
          const haloSigma = 0.0065 * edgeStrength * z0;
          zVal += rng.gaussian(0, haloSigma);
        }

        // 3. Material-dependent optical noise & active-IR stereo multipath drift
        let matSigmaMult = 1.0;
        if (matStrength > 0.01) {
          // Brushed metallic specular multipath drift (+2..8 mm) + strong disparity variance amplification
          if (metal > 0.45) {
            const specStrength = metal * gloss * matStrength;
            const multipathBias = specStrength * (0.0016 + 0.0030 * (1 - cosInc)) * z0;
            zVal += multipathBias;
            matSigmaMult += 1.35 * specStrength;
          }
          // Matte-black IR absorption lowers active stereo dot-pattern SNR (1.6x..2.8x noise)
          if (darkIR > 0.45) {
            const irLoss = (darkIR - 0.35) * matStrength;
            matSigmaMult += 1.60 * irLoss;
          }
        }

        // 4. Distance-squared stereo Gaussian noise: sigma(z) = sigmaZ0 * z^2 * matSigmaMult
        if (sigmaZ0 > 0) {
          const sigma = sigmaZ0 * zVal * zVal * matSigmaMult;
          zVal += rng.gaussian(0, sigma);
        }

        // 5. Random stereo correspondence mismatch outlier spikes (+-6..20 cm)
        if (outlierRate > 0 && rng.next() < outlierRate) {
          const sign = rng.next() < 0.5 ? -1 : 1;
          zVal += sign * rng.uniform(0.06, 0.20);
        }

        depth[idx] = zVal > 0.08 && zVal < DEPTH_MAX_RANGE_M ? zVal : 0;
      }
    }

    return { rgb, depth, cleanDepth };
  }

  private afterDraw(cb: () => void): void {
    const gl = this.renderer.getContext() as WebGL2RenderingContext;
    const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
    const poll = () => {
      if (this.disposed) {
        if (sync) gl.deleteSync(sync);
        return;
      }
      const state = sync ? gl.getSyncParameter(sync, gl.SYNC_STATUS) : gl.SIGNALED;
      if (state === gl.UNSIGNALED) {
        this.frameId = window.setTimeout(poll, 5);
      } else {
        if (sync) gl.deleteSync(sync);
        this.frameId = window.setTimeout(cb, 50);
      }
    };
    this.frameId = window.setTimeout(poll, 5);
  }

  requestRender(): void {
    if (this.disposed) return;
    if (this.renderBusy) {
      this.renderDirty = true;
      return;
    }
    this.renderBusy = true;
    this.frameId = requestAnimationFrame(() => {
      if (this.disposed) return;
      this.renderDirty = false;
      this.controls.update();
      this.renderer.render(this.scene, this.viewCam);
      const done = () => {
        this.renderBusy = false;
        if (this.renderDirty && !this.disposed) {
          this.requestRender();
        }
      };
      if (this.isSoftwareGl) {
        this.afterDraw(done);
      } else {
        done();
      }
    });
  }

  /** Immediate synchronous render for stepping live during calibration */
  renderNow(): void {
    if (this.disposed) return;
    this.controls.update();
    this.renderer.render(this.scene, this.viewCam);
  }

  dispose(): void {
    this.disposed = true;
    cancelAnimationFrame(this.frameId);
    clearTimeout(this.frameId);
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.rgbRT.dispose();
    this.depthRT.dispose();
    this.matRT.dispose();
    this.depthShaderMat.dispose();
    for (const optMat of this.opticalMatCache.values()) {
      optMat.dispose();
    }
    this.opticalMatCache.clear();
    for (const obj of this.geomObjects) {
      if (obj instanceof THREE.Mesh) {
        obj.geometry.dispose();
        (obj.material as THREE.Material).dispose();
      } else if (obj instanceof THREE.Group) {
        obj.traverse((child) => {
          if (child instanceof THREE.Mesh) {
            child.geometry.dispose();
          }
        });
      }
    }
    this.frontPillarGroup.traverse((child) => {
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose();
        (child.material as THREE.Material).dispose();
      }
    });
    this.flexCableMesh.geometry.dispose();
    (this.flexCableMesh.material as THREE.Material).dispose();
    for (const mat of Object.values(this.robotMaterials)) {
      mat.dispose();
    }
    this.renderer.dispose();
  }
}
