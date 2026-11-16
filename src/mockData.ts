/**
 * Mock Data & Deterministic Demo Simulation Module
 * ------------------------------------------------
 * Isolates all mock datasets, synthetic image rendering, mock model
 * predictions, and mock visual effects so they can be cleanly replaced
 * with real inference endpoints or PyTorch backend calls later.
 *
 * DEMO MODE ONLY: No real models, gradients, or defenses are executed.
 */

export type AttackMethod = 'FGSM' | 'PGD';
export type DefenseMethod =
  | 'None'
  | 'PGD-AT'
  | 'Gaussian Blur'
  | 'Bit-depth Reduction';

export interface ModelSpec {
  id: string;
  name: string;
  dataset: string;
  architecture: string;
  inputSize: string;
  classes: string[];
  cleanAccuracy: string;
}

export interface SampleSpec {
  id: string;
  label: string;
  dataset: string;
  modelId: string;
  trueClass: string;
  targetMisclass: string;
  secondaryMisclass: string;
  baseConfidence: number;
}

export interface PredictionState {
  predictedClass: string;
  confidence: number;
  isCorrect: boolean | null;
  statusTag: string;
  metadataLines: string[];
}

export interface MockAnalysisOutput {
  originalPred: PredictionState;
  attackedPred: PredictionState;
  defendedPred: PredictionState;
  summaryHeadline: string;
  summaryExplanation: string;
}

export interface AnalysisConfig {
  modelId: string;
  attackMethod: AttackMethod;
  epsilon: number;
  defenseMethod: DefenseMethod;
  sampleId: string;
  uploadedImageName: string | null;
  uploadedMeanSeed: number | null;
  pgdSteps: number;
  pgdAlpha: number;
  pgdRestarts: number;
  blurRadius: number;
  bitDepth: number;
  visualAmp: number;
}

export const MODELS: Record<string, ModelSpec> = {
  mnist_cnn: {
    id: 'mnist_cnn',
    name: 'MNIST / SimpleCNN',
    dataset: 'MNIST (28x28 Grayscale)',
    architecture: '2-Layer ConvNet (Mock)',
    inputSize: '28 x 28',
    classes: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
    cleanAccuracy: '99.1% (Mock)',
  },
  cifar10_resnet18: {
    id: 'cifar10_resnet18',
    name: 'CIFAR-10 / ResNet-18',
    dataset: 'CIFAR-10 (32x32 RGB)',
    architecture: 'ResNet-18 (Mock)',
    inputSize: '32 x 32',
    classes: [
      'Airplane',
      'Automobile',
      'Bird',
      'Cat',
      'Deer',
      'Dog',
      'Frog',
      'Horse',
      'Ship',
      'Truck',
    ],
    cleanAccuracy: '94.6% (Mock)',
  },
  gtsrb_resnet18: {
    id: 'gtsrb_resnet18',
    name: 'GTSRB / ResNet-18',
    dataset: 'GTSRB Traffic Signs',
    architecture: 'ResNet-18 (Mock)',
    inputSize: '32 x 32',
    classes: [
      'Stop Sign',
      'Speed Limit 50',
      'Yield',
      'No Entry',
      'Turn Right Ahead',
      'Priority Road',
    ],
    cleanAccuracy: '97.8% (Mock)',
  },
};

export const SAMPLES: SampleSpec[] = [
  {
    id: 'sample_cifar_plane',
    label: 'CIFAR-10 · Airplane',
    dataset: 'CIFAR-10',
    modelId: 'cifar10_resnet18',
    trueClass: 'Airplane',
    targetMisclass: 'Bird',
    secondaryMisclass: 'Ship',
    baseConfidence: 0.964,
  },
  {
    id: 'sample_gtsrb_stop',
    label: 'GTSRB · Stop Sign',
    dataset: 'GTSRB',
    modelId: 'gtsrb_resnet18',
    trueClass: 'Stop Sign',
    targetMisclass: 'Speed Limit 50',
    secondaryMisclass: 'Yield',
    baseConfidence: 0.982,
  },
  {
    id: 'sample_mnist_seven',
    label: 'MNIST · Digit 7',
    dataset: 'MNIST',
    modelId: 'mnist_cnn',
    trueClass: '7',
    targetMisclass: '1',
    secondaryMisclass: '9',
    baseConfidence: 0.993,
  },
  {
    id: 'sample_cifar_frog',
    label: 'CIFAR-10 · Frog',
    dataset: 'CIFAR-10',
    modelId: 'cifar10_resnet18',
    trueClass: 'Frog',
    targetMisclass: 'Cat',
    secondaryMisclass: 'Deer',
    baseConfidence: 0.938,
  },
];

export const ATTACK_METHODS: AttackMethod[] = ['FGSM', 'PGD'];
export const DEFENSE_METHODS: DefenseMethod[] = [
  'None',
  'PGD-AT',
  'Gaussian Blur',
  'Bit-depth Reduction',
];

export function computeMockMetrics(config: AnalysisConfig): MockAnalysisOutput {
  const model = MODELS[config.modelId] ?? MODELS.cifar10_resnet18;
  const sample =
    SAMPLES.find((s) => s.id === config.sampleId) ?? SAMPLES[0];

  let trueClass = sample.trueClass;
  let targetMisclass = sample.targetMisclass;
  let secondaryMisclass = sample.secondaryMisclass;
  let baseConf = sample.baseConfidence;
  let hasGroundTruth = true;

  if (config.uploadedImageName && config.uploadedMeanSeed !== null) {
    const idx = Math.abs(Math.round(config.uploadedMeanSeed)) % model.classes.length;
    trueClass = model.classes[idx];
    targetMisclass = model.classes[(idx + 2) % model.classes.length];
    secondaryMisclass = model.classes[(idx + 5) % model.classes.length];
    baseConf = 0.892 + (Math.round(config.uploadedMeanSeed * 13) % 75) / 1000;
    hasGroundTruth = false;
  } else if (!model.classes.includes(trueClass)) {
    const hash = sample.id
      .split('')
      .reduce((acc, ch) => acc + ch.charCodeAt(0), 0);
    const idx = hash % model.classes.length;
    trueClass = model.classes[idx];
    targetMisclass = model.classes[(idx + 1) % model.classes.length];
    secondaryMisclass = model.classes[(idx + 3) % model.classes.length];
    baseConf = 0.845;
    hasGroundTruth = false;
  }

  const originalPred: PredictionState = {
    predictedClass: trueClass,
    confidence: Number(baseConf.toFixed(3)),
    isCorrect: hasGroundTruth ? true : null,
    statusTag: 'ẢNH SẠCH (MOCK)',
    metadataLines: [
      `Model: ${model.name}`,
      `Dataset: ${model.dataset}`,
      'Nhiễu: ε = 0.00',
    ],
  };

  // Attack evaluation (PGD is stronger than FGSM at equivalent epsilon)
  const attackStrength =
    config.attackMethod === 'PGD'
      ? config.epsilon * (1.35 + Math.min(0.25, config.pgdSteps * 0.01))
      : config.epsilon;
  const flipThreshold = config.attackMethod === 'PGD' ? 0.05 : 0.08;

  let atkClass = trueClass;
  let atkConf = baseConf;
  let atkStatus = 'KHÔNG CÓ NHIỄU';

  if (config.epsilon <= 0.005) {
    atkClass = trueClass;
    atkConf = baseConf;
    atkStatus = 'KHÔNG CÓ NHIỄU';
  } else if (attackStrength < flipThreshold) {
    atkClass = trueClass;
    atkConf = Math.max(0.51, baseConf - attackStrength * 3.8);
    atkStatus = 'ĐỘ TIN CẬY SUY GIẢM (MOCK)';
  } else if (attackStrength < 0.18) {
    atkClass = targetMisclass;
    atkConf = Math.min(0.94, 0.56 + attackStrength * 2.1);
    atkStatus = 'TẤN CÔNG THÀNH CÔNG (MOCK)';
  } else {
    atkClass = secondaryMisclass;
    atkConf = Math.min(0.985, 0.72 + attackStrength * 0.85);
    atkStatus = 'ĐÁNH LỪA ĐỘ TIN CẬY CAO (MOCK)';
  }

  const atkMeta = [
    `Attack: ${config.attackMethod}`,
    `Epsilon (ε): ${config.epsilon.toFixed(2)}`,
    config.attackMethod === 'PGD'
      ? `Steps: ${config.pgdSteps} · α=${config.pgdAlpha.toFixed(3)} · Restarts: ${config.pgdRestarts}`
      : 'Single-step L∞ gradient sign (Mock)',
  ];

  const attackedPred: PredictionState = {
    predictedClass: atkClass,
    confidence: Number(atkConf.toFixed(3)),
    isCorrect: hasGroundTruth ? atkClass === trueClass : null,
    statusTag: atkStatus,
    metadataLines: atkMeta,
  };

  // Defense evaluation
  let defClass = atkClass;
  let defConf = atkConf;
  let defStatus = 'KHÔNG PHÒNG THỦ (PASS-THROUGH)';
  let defMeta: string[] = [
    'Defense: None',
    'Đưa trực tiếp ảnh bị tấn công vào model chuẩn',
  ];

  if (config.defenseMethod === 'PGD-AT') {
    if (attackStrength < 0.24) {
      defClass = trueClass;
      defConf = Math.max(0.68, baseConf - 0.06 - attackStrength * 0.65);
      defStatus = 'MODEL BỀN VỮNG ĐÃ KHÔI PHỤC (MOCK)';
    } else {
      defClass = targetMisclass;
      defConf = Math.min(0.79, 0.48 + attackStrength * 0.7);
      defStatus = 'VƯỢT NGƯỠNG Ở ε CAO (MOCK)';
    }
    defMeta = [
      'Defense: PGD-AT (Adversarial Training)',
      'Đánh giá trên checkpoint bền vững (Mock)',
    ];
  } else if (config.defenseMethod === 'Gaussian Blur') {
    if (
      config.epsilon <= 0.12 &&
      config.blurRadius >= 0.8 &&
      config.blurRadius <= 2.6
    ) {
      defClass = trueClass;
      defConf = Math.max(0.58, baseConf - 0.14 - config.epsilon * 1.1);
      defStatus = 'ĐÃ LÀM MỊN NHIỄU (MOCK)';
    } else if (config.epsilon <= 0.05) {
      defClass = trueClass;
      defConf = Math.max(0.55, baseConf - 0.12);
      defStatus = 'ĐÃ KHÔI PHỤC (MOCK)';
    } else {
      defClass =
        config.blurRadius > 2.6 ? secondaryMisclass : targetMisclass;
      defConf = Math.min(0.82, 0.44 + config.epsilon * 1.1);
      defStatus = 'KHÔI PHỤC MỘT PHẦN / MỜ CHI TIẾT (MOCK)';
    }
    defMeta = [
      'Defense: Gaussian Blur (Input Transform)',
      `Kernel σ = ${config.blurRadius.toFixed(1)}px`,
    ];
  } else if (config.defenseMethod === 'Bit-depth Reduction') {
    if (
      config.epsilon <= 0.14 &&
      config.bitDepth >= 2 &&
      config.bitDepth <= 4
    ) {
      defClass = trueClass;
      defConf = Math.max(0.61, baseConf - 0.11 - config.epsilon * 0.95);
      defStatus = 'LƯỢNG TỬ HÓA ĐÃ KHÔI PHỤC (MOCK)';
    } else {
      defClass = targetMisclass;
      defConf = Math.min(0.84, 0.49 + config.epsilon * 1.05);
      defStatus = 'LƯỢNG TỬ HÓA CHƯA ĐỦ MẠNH (MOCK)';
    }
    defMeta = [
      'Defense: Bit-depth Reduction (Input Transform)',
      `Độ phân giải: ${config.bitDepth}-bit (${1 << config.bitDepth} mức/kênh)`,
    ];
  }

  const defendedPred: PredictionState = {
    predictedClass: defClass,
    confidence: Number(defConf.toFixed(3)),
    isCorrect: hasGroundTruth ? defClass === trueClass : null,
    statusTag: defStatus,
    metadataLines: defMeta,
  };

  const fooled = atkClass !== trueClass;
  const recovered = defClass === trueClass;
  let summaryHeadline = '';
  let summaryExplanation = '';

  if (config.epsilon <= 0.005) {
    summaryHeadline = 'Kiểm tra gốc (Baseline): Chưa thêm nhiễu đối nghịch';
    summaryExplanation = `Với ε = 0.00, chưa có nhiễu đối nghịch nào được cộng vào ảnh "${trueClass}". Cả hai khung Ảnh gốc và Bị tấn công đều dự đoán "${trueClass}" (${(originalPred.confidence * 100).toFixed(1)}%).`;
  } else if (fooled && config.defenseMethod === 'None') {
    summaryHeadline = `${config.attackMethod} đã đánh lừa ${model.name}: ${trueClass} → ${atkClass}`;
    summaryExplanation = `Tại mức ε = ${config.epsilon.toFixed(2)}, nhiễu giả lập ${config.attackMethod} làm model đổi dự đoán từ "${trueClass}" (${(originalPred.confidence * 100).toFixed(1)}%) sang "${atkClass}" (${(attackedPred.confidence * 100).toFixed(1)}%). Do Defense đang chọn None, khung Đã phòng thủ giữ nguyên kết quả bị tấn công.`;
  } else if (fooled && recovered) {
    summaryHeadline = `${config.defenseMethod} đã khôi phục dự đoán "${trueClass}" sau tấn công ${config.attackMethod}`;
    summaryExplanation = `Tấn công giả lập ${config.attackMethod} (ε = ${config.epsilon.toFixed(2)}) khiến model chuẩn dự đoán nhầm sang "${atkClass}" (${(attackedPred.confidence * 100).toFixed(1)}%). Khi áp dụng ${config.defenseMethod}, dự đoán được khôi phục về "${defClass}" với độ tin cậy ${(defendedPred.confidence * 100).toFixed(1)}%.`;
  } else if (fooled && !recovered) {
    summaryHeadline = `${config.defenseMethod} chưa khôi phục được lớp đúng tại ε = ${config.epsilon.toFixed(2)}`;
    summaryExplanation = `Dưới nhiễu giả lập ${config.attackMethod} mạnh (ε = ${config.epsilon.toFixed(2)}), dự đoán bị lệch từ "${trueClass}" sang "${atkClass}". ${config.defenseMethod} thay đổi độ tin cậy thành ${(defendedPred.confidence * 100).toFixed(1)}% nhưng chưa khôi phục lại được nhãn "${trueClass}".`;
  } else {
    summaryHeadline = `Model giữ vững dự đoán "${trueClass}" trước ${config.attackMethod} nhẹ (ε = ${config.epsilon.toFixed(2)})`;
    summaryExplanation = `Tại ε = ${config.epsilon.toFixed(2)}, nhiễu làm giảm độ tin cậy từ ${(originalPred.confidence * 100).toFixed(1)}% xuống ${(attackedPred.confidence * 100).toFixed(1)}% nhưng chưa đủ để lật nhãn dự đoán. Áp dụng ${config.defenseMethod} tiếp tục giữ kết quả "${defClass}" (${(defendedPred.confidence * 100).toFixed(1)}%).`;
  }

  return {
    originalPred,
    attackedPred,
    defendedPred,
    summaryHeadline,
    summaryExplanation,
  };
}

/**
 * Draw crisp synthetic sample images onto an HTML5 Canvas (360x360).
 */
export function drawMockSampleOnCanvas(
  canvas: HTMLCanvasElement,
  sampleId: string
): void {
  const size = 360;
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, size, size);

  if (sampleId === 'sample_mnist_seven') {
    ctx.fillStyle = '#121216';
    ctx.fillRect(0, 0, size, size);
    ctx.strokeStyle = '#F8F8FA';
    ctx.lineWidth = 34;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(86, 92);
    ctx.lineTo(272, 92);
    ctx.lineTo(148, 294);
    ctx.stroke();

    ctx.lineWidth = 18;
    ctx.beginPath();
    ctx.moveTo(156, 192);
    ctx.lineTo(240, 192);
    ctx.stroke();
    return;
  }

  if (sampleId === 'sample_gtsrb_stop') {
    ctx.fillStyle = '#8AC4E8';
    ctx.fillRect(0, 0, size, size * 0.65);
    ctx.fillStyle = '#5C6068';
    ctx.fillRect(0, size * 0.65, size, size * 0.35);

    ctx.fillStyle = '#3C3E44';
    ctx.fillRect(size / 2 - 10, size * 0.65, 20, size * 0.35);

    const cx = size / 2;
    const cy = size * 0.43;
    const drawOctagon = (r: number, fill: string, stroke?: string) => {
      ctx.beginPath();
      for (let i = 0; i < 8; i++) {
        const a = Math.PI / 8 + (i * Math.PI) / 4;
        const x = cx + r * Math.cos(a);
        const y = cy + r * Math.sin(a);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = fill;
      ctx.fill();
      if (stroke) {
        ctx.strokeStyle = stroke;
        ctx.lineWidth = 7;
        ctx.stroke();
      }
    };
    drawOctagon(112, '#FFFFFF');
    drawOctagon(100, '#D62626');

    ctx.fillStyle = '#FFFFFF';
    ctx.font = '900 54px Archivo, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('STOP', cx, cy + 3);
    return;
  }

  if (sampleId === 'sample_cifar_frog') {
    ctx.fillStyle = '#3E7642';
    ctx.fillRect(0, 0, size, size);
    ctx.fillStyle = '#2A5630';
    ctx.beginPath();
    ctx.ellipse(180, 265, 145, 72, 0, 0, Math.PI * 2);
    ctx.fill();

    // Frog body
    ctx.fillStyle = '#6CB248';
    ctx.strokeStyle = '#192D16';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.ellipse(180, 205, 86, 68, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Eyes
    const drawEye = (ex: number) => {
      ctx.fillStyle = '#7DC452';
      ctx.beginPath();
      ctx.arc(ex, 125, 27, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = '#111114';
      ctx.beginPath();
      ctx.arc(ex, 125, 10, 0, Math.PI * 2);
      ctx.fill();
    };
    drawEye(138);
    drawEye(222);

    ctx.beginPath();
    ctx.arc(180, 195, 44, 0.15 * Math.PI, 0.85 * Math.PI);
    ctx.stroke();
    return;
  }

  // Default: CIFAR-10 Airplane
  ctx.fillStyle = '#60A8E2';
  ctx.fillRect(0, 0, size, size * 0.72);
  ctx.fillStyle = '#E0EAF4';
  ctx.fillRect(0, size * 0.72, size, size * 0.28);

  // Clouds
  ctx.fillStyle = '#F8FAFC';
  ctx.beginPath();
  ctx.ellipse(82, 76, 48, 22, 0, 0, Math.PI * 2);
  ctx.ellipse(278, 66, 46, 20, 0, 0, Math.PI * 2);
  ctx.fill();

  // Fuselage + Wings + Tail
  ctx.strokeStyle = '#181C24';
  ctx.lineWidth = 4;

  ctx.fillStyle = '#D6DCE6';
  ctx.beginPath();
  ctx.moveTo(152, 182);
  ctx.lineTo(206, 98);
  ctx.lineTo(228, 98);
  ctx.lineTo(192, 180);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#CAD2DE';
  ctx.beginPath();
  ctx.moveTo(156, 196);
  ctx.lineTo(208, 268);
  ctx.lineTo(228, 268);
  ctx.lineTo(192, 194);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#DC3A30';
  ctx.beginPath();
  ctx.moveTo(74, 192);
  ctx.lineTo(56, 142);
  ctx.lineTo(82, 142);
  ctx.lineTo(104, 186);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = '#F2F4F8';
  ctx.beginPath();
  ctx.moveTo(64, 192);
  ctx.lineTo(288, 166);
  ctx.lineTo(308, 178);
  ctx.lineTo(278, 196);
  ctx.lineTo(78, 208);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
}

/**
 * Render Original, Attacked, and Defended preview data URLs on offscreen canvases.
 */
export function renderMockPipelineDataUrls(params: {
  sourceImage: HTMLImageElement | null;
  sampleId: string;
  attackMethod: AttackMethod;
  epsilon: number;
  defenseMethod: DefenseMethod;
  blurRadius: number;
  bitDepth: number;
  visualAmp: number;
}): {
  originalUrl: string;
  attackedUrl: string;
  defendedUrl: string;
  meanSeed: number;
} {
  const size = 360;
  const origCanvas = document.createElement('canvas');
  origCanvas.width = size;
  origCanvas.height = size;
  const origCtx = origCanvas.getContext('2d')!;

  if (params.sourceImage) {
    // Center-crop cover draw into 360x360 square
    const img = params.sourceImage;
    const scale = Math.max(size / img.width, size / img.height);
    const dw = img.width * scale;
    const dh = img.height * scale;
    origCtx.drawImage(img, (size - dw) / 2, (size - dh) / 2, dw, dh);
  } else {
    drawMockSampleOnCanvas(origCanvas, params.sampleId);
  }

  const origData = origCtx.getImageData(0, 0, size, size);
  const pixels = origData.data;

  // Deterministic brightness seed for uploaded images
  let sum = 0;
  for (let i = 0; i < pixels.length; i += 16) {
    sum += pixels[i] + pixels[i + 1] + pixels[i + 2];
  }
  const meanSeed = sum / (pixels.length / 16) / 3;

  // 2. Attacked Canvas
  const atkCanvas = document.createElement('canvas');
  atkCanvas.width = size;
  atkCanvas.height = size;
  const atkCtx = atkCanvas.getContext('2d')!;
  const atkData = atkCtx.createImageData(size, size);
  const atkPixels = atkData.data;

  const visualScale =
    params.epsilon <= 0.001
      ? 0
      : Math.min(0.65, params.epsilon * params.visualAmp * 1.15) * 255;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const idx = (y * size + x) * 4;
      let nr = 0;
      let ng = 0;
      let nb = 0;

      if (visualScale > 0) {
        if (params.attackMethod === 'FGSM') {
          const s = Math.sign(Math.sin(x * 0.65) + Math.cos(y * 0.65));
          nr = s;
          ng = -s * 0.7;
          nb = s * 0.85;
        } else {
          const p1 = Math.sin(x * 0.42 + y * 0.31);
          const p2 = Math.cos(x * 0.83 - y * 0.57);
          const p3 = Math.sin((x + y) * 0.25);
          nr = Math.max(-1, Math.min(1, p1 + 0.5 * p2));
          ng = Math.max(-1, Math.min(1, p2 - 0.4 * p3));
          nb = Math.max(-1, Math.min(1, p3 + 0.5 * p1));
        }
      }

      atkPixels[idx] = Math.max(0, Math.min(255, pixels[idx] + nr * visualScale));
      atkPixels[idx + 1] = Math.max(
        0,
        Math.min(255, pixels[idx + 1] + ng * visualScale)
      );
      atkPixels[idx + 2] = Math.max(
        0,
        Math.min(255, pixels[idx + 2] + nb * visualScale)
      );
      atkPixels[idx + 3] = 255;
    }
  }
  atkCtx.putImageData(atkData, 0, 0);

  // 3. Defended Canvas
  const defCanvas = document.createElement('canvas');
  defCanvas.width = size;
  defCanvas.height = size;
  const defCtx = defCanvas.getContext('2d')!;

  if (params.defenseMethod === 'None') {
    defCtx.drawImage(atkCanvas, 0, 0);
  } else if (params.defenseMethod === 'PGD-AT') {
    // Mild visual smoothing to distinguish robust evaluation preview
    defCtx.filter = 'blur(0.7px) contrast(1.04)';
    defCtx.drawImage(atkCanvas, 0, 0);
    defCtx.filter = 'none';
  } else if (params.defenseMethod === 'Gaussian Blur') {
    defCtx.filter = `blur(${params.blurRadius.toFixed(1)}px)`;
    defCtx.drawImage(atkCanvas, 0, 0);
    defCtx.filter = 'none';
  } else {
    // Bit-depth Reduction
    const levels = Math.max(2, (1 << params.bitDepth) - 1);
    const defData = defCtx.createImageData(size, size);
    const dp = defData.data;
    for (let i = 0; i < atkPixels.length; i += 4) {
      dp[i] = Math.round((atkPixels[i] / 255) * levels) * (255 / levels);
      dp[i + 1] =
        Math.round((atkPixels[i + 1] / 255) * levels) * (255 / levels);
      dp[i + 2] =
        Math.round((atkPixels[i + 2] / 255) * levels) * (255 / levels);
      dp[i + 3] = 255;
    }
    defCtx.putImageData(defData, 0, 0);
  }

  return {
    originalUrl: origCanvas.toDataURL('image/png'),
    attackedUrl: atkCanvas.toDataURL('image/png'),
    defendedUrl: defCanvas.toDataURL('image/png'),
    meanSeed,
  };
}
