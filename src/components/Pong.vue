<template>
  <canvas ref="canvasRef" />
</template>

<script setup lang="ts">
import { Ref, onMounted, ref, watch } from 'vue';
import * as tf from '@tensorflow/tfjs';

const fps = ref(30);
const canvasRef = ref<HTMLCanvasElement | null>(null);
const gaming = ref<boolean>(false);
const playerScore = ref(0);
const computerScore = ref(0);

const player = ref({
  x: 0,
  y: 0,
  width: 8,
  height: 100,
  color: 'white',
});

const computer = ref({
  x: 0,
  y: 0,
  width: 8,
  height: 100,
  color: 'white',
});

const ball = ref({
  x: 0,
  y: 0,
  width: 10,
  height: 10,
  color: 'white',
  dx: 8,
  dy: 8,
});

const possession = ref<'left' | 'right' | null>(null);

const logging = ref(false);

const training = ref(false);

const log = ref<any[]>([]);

const model = tf.sequential();

model.add(tf.layers.dense({ units: 1, inputShape: [1], useBias: true }));
model.add(tf.layers.dense({ units: 1, useBias: true }));

model.compile({
  loss: 'meanSquaredError',
  optimizer: tf.train.adam(0.06),
});

const clearCanvas = (): void => {
  if (!canvasRef.value) return;
  const ctx = canvasRef.value.getContext('2d');
  if (!ctx) return;

  ctx.clearRect(0, 0, canvasRef.value.width, canvasRef.value.height);
};

const setCanvasBackground = (): void => {
  if (!canvasRef.value) return;
  const ctx = canvasRef.value.getContext('2d');
  if (!ctx) return;

  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvasRef.value.width, canvasRef.value.height);
};

const renderPaddle = (paddleRef: Ref<typeof player.value | typeof computer.value>, position = 'left'): void => {
  if (!canvasRef.value) return;
  const ctx = canvasRef.value.getContext('2d');
  if (!ctx) return;

  paddleRef.value.x = position === 'left' ? 0 : canvasRef.value.width - paddleRef.value.width - 0;
  ctx.fillStyle = paddleRef.value.color;
  ctx.fillRect(paddleRef.value.x, paddleRef.value.y, paddleRef.value.width, paddleRef.value.height);
};

const renderBall = (): void => {
  if (!canvasRef.value) return;
  const ctx = canvasRef.value.getContext('2d');
  if (!ctx) return;

  ctx.fillStyle = ball.value.color;
  ctx.fillRect(ball.value.x, ball.value.y, ball.value.width, ball.value.height);
};

const renderScoreboard = (): void => {
  if (!canvasRef.value) return;
  const ctx = canvasRef.value.getContext('2d');
  if (!ctx) return;

  ctx.font = '30px Arial';
  ctx.fillStyle = 'white';
  ctx.fillText(playerScore.value.toString(), canvasRef.value.width / 2 - 50, 50);
  ctx.fillText(computerScore.value.toString(), canvasRef.value.width / 2 + 50, 50);
};

const resetScoreboard = (): void => {
  playerScore.value = 0;
  computerScore.value = 0;
};

const updateBallPosition = (): void => {
  ball.value.x += ball.value.dx;
  ball.value.y += ball.value.dy;
};

const updatePossession = (): void => {
  if (!canvasRef.value) return;

  if (ball.value.x < canvasRef.value!.width / 2) {
    possession.value = 'left';
  } else {
    possession.value = 'right';
  }
};

const collideWithVerticalBorders = (): void => {
  if (!canvasRef.value) return;

  if (ball.value.y <= 0 || ball.value.y + ball.value.height >= canvasRef.value.height) {
    ball.value.dy = -ball.value.dy;
  }
};

const collideWithPaddles = (): void => {
  if (
    (ball.value.x - player.value.width <= player.value.x &&
      ball.value.y + ball.value.height >= player.value.y &&
      ball.value.y <= player.value.y + player.value.height) ||
    (ball.value.x + ball.value.width >= computer.value.x &&
      ball.value.y + ball.value.height >= computer.value.y &&
      ball.value.y <= computer.value.y + computer.value.height)
  ) {
    ball.value.dx = -ball.value.dx;
  }
};

const collideWithHorizontalBorders = (): void => {
  if (!canvasRef.value) return;

  if (ball.value.x <= 0 || ball.value.x + ball.value.width >= canvasRef.value.width) {
    if (possession.value === 'right') {
      // Aumenta el marcador del jugador
      playerScore.value++;
    } else {
      // Aumenta el marcador de la computadora
      computerScore.value++;
    }

    // Resetea la posición de la bola al centro
    centerBall();

    // Invierte la dirección de la bola
    ball.value.dx = -ball.value.dx;
  }
};

const ballCollision = (): void => {
  collideWithVerticalBorders();
  collideWithPaddles();
  collideWithHorizontalBorders();
};

const centerBall = (): void => {
  if (!canvasRef.value) return;

  ball.value.x = canvasRef.value.width / 2;
  ball.value.y = canvasRef.value.height / 2;
};

const centerPaddles = (): void => {
  const y = canvasRef.value ? canvasRef.value.height / 2 - player.value.height / 2 : 0;
  player.value.y = y;
  computer.value.y = y;
};

const createLoop = (): void => {
  setInterval(() => {
    if (!gaming.value) return;

    computerMove();

    render();

    if (!logging.value) return;

    registerTrainingData();
  }, 1000 / fps.value);
};

const getBallDistanceFromPlayer = (): number => {
  return Math.abs(ball.value.x - player.value.x);
};

const getBallDistanceFromComputer = (): number => {
  return Math.abs(ball.value.x - computer.value.x);
};

let trainingTimer: any = null;

const start = (): void => {
  gaming.value = true;

  trainingTimer = setInterval(() => {
    if (!training.value) {
      train(getTrainingData());
      clearLog();
    }
  }, 5000);
};

const stop = (): void => {
  gaming.value = false;

  clearInterval(trainingTimer);
};

const computerMove = (): void => {
  if (!canvasRef.value) return;

  const prediction = model.predict(tf.tensor([ball.value.y]));

  const value = prediction.dataSync()[0];

  computer.value.y = value;
};

const train = async (data: any[]): Promise<void> => {
  if (data.length === 0) return;

  training.value = true;

  console.log('Training model...');

  const trainingData = tf.tensor(data.map(([y1]) => y1));
  const outputData = tf.tensor(data.map(([, y2]) => y2));

  await model.fit(trainingData, outputData, {
    epochs: 2,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch: ${epoch} - Loss: ${logs?.loss}`);
      },
    },
  });

  console.log('Training completed.');

  training.value = false;
};

const createMouseControls = (): void => {
  if (!canvasRef.value) return;

  canvasRef.value.addEventListener('mousemove', (event) => {
    if (!canvasRef.value) return;
    player.value.y = event.clientY - canvasRef.value.offsetTop;
  });
};

const createControls = (): void => {
  createMouseControls();
};

const getTrainingData = (): any[] => {
  return JSON.parse(JSON.stringify(log.value));
};

const render = (): void => {
  clearCanvas();
  setCanvasBackground();
  updateBallPosition();
  updatePossession();
  ballCollision();
  renderPaddle(player, 'left');
  renderPaddle(computer, 'right');
  renderBall();
  renderScoreboard();
};

onMounted(() => {
  centerPaddles();
  centerBall();
  createControls();
  createLoop();
});

window.addEventListener('keydown', (event) => {
  if (event.key === ' ') {
    if (gaming.value) {
      stop();
    } else {
      start();
    }
  }
});

const startLog = (): void => {
  logging.value = true;
};

const stopLog = (): void => {
  logging.value = false;
};

const formatTrainingData = (p: any, b: any): any => {
  return [b.y, p.y];
};

const registerTrainingData = (): void => {
  log.value.push(formatTrainingData(player.value, ball.value));
};

const clearLog = (): void => {
  log.value = [];
};

watch(possession, (newValue) => {
  if (newValue === 'left') {
    startLog();
  } else {
    stopLog();
  }
});
</script>
