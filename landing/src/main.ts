const element = document.querySelector<HTMLCanvasElement>("#eeg-canvas");

if (!element) {
  throw new Error("No se ha encontrado el canvas EEG.");
}

const drawingContext = element.getContext("2d");

if (!drawingContext) {
  throw new Error("El navegador no permite dibujar el canvas EEG.");
}

const canvas: HTMLCanvasElement = element;
const context: CanvasRenderingContext2D = drawingContext;
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
const channels = 7;
let width = 0;
let height = 0;
let animationFrame = 0;

function resizeCanvas(): void {
  const bounds = canvas.getBoundingClientRect();
  const pixelRatio = Math.min(window.devicePixelRatio, 2);
  width = bounds.width;
  height = bounds.height;
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
}

function signal(x: number, channel: number, time: number): number {
  const phase = channel * 0.72;
  const pulseCenter = (time * 0.04 + channel * 137) % (width + 240) - 120;
  const distance = x - pulseCenter;
  return (
    Math.sin(x * 0.018 + time * 0.00045 + phase) * 7 +
    Math.sin(x * 0.065 - time * 0.0009 + phase * 1.8) * 3 +
    Math.sin(x * 0.17 + time * 0.0014 + channel) * 1.3 +
    Math.exp(-(distance * distance) / 240) * Math.sin(distance * 0.32) * 11
  );
}

function draw(time = 0): void {
  context.clearRect(0, 0, width, height);
  context.lineWidth = 1;
  for (let channel = 0; channel < channels; channel += 1) {
    const baseline = ((channel + 1) / (channels + 1)) * height;
    context.beginPath();
    for (let x = -2; x <= width + 2; x += 3) {
      const y = baseline + signal(x, channel, reducedMotion.matches ? 0 : time);
      x === -2 ? context.moveTo(x, y) : context.lineTo(x, y);
    }
    context.strokeStyle =
      channel === 2 || channel === 4
        ? "rgba(85, 214, 194, 0.38)"
        : "rgba(248, 241, 238, 0.14)";
    context.stroke();
  }
  if (!reducedMotion.matches) animationFrame = requestAnimationFrame(draw);
}

function restart(): void {
  cancelAnimationFrame(animationFrame);
  resizeCanvas();
  draw();
}

new ResizeObserver(restart).observe(canvas);
reducedMotion.addEventListener("change", restart);