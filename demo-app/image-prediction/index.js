const canvas = document.querySelector("canvas");
const ctx = canvas.getContext("2d");
img_size = 256

const loadModel = async () => {
  // const model = await tflite.loadTFLiteModel('./models/hair_segmentation.tflite'); // lite model not work yet
  const model = await tf.loadLayersModel('../models/prism/model.json');

  const startTime = performance.now();
  const outputTensor = tf.tidy(() => {
    const img = tf.browser.fromPixels(document.getElementById('sample'));
    const input = img.expandDims().div(127.5).sub(1);

    // first prediction takes > 500ms
    let outputTensor = model.predict(input);
    console.log('Prediction time: ', performance.now() - startTime);

    // 2nd prediction onwards takes < 10ms
    const startTime2 = performance.now();
    let outputTensor2 = model.predict(input);
    console.log('Prediction time 2: ', performance.now() - startTime2);

    return outputTensor.round().add(1).mul(127.5).squeeze();
  });
  const data = Array.from(outputTensor.dataSync())
  console.log('Prediction time: ', performance.now() - startTime);
  const rgbaImage = Array.from(new Float32Array(img_size*img_size*4));
  for(let i = 0, j = 0; i < img_size * img_size * 4, j < img_size * img_size; i += 4, j += 1){
    if(data[j] === 255){
      rgbaImage[i] = 0;
      rgbaImage[i + 1] = 255;
      rgbaImage[i + 2] = 0;
    } else {
      rgbaImage[i] = 0
      rgbaImage[i + 1] = 0;
      rgbaImage[i + 2] = 0;
    }
    rgbaImage[i + 3] = 50;
  }
  const imageData = new ImageData(Uint8ClampedArray.from(rgbaImage), img_size, img_size);
  ctx.putImageData(imageData, 0, 0);

  const originalImg = document.getElementById('sample');
  ctx.globalAlpha = 0.7;
  ctx.drawImage(originalImg, 0, 0, img_size, img_size);
}

loadModel();