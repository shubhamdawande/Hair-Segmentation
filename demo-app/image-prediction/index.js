const canvas = document.querySelector("canvas");
const ctx = canvas.getContext("2d");

const loadModel = async () => {
  // const model = await tflite.loadTFLiteModel('./models/hair_segmentation.tflite'); // lite model not work yet
  const model = await tf.loadLayersModel('../models/best_model_1.json');

  const outputTensor = tf.tidy(() => {
    const img = tf.browser.fromPixels(document.getElementById('sample'));
    const input = img.expandDims().div(127.5).sub(1);

    // first prediction takes > 500ms
    const startTime = performance.now();
    let outputTensor = model.predict(input);
    console.log('Prediction time: ', performance.now() - startTime);

    // 2nd prediction onwards takes < 10ms
    // const img2 = tf.browser.fromPixels(document.getElementById('sample2'));
    // const input2 = img2.expandDims().div(127.5).sub(1);
    // const startTime2 = performance.now();
    // let outputTensor2 = model.predict(input2);
    // console.log('Prediction time 2: ', performance.now() - startTime2);

    return outputTensor.round().add(1).mul(127.5).squeeze();
  });
  const data = Array.from(outputTensor.dataSync())
  const rgbaImage = Array.from(new Float32Array(320*320*4));
  for(let i = 0, j = 0; i < 320 * 320 * 4, j < 320 * 320; i += 4, j += 1){
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
  const imageData = new ImageData(Uint8ClampedArray.from(rgbaImage), 320, 320);
  ctx.putImageData(imageData, 0, 0);

  const originalImg = document.getElementById('sample');
  ctx.globalAlpha = 0.7;
  ctx.drawImage(originalImg, 0, 0, 320, 320);
}

loadModel();