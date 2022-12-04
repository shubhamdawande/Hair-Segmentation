const video = document.querySelector('video');
const canvas1 = document.getElementById('c1');
const canvas2 = document.getElementById('c2');
const ctx1 = canvas1.getContext('2d');
const ctx2 = canvas2.getContext('2d');
let model = null;
// tf.setBackend('wasm');

const loadModel = async () => {
    model = await tf.loadLayersModel('../models/best_model_1.json');
}

const startWebcam = async () => {
    await navigator.mediaDevices.getUserMedia({video: true}).then(function(stream) {
        video.srcObject = stream;
        video.addEventListener('loadeddata', predictWebcam);
    });
}

const predictWebcam = async () => {
    await loadModel();
    console.log(tf.getBackend());
    drawCanvas();
}

const drawCanvas = () => {
    // draw video to canvas1
    canvas1.height = 320; // video.videoHeight / 2;
    canvas1.width = 320; // video.videoWidth / 2;
    ctx1.drawImage(video, 0, 0, canvas1.width, canvas1.height);

    // get image from canvas1    
    const img = ctx1.getImageData(0, 0, 320, 320);
    predictImage(img);
    window.requestAnimationFrame(drawCanvas)
}

// element: htmlImage / htmlVideo / ImageData
const predictImage = async (element) => {
    const outputTensor = tf.tidy(() => {
        const img = tf.browser.fromPixels(element);
        const input = img.expandDims().div(127.5).sub(1);
        let outputTensor = model.predict(input);
        outputTensor = outputTensor.round().add(1).mul(127.5).squeeze();
        return outputTensor
    });
    const data = outputTensor.dataSync();
    const dataArr = Array.from(data);
    const rgbaImage = Array.from(new Float32Array(320*320*4));
    for(let i = 0, j = 0; i < 320 * 320 * 4, j < 320 * 320; i += 4, j += 1){
        if(dataArr[j] === 255){
        rgbaImage[i] = 0;
        rgbaImage[i + 1] = 255;
        rgbaImage[i + 2] = 0;
        } else {
        rgbaImage[i] = 0
        rgbaImage[i + 1] = 0;
        rgbaImage[i + 2] = 0;
        }
        rgbaImage[i + 3] = 100;
    }
    const imageData = new ImageData(Uint8ClampedArray.from(rgbaImage), 320, 320);
    ctx2.putImageData(imageData, 0, 0);
    const canvasOriginalImg = new Image();
    canvasOriginalImg.src = element;
    ctx2.globalAlpha = 0.9;
    ctx2.drawImage(canvasOriginalImg, 0, 0, 320, 320);
}

startWebcam()