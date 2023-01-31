const video = document.querySelector('video');
const canvas1 = document.getElementById('c1');
const canvas2 = document.getElementById('c2');
const ctx1 = canvas1.getContext('2d');
const ctx2 = canvas2.getContext('2d');
let model = null;
// tf.setBackend('wasm');
img_size = 256

const loadModel = async () => {
    model = await tf.loadLayersModel('../models/prism/model.json');
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
    canvas1.height = img_size; // video.videoHeight / 2;
    canvas1.width = img_size; // video.videoWidth / 2;
    ctx1.drawImage(video, 0, 0, canvas1.width, canvas1.height);

    // get image from canvas1    
    const img = ctx1.getImageData(0, 0, img_size, img_size);
    predictImage(img);
    window.requestAnimationFrame(drawCanvas)
}

// element: htmlImage / htmlVideo / ImageData
const predictImage = async (element) => {
    const startTime = performance.now();
    const outputTensor = tf.tidy(() => {
        const img = tf.browser.fromPixels(element);
        // const input = img.expandDims().div(127.5).sub(1);
        const input = img.expandDims().div(255);
         
        let outputTensor = model.predict(input);

        // outputTensor = outputTensor.round().add(1).mul(127.5).squeeze();
        outputTensor = outputTensor.round().mul(255).squeeze();
        return outputTensor
    });
    const data = outputTensor.dataSync();
    console.log('prediction time: ', performance.now() - startTime);
    const dataArr = Array.from(data);
    const rgbaImage = Array.from(new Float32Array(img_size*img_size*4));
    for(let i = 0, j = 0; i < img_size * img_size * 4, j < img_size * img_size; i += 4, j += 1){
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
    const imageData = new ImageData(Uint8ClampedArray.from(rgbaImage), img_size, img_size);
    ctx2.putImageData(imageData, 0, 0);
    const canvasOriginalImg = new Image();
    canvasOriginalImg.src = element;
    ctx2.globalAlpha = 0.9;
    ctx2.drawImage(canvasOriginalImg, 0, 0, img_size, img_size);
}

startWebcam()