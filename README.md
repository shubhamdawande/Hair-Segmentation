# README

### This is a neural network implementation inspired from U-NET for generating hair segmentation masks on image & video data. Built for virtual try-on of beauty products.


### Details
- Architecture: Modified U-Net network implemented with keras. Details of the NN model are inside `main.ipynb`
- Training platform: Google colab T4 GPU
- Deployment Backend: Tensorflow JS on browser via webgl
- Input dataset: Sampled images from CelebA dataset and generated segmentation masks
- Execution speed: ~10 video frames per second

### Relevant files:
- main.ipynb:- training script
- demo-app:- webapp for predicting on webcam & images
- dataset:- data preparation scripts
- converter.sh:- Tensorflow to TFJS model format converter

### Results on images:
![Screenshot](result.png)
prediction time: ~33ms

### TODO
Handle temporal inconsistency for videos