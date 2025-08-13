#!/bin/bash
mkdir -p models

# Download the ONNX model
curl -L "https://store8.gofile.io/download/web/2cd1608f-edc6-4b01-913f-5ad8b4cab4d2/emotion-ferplus-8.onnx" -o models/emotion-ferplus-8.onnx

# Download Haar Cascade
curl -L "https://github.com/opencv/opencv/raw/master/data/haarcascades/haarcascade_frontalface_default.xml" -o models/haarcascade_frontalface_default.xml
