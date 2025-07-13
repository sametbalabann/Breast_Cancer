 IDC Detection with CNN (Breast Histopathology)

This project uses a Convolutional Neural Network (CNN) to detect the presence of Invasive Ductal Carcinoma (IDC) in histopathology image patches (50x50 RGB) extracted from breast cancer biopsy slides.

Prepared a Small Dataset (~10%)

Trained the Model

Predicted on New Image
Place a test image under `data/test/`, then run:

Model
The CNN model is a simple 3-layer convolutional network trained to classify IDC-positive vs IDC-negative image patches.

 Dataset Source
 Kaggle: [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)