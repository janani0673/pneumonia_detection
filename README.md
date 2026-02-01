Pneumonia Detection Using CNN

Overview:
This project is a Pneumonia Detection system using Deep Learning (Convolutional Neural Networks). It can automatically detect pneumonia from chest X-ray images, helping healthcare professionals with faster diagnosis.

The model is trained on a public chest X-ray dataset and can classify images into:
1.Normal
2.Pneumonia

Features:
1.Automatic Pneumonia Detection from chest X-ray images.
2.CNN-based deep learning model for high accuracy.
3.User-friendly interface (can be extended with Streamlit or Flask).
4.Supports prediction for single images.
5.Dependencies managed using requirements.txt."# pneumonia_detection" 

Folder Structure:
pneumonia_detection-main/
│--app.py
├─ pneumonia_cnn.py         
├─ dataset/                 
├─ venv/                    
├─ requirements.txt         
├─ model.h5 / model.pkl      
└─ README.md   

Installation:
1.Clone the repository:

   git clone https://github.com/your-username/pneumonia_detection.git
   cd pneumonia_detection

2.Create a virtual environment:
    python -m venv venv

3.Activate the environment:
4.Windows:
     venv\Scripts\activate

5.macOS/Linux:
     source venv/bin/activate

6.Install dependencies:
      pip install -r requirements.txt

Usage:
Train the model (optional if pretrained model is provided):
      python pneumonia_cnn.py
predict_image("path/to/chest_xray.jpg")

Dataset:
The project uses Chest X-ray images dataset, which contains labeled images for Normal and Pneumonia cases.
Dataset link: Kaggle Chest X-ray Dataset

Dependencies:
Python 3.x
TensorFlow / Keras
NumPy
OpenCV (optional for image preprocessing)
Matplotlib
scikit-learn

All dependencies can be installed using:
     pip install -r requirements.txt
     
Future Work:
1.Integrate Streamlit or Flask web app for interactive predictions.
2.Improve accuracy using transfer learning (e.g., using pre-trained models like VGG16, ResNet50).
3.Add multi-class detection (different pneumonia types).
