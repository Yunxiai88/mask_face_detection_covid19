# Mask Detection And Face Recognition

## Description
* Under current COVID-19 pandemic situation, it is critical to ensure everyone stay safe and protected by wearing mask at all time. 
* There is a need for an auto mask detection, especially in public aeras like shopping malls, offices, schools etc.

* This projects aims to provide a system to automatically detect those who enter a public area not wearing a proper mask, and based on the pre-trained Deep Learning model, recognize the identity of the person, which can then be integrated to downstream system to provide possible warnings to the person or the management team.

## System Requirement
* OS: Windows 10/Linux
* GPU: NVIDIA GeForce GTX
* CUDA TOOLKIT: cuda_11.0.3
* cuDNN SDK: v11.0

## Technology
* LabelImg: clone project from (https://github.com/tzutalin/labelImg.git)
* Python: 3.x
* Flask
* OpenCV
* YOLOv4
* FaceNet
* InceptionResNetV2

## Training Details

1. **Training YOLOv4**
      * run command to Clone project from YOLOv4.
      ```
      $ git clone https://github.com/AlexeyAB/darknet.git
      ```
      * change configuration
      * Prepare the trainning data and put them in
      * run this command to train data
      * pending

2. **Training FaceNet**
     * If you want to directly use a pre-trained model for facial recognition, just skip this step.
     * If you want to implement a tranfer learning with a pre-trained model and your own dataset, you need to first download this pre-trained [model](https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit), put it in /models and unzip it. Make sure that the directory /models/20170512-110547 has 4 files.
       
       Then run
       ```bash
       $ python train_tripletloss.py
       ```
     
       The trained model will be in the /models/facenet.
     
     * If you want to train your own model from scratch. In ```train_tripletloss.py``` line 433, there is an optional argument named "--pretrained_model", delete its default value.
     
       Then run again 
       ```bash
       $ python train_tripletloss.py
       ```
     
       The trained model will also be in the /models/facenet.


## System Integration
1. **WebUI**

2. **Execution** 

download the trained weight from 
https://drive.google.com/file/d/1-7JQiyR2yLLPJJJNV30NZIgpqnwsk1Jh/view?usp=sharing
and put it under /yolov4/yolov4_custom_train_final.weights

* run this command at the project root directory:  
       ```
       $ python webApp/maskdetection.py --ip 127.0.0.1 --port 8000
       ```
* open URL in your browser:
       ```
       http://localhost:8000/
       ```
## References

* https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset
* https://www.mygreatlearning.com/blog/real-time-face-detection/
* https://colab.research.google.com/drive/1BcHkqaNdIvVAwwcBJZGyzdDLyR6Ygz5w?usp=sharing


