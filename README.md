# samrakchyan
A Machine Learnig Project for Yomari Code Camp 2022 


## Description

In this project we are going to work on perspectives in Deep Learning for Wildlife monitoring.\
The project is interdisciplinary and involves the study of animal ecology as well as computer science.\
Traditionally, the wildlife is monitored with the help of human field workers for counting the animals, patrolling the forests areas or using the camera traps.\
Nowadays, the data from the different sources is too large to incorporate the human force.
So, we believe in such scenario a Deep Learning model could solve the problem.\
The model itself will maintain more accuracy than the humans as well as it will also do this in lesser time.\
This will help to fight against illegal poaching of endangered species, preserve the biodiversity and monitor other activities in the animal world.\
With the help of this project we aim to unleash the power of Machine Learning and Deep Learning to have the better understanding of the animal world.
The classification model is trained and validated on the 14,182 images. The dataset used was a subset of data from the following link.\
Please visit (https://lila.science/datasets/missouricameratraps) for more information.\
The detection model due to fact that was trained on RetinaNet and was very GPU intensive we decided to take a look of its capability trained on only few images(159).\
The PASCAL VOC annotations format for training the detection model can be found at the following link.\
Please visit (https://drive.google.com/drive/folders/19hIwFRhshdOALWb4vwUwth1V2jBYX9aq?usp=sharing) for finding the annotations.


## Installation Guide
  1. On the terminal run: `pip install virtualenv`
  2. Create a virtual environment: `virtualenv venv`
  3. Activate the virtual environment: `venv\Scripts\activate` 
  4. create ml_models dir inside wildlife dir
  5. ([add the following machine learning models into ml_models folder(ml_models should be inside wildlife directory)](https://drive.google.com/drive/folders/1dT73B_KYImcEWfgwSWXY5-CNgsAdfJnL?usp=sharing))
  6. Finally run `pip install -r requirements.txt`
  7. To execute `python manage.py runserver`

## Fine Tuned ResNet50 Performance Graph
  ![Screenshot](resnet50.png)
