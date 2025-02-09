# gator-guardian-2025
## Inspiration
Construction sites pose inherent safety risks, and ensuring workers follow safety protocols is critical. Our project leverages Jetson Nano to run YOLO (You Only Look Once) models for real-time detection of personal protective equipment (PPE) such as masks, hard hats, safety vests, and person detection. By developing a tool like Gator Guardian, we can contribute to reducing workplace accidents and ensuring that all safety protocols are followed on the job site, ultimately saving lives and preventing injuries. 

## What it does
Our project uses a YOLO model to build a real-time construction safety detection system. It identifies and locates objects within an image to see whether workers on a construction site are wearing necessary personal protective equipment such as masks, hard hats, and safety vests. If the system detects a worker is not wearing a mask, hard hat, or safety vest, it will push a notification to the manager attaching an image to let them know that a worker is not in compliance. There have been various accidents in construction sites, but with our project's combination of hardware and machine learning, these challenges can be resolved. 

## How we built it
We built the construction safety detection system by first setting up the NVIDIA Jetson Nano. Once the Jetson was set up, we started by installing all the required packages. We then imported a pre-trained Construction-Site-Safety-PPE-Detection YOLO model, a computer vision model that can detect objects from the trained classes which include ["hardHat", "mask", "NO-hardHat", "NO-Mask", "NO-SafetyVest", "person", "safety-cone", "safety-vest", "machinery", vehicles"]. We were able to pipe our camera live stream into the model and extract what the model detected. If "NO-hardHat", "NO-Mask", "NO-SafetyVest" was found, a push notification, including an snippet of the frame it saw, is emailed to a manager to notify them of a safety violation. 

A decision was made to use a pre-trained model instead of training our own from scratch due to the lack of computing resources available to us. Several pre-trained models were tested from HuggingFace and GitHub, and we chose the most accurate one. We piped in a camera video stream to a pre-trained model and wrote scripts to optimize it to output correct image classifications. The scripts processed each image frame to identify if there was a safety violation, and if one was spotted, an email would be sent over Gmail's SMTP server to a manager's email with the suspect image frame attached.

## Challenges we ran into
Initially, we were planning on training our own model. We found large datasets of PPE images on Kaggle and HuggingFace and searched for resources available to train a computer vision model. These models had thousands of images of different types of PPE like hard hats, safety vests, gloves, masks, etc. We installed heavy machine learning modules like CUDA, CuDNN, and Pytorch that took up multiple gigabytes to prepare for training a model on the datasets we found. However, we were not provided with any NVIDIA GPU access from our university's supercomputer for the hackathon, nor did any of us have access to a different machine powerful enough to train a model of that scale. 

Finding an optimal model required much testing and thought. We began our project by running examples provided by NVIDIA's jetson-inference library. These ran on a Docker container, and we attempted to customize it by installing our own packages and modifying the Dockerfile. However, we were unsuccessful and found that using a Docker container did not suit our needs of having a more permanent environment for running models. We decided to use a Python virtual environment instead, since we needed to use many Python packages like Pytorch, ultralytics, and transformers. 

## Accomplishments that we're proud of
We managed to boost the accuracy of the YOLO model's confidence level from 20% to 98%. We were also able to overcome the limited computing power of the Jetson Nano to run an accurate machine learning model to protect the safety of workers.

## What we learned
Many software tools were tested before deciding on which to use in this project. We all learned how to use the Linux/Unix terminal proficiently, since most Jetson Nano interactions involved a command line interface. In addition, we ran Python virtual environments and Docker containers on our own machines, which were both Windows and Mac laptops. 

From experimentation, we found benefits and drawbacks of using pre-trained models. Many that we found online were inaccurate or required specific/outdated configurations. However, some were accurate and helpful. It would have been better if we could make a custom model based off our own data while building off a pre-trained one, but we did not have the computing resources to do so.

## What's next for Gator Guardian 
We would like to optimize the models so that it can detect different classes of protective equipment in different orientations. For example, a better model could be able to recognize a cut-off image of a hard hat and not flag it for a safety violation. If provided with more funds, we could improve our model quality and mobility. The Jetson Nano was a restrictive platform since it was weaker than a conventional computer, which limited our capability to run advanced computer vision models. However, a small device like the Nano can be placed on a robot and become portable. In the future, if provided with a more powerful hardware, we could run a more accurate model.

## Sources
[https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection](https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection)

[https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection](https://www.kaggle.com/datasets/mugheesahmad/sh17-dataset-for-ppe-detection)

[https://github.com/dusty-nv/jetson-inference](https://github.com/dusty-nv/jetson-inference)
