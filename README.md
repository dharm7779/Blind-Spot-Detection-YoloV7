# Car-Blindspot-Detection
Car Blindspot Object Detection is an important safety feature that is used to detect and alert drivers of objects in their vehicle's blind spot. Blind spots are areas around the vehicle that are not directly visible to the driver, and they can be a significant safety risk when driving. The objective of this project is to develop an object detection system that can detect objects in the blind spot of a car and alert the driver in real time.

The system will use a camera mounted on the side mirror of the car to capture images and video of the blind spot area. The video will then be processed using computer vision techniques, such as object detection and classification, to identify any objects in the blind spot. The system will then alert the driver with warnings to take necessary precautions, such as checking the side mirrors, before changing lanes or turning.

This project has the potential to improve road safety and reduce the number of accidents caused by blind spot collisions. It can also be further extended to include other features, such as pedestrian and obstacle detection, to make driving safer and more efficient. In this documentation, we will discuss the architecture, implementation, and evaluation of the Car Blindspot Object Detection system.

![WhatsApp Image 2023-04-06 at 03 29 10](https://user-images.githubusercontent.com/96363330/230221364-c340ff66-c768-4d54-8270-82ae20cae93e.jpg)


# Libraries and Dependencies needed
To run the car blindspot object detection code, you will need to have the following libraries and dependencies installed:
1. OpenCV: a popular computer vision library for real-time image processing.
2. Numpy: a library for numerical computing in Python.
3. Imutils: a library of convenience functions for working with OpenCV.

# YOLOv7

![image](https://user-images.githubusercontent.com/96363330/230222463-ee7be9f9-17ea-45df-ad4a-359719151606.png)

YOLO - object detection
YOLO — You Only Look Once — is an extremely fast multi object detection algorithm which uses convolutional neural network (CNN) to detect and identify objects.
Using a YOLO model is an excellent choice for car blindspot object detection. YOLO is a real-time object detection system that can detect and recognize different objects in an image or video. YOLO is known for its speed and accuracy, making it an ideal choice for applications where real-time performance is critical.

YOLO works by dividing the input image into a grid and then predicting the bounding boxes and class probabilities for each cell in the grid. This approach is much faster than other object detection methods that require the image to be scanned multiple times at different scales and locations. YOLO also provides excellent accuracy, even on small objects, making it a great choice for detecting objects in car blindspots.

The neural network has this network architecture. 

![image](https://user-images.githubusercontent.com/96363330/230216641-45f3636d-2a7f-4cde-ad29-818fee9c3985.png)

YOLOv7 compared to other models.

![image](https://user-images.githubusercontent.com/96363330/230222287-c0c1582d-6ef4-4197-8516-5ccd6049fa25.png)

In order to run the network you will have to download the pre-trained YOLO weight file. This YOLO weight file needs to be put in yolo-coco folder before running the code.

https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov7.weights

Also download the the YOLO configuration file. 

https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov7.cfg

# COCO Dataset
The COCO (Common Objects in Context) dataset is a large-scale object detection, segmentation, and captioning dataset, with over 330K images, each annotated with object labels and segmentation masks. It is widely used in the computer vision community for benchmarking object detection models due to its large size and diversity.

While custom datasets can be useful for specific applications, they are often limited in size and diversity. By using a well-established dataset like COCO, we can train our model on a diverse set of images, ensuring that it can detect a wide variety of objects accurately. Additionally, using COCO allows us to compare our model's performance with other state-of-the-art models, which have also been trained on the same dataset, making it easier to evaluate the effectiveness of our model.

In this project, we have used the COCO dataset to train our YOLOv7 model to detect objects in car blind spots. By using a pre-existing dataset like COCO, we were able to train our model on a diverse range of images, ensuring that it can accurately detect a wide variety of objects.

# USAGE
For usage, use:
```
$ python yolo.py --input "path/input.mp4" --output "path/outut.mp4" --yolo yolo-coco
```

# Additional Tips
1. Make sure you have a powerful enough GPU to run the YOLOv7 model efficiently.
2. Consider using a virtual environment to keep your Python packages isolated and prevent conflicts between different versions of libraries.
3. Experiment with the confidence threshold and non-maximum suppression parameters to find the best trade-off between accuracy and speed.
4. If you're working with a different dataset, make sure to adjust the number of classes and the paths to the images and annotation files accordingly.
5. To improve the accuracy of the model, you can fine-tune it on your own dataset or use transfer learning from a pre-trained model. 
