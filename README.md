# Detection-of-Non-stationary-objects-on-Dublin-Roads-using-transfer-learning-for-Autonomous-Vehicles

Contents of the folder:

1. TensorFlowObjectDetectionApi.zip -- Not uploaded here due to file size limitations.
This file is basically TensorFlow Object Detection API files in zipped format which need to be extracted and installed in the system according to instructions at https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md (TensorFlow, 2019).

2. main_files: This folder contains files which need to be copied to /models-master/research/object_detection folder of the installed object detection API
Folder contains following folders and files:-

annotations: This folder contains annotations or labelled data for train and test sets in XML format created with the help of python package labelimg.

images: This folder contains images for train set, test set and validation set. This folder also contains following three files.
analysis.ipynb – It contains code for Exploratory Data Analysis of images in train and
test sets.

test_labels.csv – It contains labelled data for test images converted from
annotations file in XML format.

train_labels.csv – It contains labelled data for train images converted from
annotations file in XML format.

training: This folder contains files necessary for training of Faster R-CNN model using transfer learning. Pre-trained model for Faster-RCNN with inception V2 network has been downloaded from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md and extracted into this folder. It also contains ‘faster_rcnn_inception_v2_pets.config’ file which can be used to configure training and ‘labelmap.pbtxt’ file which is used to write relevant object classes.

training-ssd: This folder contains files necessary for training of SSD model using transfer learning. Pre-trained model for SSD with MobileNet V2 network has been downloaded from https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md and extracted into this folder. It also contains ‘ssd_mobilenet_v2_coco-1.config’ file which can be used to configure training and ‘labelmap.pbtxt’ file which is used to write relevant object classes.

Other files in main_files folder are:

generate_tfrecord.py: It contains code to convert csv label files into TFRecord
format required by TF object detection API.

inference_dublin.ipynb: It contains code to run inference on images after training of
models.

inference_dublin.ipynb: It contains code to run tensorboard server.

test.record: TFRecord format for annotations of test images converted from csv test
labels file. Not uploaded here due to file size limitations.

train.record: TFRecord format for annotations of train images converted from csv
train labels file. Not uploaded here due to file size limitations.

training_custom.ipynb: It contains code for training of models.

3. trained_models: This folder contains saved models after training using transfer learning. It has three folders.

inference_graph-6861-frcnn: Faster R-CNN model saved after 6,861 steps of training

inference_graph-20698-frcnn: Faster R-CNN model saved after 20,698 steps of training

inference_graph_ssd_20000: SSD model saved after 20,000 steps of training

4. detected_images: This folder contains some of the images inferred using different models. Images have been named in following way.
Example- test0.png : an image from test set inferred through pre-trained Faster R-CNN
model.

test0-6861.png: an image from test set inferred through Faster R-CNN model
re-trained for 6,861 steps.

test0-20698.png: an image from test set inferred through Faster R-CNN model
re-trained for 20,698 steps.

test0-ssd.png : an image from test set inferred through pre-trained SSD model.

test0-ssd-20000.png: an image from test set inferred through SSD model
re-trained for 20,000 steps.
