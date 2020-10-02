#!/usr/bin/python3
# USAGE
#python3 Pi-DNN-ImageNet.py --prototxt models/squeezenet_v1.0.prototxt --model models/squeezenet_v1.0.caffemodel --labels synset_words.txt --image Pics/ILSVRC2012_val_00000001.JPEG --label_Acc synset_words_imgnet.txt --Acc_Validation val.txt
#can also use --imagedir instead of --image for a directory of only images
# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import sys
import glob
import os
import linecache
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
	help="path to input image", default=None)
ap.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',
                    default=None)
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-l", "--labels", required=True,
	help="path to ImageNet labels (i.e., syn-sets)")
ap.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
ap.add_argument('--label_Acc', help='Label of the image using synset_words from ImageNet', default = None)
#ap.add_argument('--Acc_Synset', help='Synset for Imagenet', default = None)
ap.add_argument('--Acc_Validation', help='Label of the image', default = None)
args = vars(ap.parse_args())

label_Acc = args["label_Acc"]
#Acc_Synset = args["Acc_Synset"]
Acc_Validation = args["Acc_Validation"]
# For images in the directory :Minoo
IM_DIR = args["imagedir"]
IM_NAME = args["image"]
min_conf_threshold = float(args["threshold"])

# Get path to current working directory:Minoo
CWD_PATH = os.getcwd()


# Define path to images and grab all image filenames:Minoo
if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_DIR)
    images = glob.glob(PATH_TO_IMAGES + '/*')
else:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_NAME)
    images = glob.glob(PATH_TO_IMAGES)


# load the class labels from disk
rows = open(args["labels"]).read().strip().split("\n")
classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

rows_iamgenet = open(args["label_Acc"]).read().strip().split("\n")
classes_imagenet = [r[r.find(" ") + 1:].split(",")[0] for r in rows]

with open(Acc_Validation) as inf:
        reader = csv.reader(inf, delimiter=" ")
        first_col = list(zip(*reader))[0]

with open(Acc_Validation) as inf:
	    reader = csv.reader(inf, delimiter=" ")
	    second_col = list(zip(*reader))[1]


# load the input image from disk
#image = cv2.imread(args["image"])
count_overal = 0
count_founded = 0
start = time.time()
for image_path in images:
    image = cv2.imread(image_path)
    # our CNN requires fixed spatial dimensions for our input image(s)
    # so we need to ensure it is resized to 224x224 pixels while
    # performing mean subtraction (104, 117, 123) to normalize the input;
    # after executing this command our "blob" now has the shape:
    # (1, 3, 224, 224)
    blob = cv2.dnn.blobFromImage(image, 1, (224, 224), (104, 117, 123))

    # load our serialized model from disk
   # print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    # set the blob as input to the network and perform a forward-pass to
    # obtain our output classification
    net.setInput(blob)

    preds = net.forward()
    # sort the indexes of the probabilities in descending order (higher
    # probabilitiy first) and grab the top-5 predictions
    preds = preds.reshape((1, len(classes)))
    idxs = np.argsort(preds[0])[::-1][:1]

    temp = image_path.split("/")
    im_name_2 = temp[len(temp)-1]
    #print(im_name_2)
    index = first_col.index(str(im_name_2))




    index_final_result = int(second_col[index])
	#Correct_result = linecache.getline(Acc_Synset, int(second_col[index]))
	#Correct_result = Correct_result.strip()
	#if Correct_result == res:
		#print("yes")

	# loop over the top-5 predictions and display them
    count_overal += 1
    for (i,idx) in enumerate(idxs):
            # draw the top prediction on the input image
            #print(i, idx, classes[idx])
            #if label.lower() in classes[idx]:
	    if idx == index_final_result:
                count_founded += 1
               # print("Accurate")
               # print(classes[idx])
               # print(count_overal, ": [INFO] {}. label: {}, probability: {:.5}".format(1,
                       # classes[idx], preds[0][idx]))
                break
           # print(count_overal, ": [INFO] {}. label: {}, probability: {:.5}".format(i+1,
                       # classes[idx], preds[0][idx]))
            #if preds[0][idx] > min_conf_threshold:
             #   text = "Label: {}, {:.2f}%".format(classes[idx],
              #          preds[0][idx] * 100)
               # cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                #            0.7, (0, 0, 255), 2)

            #if preds[0][idx] <= min_conf_threshold:
             #   text = "Solution is less than minimum confidence" + "Label: {}, {:.2f}%".format(classes[idx],
              #          preds[0][idx] * 100)
               # cv2.putText(image, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX,
                #            0.7, (0, 0, 255), 2)

            # display the predicted label + associated probability to the
            # console
            #if preds[0][idx] > min_conf_threshold:


    # display the output image
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)
end = time.time()
if count_overal == 0:
	print("error")
else:
	print("{:.5}".format(end - start))
	print(str(float(count_founded/count_overal)))
#print(count_founded)
#print(count_overal)

#with open("result.txt","a") as f:
   # print(file = f)
   # print(file = f)
   # print("model is " + args["model"], file = f)
   # print("label is " + label , file = f)
   #Processing Time:
   # print("{:.5}".format(end - start), file = f)
   # print("founded is " + str(count_founded), file = f)
   # print("overal is " + str(count_overal), file = f)
   # print( str(float(count_founded/count_overal)), file = fC)
# Clean up:Minoo
cv2.destroyAllWindows()
