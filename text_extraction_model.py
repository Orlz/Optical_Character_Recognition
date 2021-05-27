#!/usr/bin/env python

"""
===============================================================
----- Extract Text From Images Using EAST and Pytesseract -----
===============================================================

This script takes images as .jpeg files and runs them through a loop which does the following 3 tasks: 
    1. Identifies where the text is in the image using the the pre-trained neural network EAST
    2. Performs text recognition using pytesseract to convert the words in the image into computer recognisable text
    3. Saves the extracted text and their coordinate locations into a csv file in the Output folder

The script has been inspired by Adrian Rosebrock's use of EAST for text detection in videos. A number of ammendments have however
been made which take the form of restructuring the order, adapting the script to include argparse commands and multiple images,
adding in the pytesseract functionality, adding in the text cleaning, and saving the extracted text in csv files. Adrian's
original script can be found here: https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/


Pretrained neural network EAST: (An Efficient and Accurate Scene Text detector)
Original information on the EAST network can be found here: https://arxiv.org/abs/1704.03155v2 

Some important elements to note from this script: 

    net (line 313)  = This is the main brain of the text detector system. 
                     It is essentially a neural network developed by the EAST team which was trained on a vast array of images, 
                     to be able to detect all kinds of text from such images. This ranged from pictures of the names and numbers
                     on football players jersey's taken from action pictures of the match to the text on buses and much more. 
                     
                     
    text_detector  = This function preprocesses the images to get them into the format to work with East. It then then runs 
                     it through the EAST net and draws a rectangular ROI around the areas it believes to be text. 
                              
The script has the following optional argparse parameters: 
    -m  --min_confidence   Minimum probability needed for the area to be considered for text recognition (default: 0.5)
    -h  --height           The height dimension for the resized image (must be a multiple of 32)
    -w  --width            The width dimension for the resized image (must be a multiple of 32)         
                     

Usage: 
$ python3 src/text_extraction_model.py
                     
"""

"""
=================================
----- Import Depenendencies -----
=================================
"""

# Core libraries
import re, sys, os, argparse, csv, glob

# image processing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#EAST tools
from utils.imutils import non_max_suppression

# OCR tool
import pytesseract

# NLP
from spellchecker import SpellChecker

from PIL import Image

"""
===============================
----- Argparse Parameters -----
===============================
"""

# Initialize ArgumentParser class
ap = argparse.ArgumentParser()
    
# Argument 1: Minimum Confidence 
ap.add_argument("-m", "--min_confidence",
                type = float,
                required = False,
                help = "[INFO] The minimum probability needed for the area to be considered for text recognition (default: 0.5)",
                default = 0.5)
    
# Argument 2: Height 
ap.add_argument("-h", "--height",
                type = int,
                required = False,
                help = "[INFO] The height dimension for the resized image, must be a multiple of 32, (default: 640).",
                default = 640)
    
# Argument 3: Width 
ap.add_argument("-w", "--width",
                type = int,
                required = False,
                help = "[INFO] The width dimension for the resized image, must be a multiple of 32, (default: 640).",
                default = 640)      
        
# Parse arguments
args = vars(ap.parse_args())    

"""
===================================
----- Text Detection Function -----
===================================
"""
"""
This function uses EAST text detector to find where text is in an image and pytesseract to translate the words into text
It then stores the images with their regions of text detected in the Output folder (images) with csv's of the text (detected_text)

The function completes the following 9 steps 
    1. Resizing: It resizes the image to fit within the dimensions of the EAST CNN (multiples of 32)
    2. Preprocessing: It then standardises the image colours to take away the influence of brightness, light exposure etc. 
       (this makes all images look more similar and thus the model can generalise to suit them)
    3. Find regions with text: From here, it calls the EAST neural network and collects the confidence scores and 
       probabilities of each coordinate 
       It uses these to loop through and check whether the current region is above the minimum_conficence threshold 
    4. Check angles: The model then checks the orientation angle of the text (i.e. is text on a horizonal plane or more slanted?) 
    5. Coordinates and Confidence: the model then computes where the 4 coordinates of the bounding box should be for the text
       and also the confidence probabilities. Both variables are stored in separate lists [rect] and [confidences] respectively.
    6. Non-Max-Suppression: Filters the number of boxes to be drawn, taking only those with a high confidence 
    7. Draw the boxes: The coordinates are used to draw green rectangles around regions where text has been detected 
    8. Text recognition: pytesseract is then run over the region to try to translate the text 
    9. Save the results: The detected text files are then saved into the Output directory for each image
    
Returns: the original image with green rectangles overlayed (orig)
"""

#Calling the EAST library
net = cv2.dnn.readNet("frozen_east_text_detection.pb")

def text_detector(image, image_name, height, width):
    """
    Setup
    """
    
    orig = image                                     # Name the original image
    (origH, origW) = image.shape[:2]                 # Get the height and width using the shape function in cv2               

    """
    Step One: Resize the image
              The East detector requires images to be in a shape size where both height and text are multiples of 32.
              This is just because of the way the neural-network was trained, on images of standardised sizes.
    """
    
    (newW, newH) = (width, height)   # Creating values to resize the image to fit with the EAST package 
    rW = origW / float(newW)                         # Ensuring image width will be a multiple of 32 (requirement) 
    rH = origH / float(newH)                         # Ensuring image height will be a multiple of 32 (requirement)

    image = cv2.resize(image, (newW, newH))          # Then we'll resize the image into a 640 by 640 sqaure 
    (Height, Width) = image.shape[:2]

    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",             # An output layer of EAST which informs on the confidence of identifying text 
        "feature_fusion/concat_3"]                   # Tells us about the orientation of the text 

    
    """
    Step 2: Preprocessing: Known as "Block from Image"
    - This blob function is part of the neural network library in OpenCV
            Essentially, this is a way of standardising the images. It works by finding the mean of the image for all 3 red, blue
            and green colour channels and subtracts these to take away the influence of brightness, light exposure etc. 
            This makes the neural network more robust and generalizable when new images are fed in. 
    """

    blob = cv2.dnn.blobFromImage(image, 1.0, (Width, Height),  
        (123.68, 116.78, 103.94), swapRB=True, crop=False)     # Values are the mean RGB values calculated from initial images
                                                               # This allows us to use the same RGB values as EAST does

        
    net.setInput(blob)                                         #Employs the subtraction & standarfization of the image
    (scores, geometry) = net.forward(layerNames)               #Gives the scores and geometries from layers (line X and X)

    """
    Step 3: Detect the regions of text using a confidence probability  
    """
    
    (numRows, numCols) = scores.shape[2:4]                     #Defines number of columns & rows of the image shape
    rects = []                                                 #List to store bounding box (bb) coordinates of text 
    confidences = []                                           #List to store the probability associated with each bb

    for y in range(0, numRows):                                # This loop helps to identify where the text is in the image 
                                                               # The loop extracts the scores and geometry data for row y 
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns of this current row (y) 
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            # This filters out weak text detectionss
            if scoresData[x] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            
            """
            Step 4: Consider the angle of the text
            """

            # extract the rotation angle for the prediction and then 
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # drawing the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            
            """
            Step 5: Coordinates and confidence 
            """

            # Using the corrdinates to make and draw the bounding box
            # And adding the probability score to the respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    
    """
    Step 6: Non-Max-Suppression
    """
    #This function ensures we're using the most confident box and not drawing hundreds around the same chunk of text 
    #It works by suppressing all the lower value scores and taking only the most confident ones
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    
    """
    Step 7: Draw the bounding boxes on the image 
    """
    
    for (startX, startY, endX, endY) in boxes:             #loop over the bounding boxes detected above 
                                                           #scale the coordinated based on the respective ratios
        startX = int(startX * rW)                      
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        
        #Add padding to each bounding box 
        #this is done by calculating the deltas along the x and y axis 
        dX = int((endX - startX) * 0.05)        #padding can produce better OCR results as it leaves room for 
        dY = int((endY - startY) * 0.05)        #the surrounding letters incase they were not detected 
        
        # Use this delta value to add the padding onto the coordinate values 
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
        
        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 3)
        
        # now we can extract the padded roi 
        roi = orig[startY:endY, startX:endX]
        
        """
        Step 8: Pytesseract text detection 
        """   
        
        # initialize the list of results
        results = []
        box_coordinates = []
        box_text = []
        image_type = []
        
        #We set up tesseract to read the text as a sentence (defined using 7) 
        config = ("-l eng --oem 1 --psm 7")
        
        #pass it to pytesseract to translate 
        text = pytesseract.image_to_string(roi, config=config)
        
        #...and save the results in separate lists 
        results.append(((startX, startY, endX, endY), text))
        box_coordinates.append((startX, startY, endX, endY))
        box_text.append(text)
        image_type.append(image_name)
        
    
    # sort the results bounding box coordinates from top to bottom
    results = sorted(results, key=lambda r:r[0][1])
    
    """
    Step 9: Save the results
    """
    
    #save the extracted info as a csv 
    dataframe = pd.DataFrame({"Image": image_type, 
                              "Text": box_text,
                              "Coordinates": box_coordinates})
    
    #save the results as a txt file 
    dataframe.to_csv("Output/detected_text/"+image_name+"_text.csv", index = False)
    
    print(f"\nYour {image_name} has been processed") 
    
    
    return orig

        
"""
=========================
----- Main Function -----
=========================
"""
def main():
    """
    ---------------
    ---- Setup ----
    ---------------
    """
    print ("\nHello! Let's try to translate some old historical documents into csv files containing text") 
    print("\nI'll start up by calling the EAST pre-trained neural network")

    #Calling the EAST library
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")  
    
    print("\nThe network's ready so now we can loop through your images and see if we can detect any text among them") 
    
    height = args["height"]
    width = args["width"]      
        
    """
    -------------------------
    ---- List the images ----
    -------------------------
    """
    #As there's just a few images we'll list them here to keep our script interpretable.
    #If a larger dataset needed to use the script, these could be converted into argparse commands or a loop

    image1 = 'typewritten_letter'
    image2 = 'handwritten_note'
    image3 = 'newspaper_cover'
    image4 = 'handwritten_tag'
    image5 = 'handwritten_letter'
    image6 = 'handwritten_poem'


    #Create a list of the images in the directory 
    image_array = [image1, image2, image3, image4, image5, image6]
        
    """
    ----------------------------------------------------
    ---- Create loop to run through image directory ----
    ----------------------------------------------------
    """

    #For every image in the image_array list
    for image in image_array:
        #Get the image name
        image_name = image
        #Get the path to the image 
        image_path = "data/"+image_name+".jpeg"
        #Read in the image 
        image = cv2.imread(image_path)
        #Resize the image to fit with the EAST dimensions 
        imageO = cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)
        #Create a copy of the original image 
        imageX = imageO
        #run the text_detector function defined above on the image 
        orig = text_detector(imageO, image_name)
        #Save the image in the output folder
        cv2.imwrite("Output/images/"+image_name+"_EAST.jpeg", orig) 
    
    print("\nGood news, it looks like we found some text! Whether it is interpretable is a question for you to decide...") 
    print("\nYou can view your csv files in Output/detected_text and the image files with the text detected in Output/images") 
    
    """
    -------------------------------------
    ---- Creating a merged dataframe ----
    -------------------------------------
    """
    print("\nNext we'll collect the image files into one big dataframe and clean up the output text")

    #define path to the csv files 
    csv_path = "Output/detected_text/"

    #create variable of the files 
    files = glob.glob(csv_path + "*.csv")

    #create an empty list to store the dataframes 
    dataframes = []

    for filename in files: 
        df = pd.read_csv(filename, index_col=None, header=0)
        dataframes.append(df)
    
    big_dataframe = pd.concat(dataframes, axis = 0, ignore_index=True)

    print("\nDataframes collected, now we'll remove unwanted characters and save the cleaned text into a new column")
    
    """
    -------------------------------------
    ---- Cleaning up the text column---- 
    -------------------------------------
    """
    
    # We'll clean up the text in the text column by removing punctuation and unwanted characters 
    #[NOTE] The user may want to hash out the second line if they feel punctuation would be meaningful to the translation
    
    pattern = "\n"

    big_dataframe['Cleaned Text'] = big_dataframe['Text'].map(lambda x: re.sub(pattern, '', x))             #take away the "\n" 
    big_dataframe['Cleaned Text'] = big_dataframe['Cleaned Text'].map(lambda x: re.sub(r'[^\w\s]', '', x))  #remove punctuation 

    print("\nFinal step: we'll run a spell check on the cleaned text column to see if we can make sense of the output")
    
    """
    ---------------------------------
    ---- Performing a spellcheck ----
    ---------------------------------
    """

    #We'll run a simple spell checker on the column
    spell = SpellChecker()
    corrected_words = []
    
    #for each line in the Cleaned Text column of the dataframe
    for line in big_dataframe["Cleaned Text"]: 
        #create a spellcheck prediction 
        line = spell.correction(line)
        #add the corrected words into the list 
        corrected_words.append(line)
    
    #add the corrected_words list onto the dataframe as a new columns 
    big_dataframe["Spell Checked"] = corrected_words


    #Finally, save it as a csv file in the main Output directory 
    big_dataframe.to_csv("Output/Text_Results.csv", index = False)

    print("That's you all complete, head over to the Output folder to check our your results!") 

        
# Close the main function 
if __name__=="__main__":
    main()  
