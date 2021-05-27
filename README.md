[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![describtor - e.g. python version](https://img.shields.io/badge/Python%20Version->=3.6-blue)](www.desired_reference.com) ![](https://img.shields.io/badge/Software%20Mac->=10.14-pink)

# Optical Character Recognition

## Converting handwritten letters from WW1 into text files... can it be done? 

<div align="center"><img src="https://github.com/Orlz/CDS_Visual_Analytics/blob/main/Portfolio/poppy.png"/></div>

Optical character recognition (OCR) refers to the process of detecting words within an image and converting this into encoded text which a computer is able to recognise. It is not a new idea and has been a topic of conversation since the mid 20th century, when industries were trying to plan how to make the big conversion towards technology. OCR is however an interesting area of language analytics as it still has such a long way to go. While other language tools have been able to adapt and learn complex structures such as grammatical language and voice recognition, we still lag behind having accurate tools to translate handwritten text. 

This assignment therefore returns to the topic of old war documents and considers why so few of these exist in text recognisable forms. We look first at what steps are needed to convert images of text into a format the computer will be able to recognise, and then turn to consider how one could go about building a more standardised pipeline for the process. 


## Table of Contents 

- [Assignment Description](#Assignment)
- [Background Information](#Background)
- [Methods](#Methods)
  * [Part 1](#Part-1)
  * [Part 2](#Part-2)
- [Operating the Scripts](#Operating)
  * [Data Collection](#Modelling)
  * [Modelling the CNN](#Data)
- [Discussion of results](#Discussion)

## Assignment Description

_This assignment is the self-assignment from the Language Analytics Summer 2021 course._ 

**Using Optical Character Recognition on historical documents** 

In this assignment, we zoom into the problems surrounding converting handwritten documents into text files. The assignmentis tackled with a two-step approach: firstly it considers what steps need to be taken to get images of old historical images into a format where they can be most easily interpreted by an OCR method. We then zoom out to consider what OCR methods are available to us, asking the question of "how could we go about converting these documents in a standardised way?". 


Due to the diverging nature of the two steps, one zooming in and the other zooming out, two resources have been prepared to support the assignment: 
i)	Part 1: A notebook file (OCR_Image_Processing.ipynb)  
ii)	Part 2: A python script (text_extraction_model.py) 

***Purpose of the assignment:***

This assignment was designed to show knowledge of: 
1.	How to integrate language analytics tools into a real-world scenario  
2.	How to perform optical character recognition on handwritten documents 
3.	How to compare and consider the specific needs and qualities of different neural network approaches  


## Background Information 

Have you ever heard the phrase _"It was nothing to write home about"?_ 

<div align="center"><img src="https://github.com/Orlz/CDS_Visual_Analytics/blob/main/Portfolio/mail.png"/></div>

It's a phrase which is usually used to describe something small and unremarkable, not worthy of 'writing home about'. Yet if we take this phrase and place it into the context of the days when writing home was the only form of communication one could use, such as during the First World War, then it would seem that the content of letters during these times could hold some very valuable information. They'd tell stories which people thought were worthy of writing home about, and perhaps give us a raw insight into the thoughts, hopes, and fears of people during these times. 

Luckily for anyone interested in the topic, we know that a lot of letters were sent during the four years of World War I (1914 - 1918). Records show that the British Army Postal Service delivered somewhere in the region of 2 billion letters in this time, and as they are only one nation in a war of many nations, we can assume that other national postal services were delivering just as many. This begs the question of where have all the letters gone? They're not the kind of documents which families would throw away. Yet, a quick internet search will soon reveal that it is surprisingly difficult to find much big data on the topic of soldier communication during the war. This seems like a real loss for history, and inspired me to consider why this might be and how language analytics could play its part in fixing the problem. 

What seems to be the reason for such a lack of digital resources is the difficulty in translating these handwritten documents into a text format. To type them up by hand would be a time consuming job and would require a small number of people to have access to large numbers of the documents. Yet the tools we have available don't seem to be doing the job. Imagine if this could be different, and that there was software which could take an image of these old documents and translate the words into a text file which could be used for further analysis. The thought of such a tool would open the possibility of collecting these documents en masse and exploring the text held within. If the process was made easy enough, I'm sure we'd be on our way to having some very valuable language analysis in our hands. 

This assignment therefore poses as an exploratory venture into optical character recognition and how we could use modern day tools to bridge the gap. I hope you enjoy the ride! 

## Methods 

This assignment was conducted on the following six war-time images: 

1. A handwritten poem (the original 'In Flanders Fields')
2. A handwritten letter 
3. A handwritten note
4. A handwritten luggage tag 
5. A letter typed from the typewriter of the time 
6. A newspaper cover 

The images were selected to explore how OCR would manage on each of the different styles and give us a better idea of what kind of documents we would benefit from trying OCR on first. These images can be viewed in the 'data' folder. 

## Part-1

The first stage of the analysis was to get a gauge of how well OCR tools would work on documents with different imaging techniques applied to them. Pytesseract was selected to conduct the text recognition, due to its high performance in other areas of text recognition and easy integration with Python. Tesseract is an open-souce OCR model which was one of Google's baby's. It has been vastly used in other text recognision tasks with high levels of accuracy. The model works best on printed fonts and is known to struggle with handwriting, however I was interested to see how much it struggled and if there was a way to make the handwritng more recognisable. More information on tesseract and the way it works can be found [here](https://github.com/tesseract-ocr/tesseract) and in the written hand-in. 

To test whether ammending the images would help the model to recognise text, a number of functions available in OpenCV's image processing package were applied to the image. These included experimenting with converting images to greyscale, applying binary thresholding, reducing noise, dilating and eroding the text, opening up the text to be bolder in style, outlining the text to enlarge the font size, applying canny edge detection and de-skewing to the images. Each of these methods are well documented through the notebook with examples of what each does to the image. The reader is encouraged to look through and see how each function transforms the image. With each amendment to the image, tesseract’s “image_to_string()” function was used to see how much text could be extracted from the images. 

These are then compared and tested on new images to see how well the techniques generalise across the images.  This part of the assignment is to be seen as an exploration to expose the issues with translating handwriting and also to give us a better idea of what elements in an image are being picked up to help with the text recognition. 

## Part-2

The second stage of the assignment asks the question “How could we systematically detect handwriting and convert it into text files which could be used for language analysis?”. It is a response to the recognition that image processing is a good technique to extract the text from a single image, but it is time consuming and individualistic meaning that we won’t be building up our datasets fast if we’re processing every image on its own. 

The script employs two neural networks to help with the task of converting handwriting into text files. The first is a pre-trained neural network named [EAST](https://arxiv.org/abs/1704.03155v2) (an Efficient and Accurate Scene Text Detector) which is used to detect where text is in the image. It does this by calculating confidence probabilities of whether text is at a particular coordinate location and if it deemed to be confidence enough, the region will be identified as a region of text. A green rectangle will then be drawn around the area and the coordinate location of this will be recorded. 

The second neural network is of course tesseract, as used in part 1. Here, pytesseract is directed towards the green rectangles detected by EAST and tries to conduct text recognition on these smaller chunks of writing. This way, the technology is given a much more precise region to try to extract text from. To get the best results, pytesseract’s parameters are tweaked to assume the writing is going to be a line of text (by setting the ‘psm’ to 7). The information from both networks (the coordinates and confidence values from EAST, and the text detected from pytesseract) are stored into a dataframe for each image, where each row represents a separate row of text. 

The final stage in the script takes the dataframes from each individual image and compiles them together into one overall dataframe, which can be found in the output folder as ‘Text_Results”. From here, it tries to clean up the text column by removing unwanted punctuation and conducting a simple spell check. It should be noted that tesseract has it’s own spellcheck built into the model, but as this didn’t seem to output sensible words, a second spell check was applied in the hope it could bring some more meaning to the text. The output file is therefore a dataset containing the image name, the various variations of the text which has been able to be detected, and their coordinate location. 

It is recommended the reader looks through the python script for detailed explanations of each step taken through the process. 


## Operating the Scripts 

_The reader is recommended to first look through the notebook, which is located in the src folder._ 

There are 3 steps to get the script up and running: 

**1. Clone the repository**

The easiest way to access the files is to clone the repository from the command line using the following steps 

```bash
#clone repository
git clone https://github.com/Orlz/Optical_Character_Recognition.git

```


**2. Create the virtual environment**

You'll need to create a virtual environment which will allow you to run the script using all the relevant dependencies. This will require the requirements.txt file attached to this repository. 


To create the virtual environment you'll need to open your terminal and type the following code: 

```bash
bash create_virtual_environment.sh
```
And then activate the environment by typing:

```bash
$ source language_analytics05/bin/activate
```

**3. Run the script** 

With the repository cloned and the virtual encironment activated, the following command will run the script: 


```bash
$ python3 text_extraction_model.py
```

Three optional parameters are available to be used:  

```
Letter call  | Parameter              | Required? | Input Type     | Description                                         
-----------  | -------------          |--------   | -------------  |                                                     
`-m`         | `--minimum_confidence` | No        | Float          | Minimum confidence probability for text recognition                   
`-h`         | `--height`             | No        | Integer        | The height dimension for the resized image 
`-w`         | `--width`              | No        | Integer        | The width dimension for the resized image            
```


## Discussion of Results 

Unsurprisingly, the task of converting our old war documents into text files has been somewhat unfruitful in its efforts. Our script was able to successfully run through the images, identify text, and extract it into a dataframe. However, upon inspection, the words which have been scraped are more often meaningless characters strung together. Even articles which should have been easier to translate, such as the newspaper or the neatly written note, have returned quite poor textual translations. This is a problem, because it shows that we are not yet close to being able to extract meaningful text from images of this quality, with these tools. However, what we can take from this exploration is direction for where we can start to build the tools. 

Comparing the notebook's text results to that of the script is a good place to start. Here, we see that with some individual image manipulation, we can highlight the text to be more recognisable to a computer. In this state, tesseract is able to detect significantly more words. The handwritten tag provides a good example of this: in the notebook with just a greyscale function applied to the image, tesseract is able to detect 5 of the 8 words on the tag. That's around 63% accuracy. By comparison, our script was not able to detect any of the words in this image. This tells us two things: firstly that applying a greyscale filter is advantageous for the text recognition (the EAST detector cannot take greyscale images as it applies its own filter to the image which needs to use the three colour channels of the image). Secondly, it appears that tesseract works better on images when it is considering the text as a unit of text (with psm = 6) rather than considering each line at a time (with psm =7). This may indicate that the network is using its knowledge of the surrounding words to make its predictions in our notebook, and is not able to do this with the approach we take in the script. We can consider how the two images were recognised below: 

**Left =  greyscaled image,  Right = image after passing through EAST detection**

 <img src="https://github.com/Orlz/CDS_Visual_Analytics/blob/main/Portfolio/greyscaled_tag.png" alt="alt text" width="400" height="300">  <img src="https://github.com/Orlz/Optical_Character_Recognition/blob/main/Output/images/handwritten_tag_EAST.jpeg" alt="alt text" width="400" height="300"> 
 
 The problem appears to lie with tesseracts ability to recognise the text. The EAST detector is able to find the text regions and isolate the words, but tesseract just struggles to make any sense of them. Perhaps this indicates that to tackle this challenge, we need to be using a neural network which is pre-trained on handwriting, instead of the vast range of images tesseract was trained on. This is not such an easy transition to make, as to build suck a network would require a lot of manual labour to collect enough handwritten images with enough different styles to train the model. Luckily for future researchers, there's a team who are already looking into this! Harald Scheidel and his team have been working on developing such a model over the past number of years using tensorflow, and offer a great next step for a project like this! you can read more about the model and its developments [here](https://github.com/githubharald/SimpleHTR). 
 
To bring our explorations to a conclusion, we can say that it is by no coincidence that our beloved war documents have not been translated into a text file format. Indeed, many of the documents are hard for a human to make sense of, never mind a computer! However, with better models and more training on handwritten content, it is likely that we could overcome this challenge. EAST demonstrated how easily it can find the regions containing text, and with a bit of manipulating we could easily make such tools work for this context. For example, we could image using an EAST style text detector alongside a more adaptable OCR to extract newspaper headlines from the time and compile a dataset similar to the one we used in the sentiment analysis assignment (albeit with many less headlines than a million!). 

I believe pursuing this challenge is one that will pay off and look forward to the future of language analytics working hand in hand with history! 

I hope you've enjoyed the portfolio! 

___Teaching credit___ Many thanks to Ross Deans Kristiansen-McLachlen for providing an interesting and supportive venture into the world of Language Analytics! 

<div>Icons made by <a href="https://www.freepik.com" title="Freepik">Freepik</a> from <a href="https://www.flaticon.com/" title="Flaticon">www.flaticon.com</a></div>
