## CROHME Data Extractor


This is a series of scripts for extracting and analyzing the CROHME dataset of handwritten math symbols. Images are in a format (InkXML) that allows them to be scaled and generated at any size so that they can be a drop-in for many datasets, including MNIST. Image extraction settings can be easily configured to make the images work better with certain types of models (e.g. CNNs). 

This is a slightly modified version of the extractor found [here](https://github.com/ThomasLech/CROHME_extractor). The main modifications were using PIL.ImageDraw instead of scikit-image to draw lines, so the extract script now draws realistic lines of a desired thickness (default 3px), and also all of the output images are now saved to a folder instead of a pickle binary. The point of drawing realistic lines is that some models, especially CNNs, seem to do poorly when lines have no thickness (no edges to detect!). With images extracted from this modified version, I am able to run a AC-GAN model to conditionally generate new math symbols, whereas I was not able to do this with the original extractor.

## Setup
Python version: **3.5**.

1. Extract **_CROHME_full_v2.zip_** (found inside **_data_** directory) contents before running any of the above scripts.

2. Install specified dependencies with pip (Python Package Manager) using the following command:
``` bash
pip install -U -r requirements.txt
```


## Scripts info
1. **_extract.py_** script will extract **square-shaped** bitmaps.  
With this script, you have control over data being extracted, namely:
    * Extracting data belonging to certain dataset version.
    * Extracting certain categories of classes, like **digits** or **greek** (see categories.txt for details).
    * Extracting images with line strokes drawn of any desired thickness (default 3px)
    
    **Usage**: `python extract.py <out_format> <box_size> <dataset_version=2013> <category=all>`

    **Example usage**: `python extract.py pixels 32 2011+2012+2013 digits+operators+lowercase_letters+greek`

2. **_visualize.py_** script will plot single figure containing a random batch of your **extracted** data.

    **Usage**: `visualize.py <number_of_samples> <number_of_columns=4>`

    **Example usage**: `python visualize.py 40 8`

    **Plot**:
    ![crohme_extractor_plot](https://user-images.githubusercontent.com/22115481/30137213-9c619b0a-9362-11e7-839a-624f08e606f7.png)

3. **_extract_hog.py_** script will extract **HoG features**.  
This script accepts 1 command line argument, namely **hog_cell_size**.  
**hog_cell_size** corresponds to **pixels_per_cell** parameter of **skimage.feature.hog** function.  
We use **skimage.feature.hog** to extract HoG features.  
Example of script execution: `python extract_hog.py 5`  <-- pixels_per_cell=(5, 5)  
This script loads data previously dumped by **_parse.py_** and again dumps its outputs(train, test) separately.


4. **_extract_phog.py_** script will extract **PHoG features**.  
For PHoG features, HoG feature maps using different cell sizes are concatenated into a single feature vector.  
So this script takes arbitrary number of **hog_cell_size** values(HoG features have to be previously extracted with **_extract_hog.py_**)  
Example of script execution: `python extract_phog.py 5 10 20` <-- loads HoGs with respectively 5x5, 10x10, 20x20 cell sizes.


5. **_histograms_** folder contains histograms representing **distribution of labels** based on different label categories. These diagrams help you better understand extracted data.


## Distribution of labels
![all_labels_distribution](https://cloud.githubusercontent.com/assets/22115481/26694312/413fb646-4707-11e7-943c-b8ecebd0c986.png)
Labels were combined from **_train_** and **_test_** sets.
