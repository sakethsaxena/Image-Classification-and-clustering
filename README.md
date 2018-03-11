# K-means classfication of images of headshots and landscapes 
# KNN and Heirarchical clustering of images based on euclidean distance

![San Jose State University](https://i.imgur.com/cShW5MA.gif?1)
![..](https://i.imgur.com/QIGOoLy.png?1)


A script developed to classify images using a KNN classifier and perform K-means clustering and single linkage clustering on image datasets.

The datasets used are self curated images of "professional headshots" and "landscape images"

The KNN classifier uses a lookup table generated from training the model on a labeled dataset 

The clustering algorithms cluster similar data together 

The script use the euclidean distance as the similarity measure to cluster and classifiy images.
_________________________________________________________________________________________________

Take a look at the report.pdf file to see the performance results 

_________________________________________________________________________________________________



Go through following instructions before running the code:
•	The program is run by the following command
  
  o	`python Image_Classification_Clustering`
  
•	External libraries to be installed - 
  
  o	`PIL, matplotlib`

•	On first run of the program - please select option 2 for training component and provide the directory paths for landscape and headshot 
    files when prompted, this is to be done to update the location of images in the lookuptable, otherwise while clustering the wrong 
    location of image will be looked up and the code will exit with error
    
•	The datasets for flags, headshots, landscapes and RnB(3 -red and 3 - black) images are in the “dataset” folder

•	For hierarchical clustering by flag - you will require a directory path pointing towards location of flag images. 

•	Ensure that the directory paths being given as input do not contain files of any other format or sub directories.

•	Hierarchical clustering opens a web browser to show the clusters 

•	Cross fold validation takes time if the number of images is more than 500, please be patient, in such a case.

