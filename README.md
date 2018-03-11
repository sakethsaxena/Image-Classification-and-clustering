Go through following instructions before running the code:

•	The program is run by the following command
  o	python Image_Classification_Clustering
•	external libraries to be installed - 
  o	PIL, matplotlib

•	On first run of the program - please select option 2 for training component and provide the directory paths for landscape and headshot 
    files when prompted, this is to be done to update the location of images in the lookuptable, otherwise while clustering the wrong 
    location of image will be looked up and the code will exit with error
•	The datasets for flags, headshots, landscapes and RnB(3 -red and 3 - black) images are in the “dataset” folder
•	For hierarchical clustering by flag - you will require a directory path pointing towards location of flag images. 
•	Ensure that the directory paths being given as input do not contain files of any other format or sub directories.
•	Hierarchical clustering opens a web browser to show the clusters 
•	Cross fold validation takes time if the number of images is more than 500, please be patient, in such a case.
•	Running hierarchical clustering on 3 red and 3 black images returns the following output:
 
