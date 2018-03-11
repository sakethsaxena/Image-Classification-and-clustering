import shelve
try:
    from PIL import Image
except:
    print "please install PIL package"
    exit()
from os import listdir
from os.path import isfile, join
from operator import itemgetter
from collections import Counter
from random import shuffle, randint, sample
try:
    import matplotlib.pyplot as plt
except:
    print "Please install matplotlib"
    exit()
import sys
try:
    import webbrowser
except:
    print "Please install webbrowser packcage"
    exit()

# The environment class 
# takes input from user and passes on to the agent
# takes output from agent, does post-processing and returns output to user
class Environment:

    # initialize input
    def __init__(self, mode):
        self.mode = mode
        try:
            self.mode = int(self.mode)
        except:
            print("\n Please enter a valid integer for your menu options")
            return

        if self.mode  not in [1,2,3,4,5,6]:
            print "Please rerun the program and input choices in range [1,2]"
            return
        else:
            
            # if classification of image chosen
            if self.mode == 1:
                self.image_path = raw_input("Please enter image path:\n")
                

                # checking if image path is valid
                try:
                    im = Image.open(self.image_path)
                except:
                    print "Image not found, please enter valid path"
                    return
                
                self.knn_k = raw_input("Please enter k value for k nearest neighbours:\n")
                
                try:    
                    self.knn_k = int(self.knn_k)
                except:
                    print "Please enter valid integer value for k"
                    return

                if self.knn_k == 0:
                    print "K cannot be zero"
                    return
                
            if self.mode == 2:
                self.imageDirectoryPaths = ['','']
                self.imagesList = {}

                # user input for path to landscape images directory
                self.imageDirectoryPaths[0] = raw_input("Please Enter the directory path for landscape photos")

                # preprocessing the path to check if its empty and check for trailing slash
                
                self.imageDirectoryPaths[0] = self.preprocesspaths(self.imageDirectoryPaths[0])
                
                
                
                # getting image files from the path, also checks if the directory is empty or does not exist 
                self.imagesList['landscapes'] = self.getimagefiles(self.imageDirectoryPaths[0])
                
                
                
                # user input for path to headshot images directory
                self.imageDirectoryPaths[1] = raw_input("Please Enter the directory path for headshot photos")
                
                self.imageDirectoryPaths[1] = self.preprocesspaths(self.imageDirectoryPaths[1])
                
                self.imagesList['headshots'] = self.getimagefiles(self.imageDirectoryPaths[1])
                
            # if cross validation chosen            
            if self.mode == 3:
                
                self.knn_k_cross = raw_input("Please enter k value for k nearest neighbours:\n")
                
                try:    
                    self.knn_k_cross = int(self.knn_k_cross)
                except:
                    print "Please enter valid integer value for k"
                    return

                if self.knn_k_cross == 0:
                    print "K cannot be zero"
                    return
            # If heirarchical clustering chosen
            if self.mode == 5:
                self.flag = raw_input("Enter \n1. To cluster headshots and landscapes \n2. For flags\n")
                try:
                    self.flag = int(self.flag)
                except:
                    print "Please input valid choice - 1 or 2"
                    return
                
                if self.flag not in [1,2]:
                    print "Please input valid choice - 1 or 2"
                    return

                if self.flag == 2:
                    self.flag_directory = raw_input("Please enter directory path where the images are located")
                    self.flag_directory = self.preprocesspaths(self.flag_directory)
                    self.hImage_list = self.getimagefiles(self.flag_directory)
                    
                
            self.comp_env()
    
    
    
    # Checking if path entered is empty
    # appending trailing slash if path does not contain one
    def preprocesspaths(self,im_path):
        if im_path == '':
            print "Please rerun the program and enter valid path.\n"
            exit()

        if not im_path.endswith('/'):
            im_path = im_path+'/'
        return im_path
    

    # Used by trainer to return list of images in directory
    def getimagefiles(self,im_path):
        try:
            imagefiles =  [im_path+f for f in listdir(im_path) if isfile(join(im_path, f))]
            if imagefiles == []:
                print "No files were found in the directory, please check and input proper directory."
                exit()
            return imagefiles
        except:
            print "No such directory found!\n"+im_path +"\nPlease rerun the code and  enter a valid directory path.\n"
            exit()

    # Training Component
    def trainer(self):
        # Extracts the features from the images and stores them into shelve db
        def featureCollector(im_list,class_val):
            db = shelve.open('lookuptable')
            db[class_val] = []
            temp_list = []        
            for imagefile in im_list:
                im = Image.open(imagefile)
                # Resizing the image
                im = im.resize((32,32))
                # Raw pixel Intensities
                raw_pixel_feature = []
                for featuretup in list(im.getdata()): 
                    for feature in featuretup:
                        raw_pixel_feature.append(feature)
                raw_pixel_feature.append(class_val)
                raw_pixel_feature.append(imagefile)
                temp_list.append(raw_pixel_feature)
            db[class_val] = temp_list
            db.close()
        
        # Sending images to feature collector
        for class_val, im_list in self.imagesList.items():
            featureCollector(im_list, class_val)
    




    # The controller for Environment class
    # Takes input from user
    # sends percepts to the sensor 
    # Recieves output from actuator
    # Post processes and returns the output to the user
    # Performs heirarchical clustering, k-means clustering  
    def comp_env(self):
        
        # To do KNN classification
        if self.mode ==1:
            agentIns =  Agent()
            agentIns.sensor(self.image_path,0,self.knn_k,self.mode)
            k_nearest_neighbours = agentIns.actuator()

            if len(k_nearest_neighbours) == 1:
                print "Your image is classified as "+k_nearest_neighbours[0][1]
            else:
                class_labels = []
                for value in k_nearest_neighbours:
                    class_labels.append(value[1])
                class_labels = Counter(class_labels).most_common(1)

                print "Classification completed\n"
                print "Your image is in the class of "+str(class_labels[0][0])+"\n"
                return
                
        # To train the model with new dataset
        # this mode rewrites the current lookuptable
        if self.mode == 2:
            self.trainer()
            print "\nTraining Finished!\n\n"

        # 3-fold cross validation
        if self.mode == 3:
            lookup_shelve = shelve.open('lookuptable')
            dataset_list = []
            fold_1 = []
            fold_2 = []
            fold_3 = []
            # First we generate a list of feature vectors with class label appended at the end
            print "Generating dataset.."
            for class_val in lookup_shelve.keys():
                for feature_vector in lookup_shelve[class_val]:
                    dataset_list.append(feature_vector)
            lookup_shelve.close()
            print "Shuffling the dataset..."
            shuffle(dataset_list)
            print "Partitioning dataset...."
            count_landscape = 0
            count_headshot = 0

            for feature_vector in dataset_list:
                if feature_vector[-2] == "landscapes":
                    if count_landscape == 0:
                        fold_1.append(feature_vector)
                        count_landscape+=1
                    elif count_landscape == 1:
                        fold_2.append(feature_vector)
                        count_landscape+=1
                    elif count_landscape == 2:
                        fold_3.append(feature_vector)
                        count_landscape = 0

                if feature_vector[-2] == "headshots":
                    if count_headshot == 0:
                        fold_1.append(feature_vector)
                        count_headshot+=1
                    elif count_headshot == 1:
                        fold_2.append(feature_vector)
                        count_headshot+=1
                    elif count_headshot == 2:
                        fold_3.append(feature_vector)
                        count_headshot = 0
                
            print "Begining Validation..."
            print "Validating fold 1"
            
            training_set = fold_2+fold_3
            accuracy1 = self.validator(fold_1,training_set,self.knn_k_cross)
            print "Accuracy for fold 1 is : "+str(accuracy1)+"%"
            
            print "Validating fold 2"
            training_set = fold_1+fold_3
            accuracy2 = self.validator(fold_2,training_set,self.knn_k_cross)
            print "Accuracy for fold 2 is : "+str(accuracy2)+"%"
            
            print "Validating fold 3"
            training_set = fold_1+fold_2
            accuracy3 = self.validator(fold_3,training_set,self.knn_k_cross)
            print "Accuracy for fold 3 is : "+str(accuracy3)+"%"
            print "\n"
            
            # bar plot for accuracies for k as user input,
            # plotted at the end
            names = ['Fold_1', 'Fold_2', 'Fold_3']
            values = [accuracy1, accuracy2, accuracy3]
            plt.figure(1, figsize=(20,7))
            plt.subplot(121)
            bar1 = plt.bar(names, values)
            plt.xlabel("Validation datasets")
            plt.ylabel("Accuracy %")
            plt.title("Accuracies for 3-fold cross validation with k = "+str(self.knn_k_cross))

            for rect in bar1:
                height = rect.get_height()
                plt.text(rect.get_x() + rect.get_width()/2.0, height, '%s%s' % (height,"%"), ha='center', va='bottom')
            plt.draw()
            accuracy_dict = {}
            accuracy_dict[self.knn_k_cross] = round((accuracy1+accuracy2+accuracy3)/3, 2)
            print "Average accuracy for 3 cross fold validation with k = "+str(self.knn_k_cross)+" is : "+str(accuracy_dict[self.knn_k_cross])+"%\n"
            print "The data will be plotted when the following step is completed"

            # 3 fold cross validation by varying k between 1-10
            print "Performing 3 fold cross validation by varying k between 1-10"

            for i in range(1,11):
                if i == self.knn_k_cross:
                    continue
                else:
                    print "Running 3 crossfold validation for k="+str(i)
                    training_set = fold_2+fold_3
                    accuracy1 = self.validator(fold_1,training_set,i)

                    training_set = fold_1+fold_3
                    accuracy2 = self.validator(fold_2,training_set,i)

                    training_set = fold_1+fold_2
                    accuracy3 = self.validator(fold_3,training_set,i)
                    average_accuracy = (accuracy1+accuracy2+accuracy3)/3
                    accuracy_dict[i] = round(average_accuracy,2)
                    print "Accuracy for k = "+str(i)+" is "+str(accuracy_dict[i])+"%"

            max_accuracy = max(accuracy_dict.iteritems(), key=itemgetter(1))
            print "\nThe maximum accuracy is "+str(max_accuracy[1])+" for k="+str(max_accuracy[0])

            
            # Generate graph for varying k against accuracy
            x_val = accuracy_dict.keys()
            y_val = accuracy_dict.values()
            plt.subplot(122)
            plt.plot(x_val,y_val)
            plt.ylabel("Accuracy %")
            plt.xlabel("K - Value")
            plt.title("Accuracy for varying k value between 1-10")
            plt.show()

        # printing accuracies for k-means
        if self.mode == 4:
            accuracy_1, accuracy_2, total_accuracy, cluster_1, cluster_2, centroid_1_class,centroid_2_class = self.k_means_clustering()
            print "\n"
            print "Accuracy for cluster 1 ("+centroid_1_class+") : "+str(accuracy_1)+"%"
            print "Accuracy for cluster 2 ("+centroid_2_class+") : "+str(accuracy_2)+"%"
            print "\n"
            print "Overall Accuracy for k- means with k = 2 : "+str(total_accuracy)+"%"
            print "\n"


        # perform heirarchical clustering
        if self.mode == 5:
            # Do Clustering on landscape and headshots
            if self.flag == 1:
                lookup_shelve = shelve.open('lookuptable')
                dataset_list = []
                
                # Flattening dataset
                for class_val in lookup_shelve.keys():
                    for feature_vector in lookup_shelve[class_val]:
                        dataset_list.append(feature_vector)
                lookup_shelve.close()

                self.cluster = self.Heirarchical_clustering(dataset_list)
                
            
            # To do Clustering on flags
            if self.flag == 2:
            
                dataset_list = []        
                # Flattening flag dataset

                for imagefile in self.hImage_list:
                
                    im = Image.open(imagefile)
                
                    # Resizing the image
                    im = im.resize((32,32))
                
                    # Raw pixel Intensities
                    raw_pixel_feature = []
                    for featuretup in list(im.getdata()): 
                        for feature in featuretup:
                            raw_pixel_feature.append(feature)
                    raw_pixel_feature.append(0)
                    raw_pixel_feature.append(imagefile)
                    dataset_list.append(raw_pixel_feature)
                self.cluster = self.Heirarchical_clustering(dataset_list)
            

            # show result in web browser
            htmlcontent=""
            for idx, row in enumerate(self.cluster):
                print "Cluster "+str(idx+1)+" contains the following images "+str(row)
                htmlcontent += "<div style='border:1px solid black;height:70vh;width:40 vw;overflow-y:auto;'><h1>Cluster "+str(idx+1)+" gives the following images</h1>"
                for val in row:
                    htmlcontent+= "<img src='"+dataset_list[val][-1]+"' style='padding:20px' height='100' breadth='100'>"
                htmlcontent += "</div>"
            htmlfile = open('index.html','w')
            htmlfile.write(htmlcontent)
            webbrowser.open('index.html')




    # used by cross validation
    # Predicts the class of validation set and returns the accuracy of agent for each dataset
    def validator(self,validation_set,training_set, k):
        
        agentIns = Agent()
        correct_predictions = 0
        
        for feature_vector in validation_set:
                        
            agentIns.sensor(feature_vector,training_set,k,self.mode)
            k_nearest_neighbours = agentIns.actuator()
            predicted_class = []
            actual_class = feature_vector[-2]

            if len(k_nearest_neighbours) == 1:    
                predicted_class = k_nearest_neighbours[0][1]
                
            else:
                for value in k_nearest_neighbours:
                    predicted_class.append(value[1])

                predicted_class = Counter(predicted_class).most_common(1)
                predicted_class = predicted_class[0][0]
            
            if predicted_class == actual_class:
                correct_predictions += 1
        accuracy = round(float(correct_predictions)/len(validation_set),4)*100
        return accuracy

    # Returns two clusters after k-means clustering
    def k_means_clustering(self):
        lookup_shelve = shelve.open('lookuptable')
        
        centroids = []
        dataset_list = []

        # Flattening dataset
        for class_val in lookup_shelve.keys():
            for feature_vector in lookup_shelve[class_val]:
                dataset_list.append(feature_vector)
        lookup_shelve.close()
        
        # select initial seeds
        centroids_vectors = sample(dataset_list, 2)
        

        # To make sure that both categories of centroids are selected
        while True:
            if centroids_vectors[0][-2] == centroids_vectors[1][-2]:
                centroids_vectors = sample(dataset_list, 2)
            else:
                break

        # index values of initial centroid
        centroid_id = []
        for centroid in centroids_vectors:
            centroid_id.append(dataset_list.index(centroid))

        cluster_1 = [centroid_id[0]]
        cluster_2 = [centroid_id[1]]   
        
        centroid_1_class = centroids_vectors[0][-2]
        centroid_1 = centroids_vectors[0]
        centroid_2_class =  centroids_vectors[1][-2] 
        centroid_2 = centroids_vectors[1]
        
        agentIns = Agent()
        for idx, feature_vector in enumerate(dataset_list):
            distance = []
            if idx in centroid_id:
                continue
            else:
                # perform clustering for initial centroid
                for centroid in centroid_id:
                    distance.append([agentIns.euclidean_distance(feature_vector[:-2],dataset_list[centroid][:-2]),feature_vector[-2],idx])
   
                if distance[0][0] < distance[1][0]:
                    cluster_1.append(distance[0][2])
                elif distance[0][0] > distance[1][0]:
                    cluster_2.append(distance[1][2])        
        i = 1
        while True:
            # Calculate new centroid 1
            print "Iteration no : "+str(i)
            sum_list = [0]*len(dataset_list[0][:-2])
            for index_val in cluster_1:
                for idx, value in enumerate(dataset_list[index_val][:-2]):
                    # print value
                    sum_list[idx]+=value
            new_centroid_1 =[x/len(cluster_1) for x in sum_list] 
            
            # Calculate new centroid 2

            sum_list = [0]*len(dataset_list[0][:-2])
            for index_val in cluster_2:
                for idx, value in enumerate(dataset_list[index_val][:-2]):
                    sum_list[idx]+=value
            new_centroid_2 =[x/len(cluster_2) for x in sum_list] 
            
            distance1 = agentIns.euclidean_distance(new_centroid_1,centroid_1)
            distance2 = agentIns.euclidean_distance(new_centroid_2,centroid_2)

            # Stopping if the distance between new and old centroids don't change
            if distance1 < 0.01 and distance2 < 0.01: 
                all_correct_predictions = 0
                correct_predictions = 0               
                for index_val in cluster_1:
                    if dataset_list[index_val][-2] == centroid_1_class:
                        all_correct_predictions += 1
                        correct_predictions+=1
                accuracy_1 = round(float(correct_predictions)/len(cluster_1),4)*100
                correct_predictions = 0               
                for index_val in cluster_2:
                    if dataset_list[index_val][-2] == centroid_2_class:
                        correct_predictions+=1
                        all_correct_predictions += 1
                accuracy_2 = round(float(correct_predictions)/len(cluster_2),4)*100
                total_accuracy = round(float(all_correct_predictions)/(len(cluster_1)+len(cluster_2)),4)*100
                return accuracy_1, accuracy_2, total_accuracy, cluster_1, cluster_2, centroid_1_class, centroid_2_class
            # Computing new clusters 
            else:
                centroid_1 = new_centroid_1
                centroid_2 = new_centroid_2
                cluster_1 = []
                cluster_2 = []
                i+=1
                for idx, feature_vector in enumerate(dataset_list):
                    distance = []
                    for centroid in [centroid_1, centroid_2]:
                        distance.append([agentIns.euclidean_distance(feature_vector[:-2],centroid),feature_vector[-2],idx])
                    
                    if distance[0][0] < distance[1][0]:
                        cluster_1.append(distance[0][2])
                
                    elif distance[0][0] > distance[1][0]:
                        cluster_2.append(distance[1][2])        

                    

    # Performs Heirarchical clustering of dataset 
    # returns two clusters
    def Heirarchical_clustering(self,dataset):
        
        self.distance_matrix = {}
        agentIns = Agent()
        i = 1

        # initializing distance matrix
        for idx, rowx in enumerate(dataset):
            self.distance_matrix[idx] = {}
            for idy, rowy in enumerate(dataset):
                if idx == idy:
                    continue
                self.distance_matrix[idx][idy] = -1

        min_value = sys.maxint
        cluster = [] 
            
        # generating distance matrix
        list_of_ids = [idx for idx, x in enumerate(dataset)]
        for idx, rowx in enumerate(dataset):
            print "Computed distance of image "+str(i)
            i+=1

            for idy, rowy in enumerate(dataset):
                if idx == idy:
                    continue
                
                elif self.distance_matrix[idy][idx] != -1:
                    continue

                if self.distance_matrix[idy][idx] == -1 or self.distance_matrix[idx][idy] == -1:
                    current_distance = agentIns.euclidean_distance(rowx,rowy)

                    # since we select the least value for the initial cluster
                    # picking out the least value pair while forming the distance matrix

                    if current_distance < min_value:
                        min_value = current_distance 
                        cluster = [[idx,idy]]
                    self.distance_matrix[idx][idy] = current_distance
                    self.distance_matrix[idy][idx] = current_distance
            
        # Removing the cluster values from list of IDs
        for row in cluster:
            for x in row:              
                list_of_ids.remove(x)    
    
        print "Clustering started"
        i=1
        least_value_ids_x = []
        least_value_ids_c = [] 
        least_value_ids_o = []

        while True:
            min_value_o = sys.maxint
            min_value_c= sys.maxint
            min_value_x = sys.maxint
            print "iteration no "+str(i)
            i+=1

            # if dataset has a single point left in the penultimate iteration return the point and the cluster 
            if len(list_of_ids) == 1 and len(cluster) == 1:
                cluster.append(list_of_ids)
                return cluster
                break
            # if dataset has no points and two clusters are formed return cluster
            elif list_of_ids == [] and len(cluster) == 2:
                return cluster
                break

            
            if list_of_ids != []:            
                min_value_o = sys.maxint
                least_value_ids_o = []
                # least distance in unclustered data
                for val in list_of_ids:
                    for val_x in list_of_ids:
                        if val == val_x:
                            continue
                        if min_value_o > self.distance_matrix[val][val_x]:
                            min_value_o = self.distance_matrix[val][val_x]
                            least_value_ids_o = [val,val_x]

                # least distance between points in cluster and unclustered data
                #  i.e. min(cluster,list_of_ids)
                least_value_ids_c = []
                for val in list_of_ids:
                    for row in cluster:
                        for val_x in row:                       
                            if val == val_x:
                                continue
                            if min_value_c > self.distance_matrix[val][val_x]:
                                min_value_c = self.distance_matrix[val][val_x]
                                least_value_ids_c = [val,val_x]
            
            # Least distance between the clusters
            min_value_x = sys.maxint
            least_value_ids_x = []
            for idx, row in enumerate(cluster):
                for idy, rowx in enumerate(cluster):
                    if idx == idy:
                        continue
                    for val in row:
                        for valx in rowx:
                            if self.distance_matrix[val][valx] < min_value_x:
                                min_value_x = self.distance_matrix[val][valx]
                                least_value_ids_x = [idx,idy]
                                    


            # if least value forms new cluster
            if min_value_o < min_value_c and min_value_o < min_value_x:
                cluster.append(least_value_ids_o)
                for value in least_value_ids_o:
                    list_of_ids.remove(value)
             
            # If existing cluster needs to be appeneded
            elif min_value_c < min_value_o and min_value_c < min_value_x:
                cluster_to_append = -1
                for idx, row in enumerate(cluster):
                    if least_value_ids_c[1] in row:
                        cluster_to_append = idx
                        break    
                cluster[cluster_to_append].append(least_value_ids_c[0])
                list_of_ids.remove(least_value_ids_c[0])

            # If existing clusters are merged
            elif min_value_x < min_value_c and min_value_x < min_value_o:
                temp_list = []
                temp_cluster = []
                smaller_id  = sys.maxint
                for idy in least_value_ids_x:
                    if idy < smaller_id:
                        smaller_id = idy
                    for value in cluster[idy]:
                        temp_list.append(value)
                    temp_cluster.append(cluster[idy])

                new_cluster = [x for x in cluster if x not in temp_cluster]
                new_cluster.append(temp_list)                
                cluster = new_cluster



# Agent function Performs knn classification
class Agent:
    

    # sensors takes precept sequences from the environment and
    # sends them to the agent fuction
    def sensor(self,image_path, training_set, knn_k,mode):
        self.agentfunction(image_path, training_set, knn_k, mode)
    

    # Agent function processes percept sequences and produces a list of k nearest neighbours
    def agentfunction(self, image_path, training_set, knn_k, mode):
        self.knn_list = []
        if mode == 1:
            im = Image.open(image_path)
            im = im.resize((32,32))

            # Raw pixel Intensities
            raw_pixel_feature = []
            for featuretup in list(im.getdata()): 
                for feature in featuretup:
                    raw_pixel_feature.append(feature)

            
            lookup_shelve = shelve.open('lookuptable')
            

            for class_val in lookup_shelve.keys():            
                for feature_vector in lookup_shelve[class_val]:
                    self.knn_list.append([self.euclidean_distance(feature_vector[:-2],raw_pixel_feature), class_val])

            lookup_shelve.close()
        
        if mode == 3:
            
            for feature_vector in training_set:
                self.knn_list.append([self.euclidean_distance(feature_vector[:-2],image_path[:-2]), feature_vector[-2]])
            
        self.knn_list = sorted(self.knn_list, key=itemgetter(0))[:knn_k]

    # sends the k nearest neighbours list to the environment
    def actuator(self):
        return self.knn_list

    # calculates euclidean distance between feature vectors of input image and dataset    
    def euclidean_distance(self,feature_vector, raw_pixel_feature):
        dist = 0
        feature_vector = feature_vector[:-2]
        raw_pixel_feature = raw_pixel_feature[:-2]
        for i in range(len(feature_vector)):
            dist += (feature_vector[i]-raw_pixel_feature[i])**2
        return round(dist**0.5, 4)


while True:
    mode = raw_input(
            "Please enter the mode in which you wish to run the program\n1. For Classifying Image based on existing training set\n2. For training model using your own training dataset(Note this will completely update the training dataset)\n3. To perform 3-Cross fold validation\n4. To do K-means Clustering.\n5. To do hierarchical clustering \n6. To exit\n\n"
            )

    try:
        mode = int(mode)
    except:
        print "Please enter valid integer from the choices"
        continue
    # Add exception handling for just enter
    if mode == None or mode == '':
        continue

    if int(mode) == 6:
        print "Thank you for using Saketh's Image Classifier, Good bye!"
        break
    else:
        Environment(mode)

