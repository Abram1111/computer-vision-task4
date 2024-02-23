import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
from skimage.segmentation import mark_boundaries
from skimage import io, transform


#KMEANS
def kmeans(img,K,threshold=0.85,max_iter=100):
    # Change color to RGB (from BGR)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    pixel_vals = image.reshape((-1,3))        # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = np.float32(pixel_vals)       # Convert to float type
    pixel_vals[np.isnan(pixel_vals)] = 1e-6   # Check for NaN values in the pixel values array and set them to a small value

    
    centroids = pixel_vals[np.random.choice(pixel_vals.shape[0], K, replace=False), :]    # Initialize the cluster centroids randomly
    # print(centroids)
    centroids[np.isnan(centroids)] = 1e-6  # Check for NaN values in the centroids array and set them to a small value

    old_centroids = np.zeros_like(centroids)    # Initialize the old centroids

    for i in range(max_iter):
        # Assign each pixel to the nearest centroid
        distances = np.sqrt(((pixel_vals - centroids[:, np.newaxis])**2).sum(axis=2))
        # print(distances)
        labels = np.argmin(distances, axis=0)

        # Update the cluster centroids
        for k in range(K):
            centroids[k] = np.mean(pixel_vals[labels == k], axis=0)


        centroids[np.isnan(centroids)] = 1e-6                      # Check for NaN values in the centroids array and set them to a small value
        if np.abs(centroids - old_centroids).mean() < threshold:     # Check for convergence
            break
        old_centroids = centroids.copy()

    # Convert data into 8-bit values
    centers = np.uint8(centroids)
    segmented_data = centers[labels.flatten()]

    # Reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((image.shape))

    # return cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    return segmented_image

def mean_shift(img,window=70,threshold=1.0):
    def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
    row, col, _ = img.shape
    segmented_image = np.zeros((row,col,3), dtype= np.uint8)
    feature_space   = np.zeros((row * col,5))
    counter=0 
    current_mean_random = True
    current_mean_arr = np.zeros((1,5))


    for i in range(0,row):
        for j in range(0,col):      
            feature_space[counter]=[img[i][j][0],img[i][j][1],img[i][j][2],i,j]
            counter+=1

    while(len(feature_space) > 0):
        t1=time.time()
        print (len(feature_space))
        #selecting a random row from the feature space and assigning it as the current mean    
        # Select a random row from the feature space and assign it as the current mean
        if current_mean_random:
            current_mean_index = random.randint(0, feature_space.shape[0] - 1)
            current_mean_arr[0] = feature_space[current_mean_index]
        below_threshold_arr=[]

        distances = np.zeros(feature_space.shape[0])
        for i in range(0,len(feature_space)):
            distance = 0
            #Finding the eucledian distance of the randomly selected row i.e. current mean with all the other rows
            for j in range(0,5):
                distance += ((current_mean_arr[0][j] - feature_space[i][j])**2)
                    
            distances[i] = distance**0.5

            #Checking if the distance calculated is within the window. If yes taking those rows and adding 
            #them to a list below_threshold_arr
        below_threshold_arr = np.where(distances < window)[0]
        
        

        mean_color = np.mean(feature_space[below_threshold_arr, :3], axis=0)
        mean_pos = np.mean(feature_space[below_threshold_arr, 3:], axis=0)
        # Calculate Euclidean distance between mean color/position and current mean
        mean_color_distance = euclidean_distance(mean_color, current_mean_arr[0][:3])
        mean_pos_distance = euclidean_distance(mean_pos, current_mean_arr[0][3:])
        mean_e_distance = mean_color_distance + mean_pos_distance


        if(mean_e_distance < threshold):                
            new_arr = np.zeros((1,3))
            new_arr[0] = mean_color
            # When found, color all the rows in below_threshold_arr with 
            #the color of the row in below_threshold_arr that has i,j nearest to mean_i and mean_j
            current_mean_random = True
            segmented_image[feature_space[below_threshold_arr, 3].astype(int), feature_space[below_threshold_arr, 4].astype(int)] = new_arr
            # Remove below-threshold pixels from feature space
            feature_space[below_threshold_arr, :] = -1
            feature_space = feature_space[feature_space[:, 0] != -1]
            
        else:
            current_mean_random = False
            current_mean_arr[0, :3] = mean_color
            current_mean_arr[0, 3:] = mean_pos
    return cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)

#----------------------------------- Region Growing---------------------------
# stack as a class
class Stack():
    def __init__(self):
        self.item = []
        self.obj=[]
    def push(self, value):
        self.item.append(value)
    def pop(self):
        return self.item.pop()

    def size(self):
        return len(self.item)

    def isEmpty(self):
        return self.size() == 0

    def clear(self):
        self.item = []

class regionGrowSegmentation():
  
    def __init__(self,image_path,threshold):
        # reading new image
        # self.readImage(im_path)
        self.image = cv2.imread(image_path,1)  
        self.h, self.w,_ =  self.image.shape
        # initilalize all image variables
        self.passedBy = np.zeros((self.h,self.w), np.double)
        self.currentRegion = 0
        self.iterations=0
        self.output=np.zeros((self.h,self.w,3), dtype='uint8')
        self.stack = Stack()
        self.thresh=float(threshold)

    def getNeighbour(self, x0, y0):
        neighbour = []
        for i in (-1,0,1):
            for j in (-1,0,1):
                if (i,j) == (0,0): 
                    continue
                x = x0+i
                y = y0+j
                if self.limit(x,y):
                    neighbour.append((x,y))
        return neighbour
    
    def ApplyRegionGrow(self):
        # selecting 10 seeds randomly
        randomseeds=[[self.h/2,self.w/2],
                        [self.h/3,self.w/3],[2*self.h/3,self.w/3],[self.h/3-10,self.w/3],
                        [self.h/3,2*self.w/3],[2*self.h/3,2*self.w/3],[self.h/3-10,2*self.w/3],
                        [self.h/3,self.w-10],[2*self.h/3,self.w-10],[self.h/3-10,self.w-10]
                    ]
        np.random.shuffle(randomseeds)

        for x0 in range (self.h):
            for y0 in range (self.w):
                # checking intensity of each pixel
                if self.passedBy[x0,y0] == 0 and (int(self.image[x0,y0,0])*int(self.image[x0,y0,1])*int(self.image[x0,y0,2]) > 0) :  
                   
                    self.currentRegion += 1
                    self.passedBy[x0,y0] = self.currentRegion
                    self.stack.push((x0,y0))
                    self.prev_region_count=0
                    while not self.stack.isEmpty():
                        x,y = self.stack.pop()
                        self.BFS(x,y)
                        self.iterations+=1
                    # if we exceed iteration number or image size
                    if(self.iterations>200000 or np.count_nonzero(self.passedBy > 0) == self.w*self.h):
                        break
                    
                    if(self.prev_region_count<8*8):     
                        self.passedBy[self.passedBy==self.currentRegion]=0
                        x0=random.randint(x0-4,x0+4)
                        y0=random.randint(y0-4,y0+4)
                        x0=max(0,x0)
                        y0=max(0,y0)
                        x0=min(x0,self.h-1)
                        y0=min(y0,self.w-1)
                        self.currentRegion-=1

        for i in range(0,self.h):
            for j in range (0,self.w):
                val = self.passedBy[i][j]
                if(val==0):
                    self.output[i][j]=255,255,255
                else:
                    self.output[i][j]=val*10,val*10,val*10
        # if(self.iterations>200000):
        #     print("Max Iterations")
        # print("Iterations : "+str(self.iterations))
        # # cv2.imshow("",self.output)
        # # cv2.waitKey(0)
        # # cv2.destroyAllWindows()
        return self.output

    def BFS(self, x0,y0):
        regionNum = self.passedBy[x0,y0]
        elems=[]
        elems.append((int(self.image[x0,y0,0])+int(self.image[x0,y0,1])+int(self.image[x0,y0,2]))/3)
        var=self.thresh
        neighbours=self.getNeighbour(x0,y0)
        
        for x,y in neighbours:
            if self.passedBy[x,y] == 0 and self.distance(x,y,x0,y0)<var:
                if(self.iterations>200000 or np.count_nonzero(self.passedBy > 0) == self.w*self.h):
                    break;
                self.passedBy[x,y] = regionNum
                self.stack.push((x,y))
                elems.append((int(self.image[x,y,0])+int(self.image[x,y,1])+int(self.image[x,y,2]))/3)
                var=np.var(elems)
                self.prev_region_count+=1
            var=max(var,self.thresh)
    def limit(self, x,y):
        return  0<=x<self.h and 0<=y<self.w
    def distance(self,x,y,x0,y0):
        return ((int(self.image[x,y,0])-int(self.image[x0,y0,0]))**2+(int(self.image[x,y,1])-int(self.image[x0,y0,1]))**2+(int(self.image[x,y,2])-int(self.image[x0,y0,2]))**2)**0.5




#----------------------------------AgglomerativeClustering-------------------------------------------------------#

class AgglomerativeClustering:
    def __init__(self, n_clusters=2, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit(self, X):
        self.n_samples, self.n_features = X.shape
        self.labels_ = np.arange(self.n_samples)
        self.n_components_ = self.n_samples
        self.distances_ = self._compute_distances(X)
        self.tree_ = self._compute_tree()
        self.labels_ = self._extract_labels()
        return self

    def _compute_distances(self, X):
        distances = np.zeros((self.n_samples, self.n_samples))
        for i in range(self.n_samples):
            for j in range(i+1, self.n_samples):
                distances[i][j] = np.linalg.norm(X[i] - X[j])
                distances[j][i] = distances[i][j]
        return distances

    def _compute_tree(self):
        children = np.zeros((self.n_samples - 1, 2))
        distances = np.zeros(self.n_samples - 1)
        component_labels = np.arange(self.n_samples)

        for i in range(self.n_samples - 1):
            (component1, component2), dist = self._get_closest_components(component_labels)
            children[i] = [component1, component2]
            distances[i] = dist
            component_labels[component1] = self.n_samples + i
            component_labels[component2] = self.n_samples + i

        return children, distances

    def _get_closest_components(self, component_labels):
        min_dist = np.inf
        component1 = None
        component2 = None

        for i in range(self.n_components_):
            for j in range(i+1, self.n_components_):
                if component_labels[i] != component_labels[j]:
                    dist = self.distances_[i][j]
                    if dist < min_dist:
                        min_dist = dist
                        component1 = i
                        component2 = j

        return (component1, component2), min_dist

    def _extract_labels(self):
        labels = np.zeros(self.n_samples, dtype=int)
        n_clusters = self.n_samples
        for i in range(self.n_samples - 1):
            if n_clusters == self.n_clusters:
                break
            c1, c2 = self.tree_[0][i], self.tree_[1][i]
            if c1 < self.n_samples:
                labels[c1] = n_clusters
                n_clusters += 1
            if c2 < self.n_samples:
                labels[c2] = n_clusters
                n_clusters += 1
        return labels

    def predict(self, X):
        distances = np.zeros((X.shape[0], self.n_samples))
        for i in range(X.shape[0]):
            for j in range(self.n_samples):
                distances[i][j] = np.linalg.norm(X[i] - X[j])

        labels = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            distances_i = distances[i]
            indices = np.argsort(distances_i)
            cluster_sizes = np.zeros(self.n_samples, dtype=int)
            for j in range(self.n_samples):
                cluster_sizes[self.labels_[indices[j]]] += 1
                if cluster_sizes[self.labels_[indices[j]]] == self.n_clusters:
                    labels[i] = self.labels_[indices[j]]
                    break
        return labels





