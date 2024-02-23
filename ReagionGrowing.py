import random
import numpy as np
import cv2
from skimage.io import imread,imsave
from skimage.filters import rank,gaussian
from skimage.morphology import disk
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.segmentation import mark_boundaries
from skimage.measure import label
from skimage.color import rgb2hsv


#----------------------------------- Region Growing---------------------------

#----------------------------------- Second method ---------------------------
def get_descr(region):
    return [region[:,0].mean(), region[:,1].mean(), region[:,2].mean()]
def get_markers(im, indices=False):
    im_ = gaussian(im, sigma=4)
    gradr = rank.gradient(im_[:,:,0],disk(5)).astype('int')
    gradg = rank.gradient(im_[:,:,1],disk(5)).astype('int')
    gradb = rank.gradient(im_[:,:,2],disk(5)).astype('int')
    grad = gradr+gradg+gradb
    
    return peak_local_max(grad.max()-grad,threshold_rel=0.5, min_distance=60),grad

def RegionGrowing (filename):
    image = imread(filename)
    markers, grad = get_markers(image, False)
    markers = label(markers)
    ws = watershed(grad, markers)
    descriptors = np.zeros((ws.max()+1,3))
    im_descriptors = np.zeros_like(image)

    for i in range(ws.min(),ws.max()+1):
        descriptors[i] = get_descr(image[ws==i])
        im_descriptors[ws==i] = descriptors[i]
    hsv = rgb2hsv(im_descriptors)
    mask = (hsv[:,:,0]<0.15)*(hsv[:,:,1]>0.3)

    return mask




#----------------------------------- First method---------------------------

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
        self.image = cv2.imread(image_path )  
        self.h = self.image.shape[0]
        self.w=  self.image.shape[1]
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




