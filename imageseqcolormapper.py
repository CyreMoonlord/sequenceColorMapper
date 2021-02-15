'''
This code was designed to create a visual representation of a series of image files
As written it selects the four most dominant colors and creates a 1px line composed of those colors
File handling bits will need to be adjusted for anything other than the macOS
'''

#timer is optional, I put mine in anything that runs more than 10 minutes :)
import time
def stop_clock():
  if (time.time() - start_time) < 60:
    print("--- %s seconds ---" % (time.time() - start_time))
  elif (time.time() - start_time) < 3600:
    print("--- {} minutes {} seconds ---".format((int((time.time() - start_time)/60)),((time.time() - start_time)%60)))
  else:
    print("--- {} hours {} minutes {} seconds ---".format((int((time.time() - start_time)/3600)),(int(((time.time() - start_time)%3600)/60)),((time.time() - start_time)%60)))
  return
#starts timer
start_time = time.time()

#imports
import os
from PIL import Image
import numpy as np
import scipy
import scipy.cluster
import imageio

#folder for project files
projectFolder = os.path.expanduser('~/eProjects/MovieColors/')
#source of image sequence
sourceFolder = projectFolder + 'Coco/'
#sourceFolder = projectFolder + 'TestFrames/'
#sourceFolder = projectFolder + 'ColorTest Images/'

#Note: Could apply other sorting method depending on source images
seqlist = sorted(os.listdir(sourceFolder))

#totalImages is for reference only
totalImages = 0

#sets an initial row of black pixels
colorMap = np.zeros((4,3))

#number of clusters for k-means
numColors = 4

for fn in seqlist:
  path = sourceFolder + fn
  #in case there are non-images (hidden files)
  try:
    frame = Image.open(path)
    imgAr = np.asarray(frame)
    shape = imgAr.shape
    imgAr = imgAr.reshape(np.product(shape[:2]), shape[2]).astype(float)

    colorValues, dist = scipy.cluster.vq.kmeans(imgAr, numColors)
    totalImages += 1

    #if 4 clusters are not present the first color is used to make up the balance
    while len(colorValues) < 4:
      colorValues = np.concatenate((colorValues, [colorValues[0]]), axis=0)

    colorMap = np.append(colorMap, colorValues, axis=0)

  except IOError:
    # file not an image
    pass

#save the numpy array (optional)
np.save('colorMapArray.npy', colorMap)

pxHeight = int((colorMap.shape[0])/4)
imageio.imwrite('ColorMap.png', colorMap.reshape(pxHeight,4,3).astype(np.uint8))

#Saves version with single pixels per color on each line and version with 500px per color
cMap = Image.open(projectFolder + 'ColorMap.png')
cMap = cMap.resize((2000,pxHeight), Image.NEAREST)
cMap = cMap.save(projectFolder + 'ColorMap2000.png')

#for reference
print(totalImages)
stop_clock()


