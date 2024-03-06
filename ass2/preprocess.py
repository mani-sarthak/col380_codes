import cv2
import numpy as np
import sys

np.set_printoptions(linewidth=np.inf,formatter={'float': '{: 0.6f}'.format})

img = cv2.imread(sys.argv[1],0)
if img.shape != [28,28]:
    img2 = cv2.resize(img,(28,28))
    
img = img2.reshape(28,28,-1);

#revert the image,and normalize it to 0-1 range
img = 1.0 - img/255.0

mat = np.matrix(img)

for row in mat.A:
    array_string = np.array2string(row, separator='')[1:-1]
    print(array_string)
