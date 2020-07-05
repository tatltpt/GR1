import cv2 
import matplotlib.pyplot as plt

#reading image
img1 = cv2.imread('img1.png')  
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

#keypoints
sift = cv2.xfeatures2d.SIFT_create()
keypoints_1, descriptors_1 = sift.detectAndCompute(img1,None)

img_1 = cv2.drawKeypoints(gray1,keypoints_1,img1)
cv2.imwrite('sift_keypoint.png',img_1)
plt.imshow(img_1)
plt.show()
