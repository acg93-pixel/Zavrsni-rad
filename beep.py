import cv2# Standard imports
import cv2
import numpy as np

im = cv2.imread('/home/ana/Pictures/slika1.png')

gaussian_3 = cv2.GaussianBlur(im, (9, 9), 10.0)
unsharp_image = cv2.addWeighted(im, 1.5, gaussian_3, -0.5, 0, im)
cv2.imwrite("slika2_unsharp.jpg", unsharp_image)

cv2.imshow("slika2_unsharp.jpg", unsharp_image)

# Set up the detector with default parameters.
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 200

# Filter by Area.
params.filterByArea = True
params.minArea = 0.1
params.maxArea = 20


# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.001

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.001

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.001

detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(unsharp_image)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(unsharp_image, keypoints, np.array([]), (0, 255, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)

print (len(keypoints))

cv2.waitKey(0)
cv2.destroyAllWindows()


