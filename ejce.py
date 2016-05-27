# Standard imports
import cv2
import numpy as np

# Read image
im = cv2.imread('/home/ana/Pictures/slika4.png')

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
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 255, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)

print
var = ["broj cestica", len(keypoints)]
cv2.waitKey(0)
cv2.destroyAllWindows()
