import cv2


def extract_features(image):
    # Initialize the SIFT feature detector and extractor
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors for the image
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    # Draw keypoints on the images and display the images with keypoints
    # image_with_keypoints = cv2.drawKeypoints(image, keypoints, None)
    # cv2.imshow("Keypoints", image_with_keypoints)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    return keypoints, descriptors
