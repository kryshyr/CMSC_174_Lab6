import os

import cv2
import numpy as np

""" EXTRACT FEATURES """
def extract_features(image):
    
    # initialize the SIFT feature detector and extractor
    sift = cv2.SIFT_create()
    
    # detect keypoints and compute descriptors for the image
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    return keypoints, descriptors


""" MATCH FEATURES """
def match_features(desc1, desc2):
    
    """ Source: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html"""
    """ Source: https://medium.com/@lucasmassucci/exploring-correlations-in-images-with-sift-and-flann-an-efficient-approach-to-feature-matching-1fdb33697f5e"""
    
    # initialize the feature matcher using FLANN matching
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # find the matches between the two sets of descriptors
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    
    return good_matches


""" FIND HOMOGRAPHY """
def find_homography(kp1, kp2, matches):
    """ finds homography matrix using matched keypoints """
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
    
    return homography
    

""" WARP AND BLEND """
def warp_and_blend(image1, image2, H):
    
    """ Source: https://www.opencvhelp.org/tutorials/advanced/image-stitching/"""
    
    # getting the image dimensions
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # defining corners of both the images
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)

    # transform the first image corners
    transformed_corners1 = cv2.perspectiveTransform(corners1, H)

    # stacking all the corners together
    all_corners = np.vstack((transformed_corners1, corners2))

    # to compute bounding box dimensions that will enclose both the images
    # will be used to define the dimensions of the new canvas that will hold the images after stitching
    x_min, y_min = np.int32(all_corners.min(axis=0).ravel())
    x_max, y_max = np.int32(all_corners.max(axis=0).ravel())

    # tompute translation matrix to shift coordinates
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    # warp image1 to the new coordinate system
    warped_image1 = cv2.warpPerspective(image1, translation @ H, (x_max - x_min, y_max - y_min))

    # to create a blank canvas that is large enough for both images
    canvas = np.zeros_like(warped_image1, dtype=np.uint8)

    # to overlay image2 onto the warped image1
    x_offset, y_offset = -x_min, -y_min
    canvas[y_offset:y_offset+h2, x_offset:x_offset+w2] = image2

    # converting images to float32 for blending
    warped_image1 = warped_image1.astype(np.float32)
    canvas = canvas.astype(np.float32)

    # apply intensity adjustment to achieve uniform brightness
    mask1 = (warped_image1 > 0).astype(np.uint8) * 255

    # compute blending mask (gaussian blur for smooth transition)
    blend_mask = cv2.GaussianBlur(mask1, (21, 21), 10)
    blend_mask = blend_mask.astype(np.float32) / 255.0

    blended_image = (warped_image1 * blend_mask + canvas * (1 - blend_mask)).astype(np.uint8)

    return blended_image


""" STITCH IMAGES """
def stitch_images(images):
    """ stitches a list of images into a panorama"""
    
    stitched_image = images[0]
    
    for i in range(1, len(images)):
        kp1, des1 = extract_features(stitched_image)
        kp2, des2 = extract_features(images[i])
        
        matches = match_features(des1, des2)
        
        H = find_homography(kp1, kp2, matches)
        if H is not None:
            stitched_image = warp_and_blend(stitched_image, images[i], H)
            
    return stitched_image


""" LOAD IMAGES """
def load_images(directory = "data"):
    
    """ load images from the specified directory and return them as a list """
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist.")

    image_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.jpg'))])
    images = [cv2.imread(os.path.join(directory, img_file)) for img_file in image_files]

    return images


""" MAIN FUNCTION """
if __name__ == "__main__":
    images = load_images()
    stitched_image = stitch_images(images)
    cv2.imwrite("Ferrer-Saluria_lab06_stitch.png", stitched_image)
    # cv2.imshow("Stitched Image", stitched_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
