import cv2
import numpy as np


def warp_and_blend(image1, image2, homography):
    
    # warp the first image using the homography matrix
    result = cv2.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))
    
    # ensure the images are the same size
    if result.shape != image2.shape:
        result = cv2.resize(result, (image2.shape[1], image2.shape[0]))

    # blending the first image with the second image using alpha blending
    alpha = 0.5
    blended_image = cv2.addWeighted(result, alpha, image2, 1 - alpha, 0)

    return blended_image
