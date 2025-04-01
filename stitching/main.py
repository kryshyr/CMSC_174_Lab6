import cv2
from feature_extraction import extract_features
from feature_matching import match_features
from homography import compute_homography
from image_loader import load_images
from warping_blending import warp_and_blend


def stitch_images(images):
    
    """ stitch the list of images into a panorama """
    stitched_image = images[0]

    for i in range(1, len(images)):
        kp1, desc1 = extract_features(stitched_image)
        kp2, desc2 = extract_features(images[i])

        matches = match_features(desc1, desc2)
        if len(matches) < 10:
            print(f"Not enough matches for image {i}. Skipping.")
            continue

        H = compute_homography(kp1, kp2, matches)
        stitched_image = warp_and_blend(stitched_image, images[i], H)

    return stitched_image

if __name__ == "__main__":
    images = load_images()
    
    if len(images) < 2:
        print(" Not enough images to stitch. Please provide at least two images.")
    else:
        stitched_image = stitch_images(images)
        cv2.imwrite("final_panorama.jpg", stitched_image)
        print("Final panorama saved as final_panorama.jpg")
