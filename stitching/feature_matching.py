# import cv2


# def match_features(desc1, desc2):
    
#     """ matches features between two sets of descriptors using flann.knnMatch """
#     """ Source: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html"""
    
#     # initialize the feature matcher using FLANN matching
#     index_params = dict(algorithm=1, trees=5)
#     search_params = dict(checks=50)
    
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
    
#     # find the matches between the two sets of descriptors
#     matches = flann.knnMatch(desc1, desc2, k=2)
    
#     # selecting only good matches.
#     matchesMask = [[0,0] for i in range(len(matches))]
    
#     # apply Lowe's ratio test
#     for i, (m, n) in enumerate(matches):
#         if m.distance < 0.7 * n.distance:  # if the first match is much better than the second match
#             matchesMask[i] = [1, 0]  # mark it as a good match
    
#     # return only good matches
#     good_matches = []
#     for i, match in enumerate(matches):
#         if matchesMask[i][0] == 1:
#             good_matches.append(match[0])  # append the first match
            
#     return good_matches



import cv2


def match_features(desc1, desc2):
    """ matches features between two sets of descriptors using flann.knnMatch """
    """ Source: https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html"""
    """ Source: https://medium.com/@lucasmassucci/exploring-correlations-in-images-with-sift-and-flann-an-efficient-approach-to-feature-matching-1fdb33697f5e"""
    
    # initialize the feature matcher using FLANN matching
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # find the matches between the two sets of descriptors
    matches = flann.knnMatch(desc1, desc2, k=2)
    
    # apply Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    return good_matches
