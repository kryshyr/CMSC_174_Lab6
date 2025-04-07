import cv2
import numpy as np

cap = cv2.VideoCapture('actionshot_fb.mp4')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

bg_frame = None

frames = []
for _ in range(frame_count): # get all frames
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

bg_frame = frames[-1] # get last frame for bg
background_gray = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2GRAY)
background_blur = cv2.GaussianBlur(background_gray, (5, 5), 0)

actionshot = bg_frame.copy()

frame_interval = 35 # skip frames to show action shot
frame_idx = list(range(0, frame_count, frame_interval))

for i in frame_idx:
    frame = frames[i] # frames where subj will be used to put on bg

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # difference between the last frame and the current frame
    diff = cv2.absdiff(background_blur, gray_blur)

    # combine edges and thresholded diff for optimized masking
    edges = cv2.Canny(diff, 50, 150)
    _, thresh_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    combined_mask = cv2.bitwise_or(thresh_diff, edges)

    # morphological operations for smooth mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(combined_mask, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # remove small noisy area
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask)
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)

    # put moving subject to canvas for action shot effect
    moving_subject = cv2.bitwise_and(frame, frame, mask=clean_mask)
    inv_mask = cv2.bitwise_not(clean_mask)
    background_part = cv2.bitwise_and(actionshot, actionshot, mask=inv_mask)
    actionshot = cv2.add(background_part, moving_subject)

cap.release()
cv2.imwrite('actionshot.jpg', actionshot)
