import os

import cv2
import numpy as np

os.chdir("D:/UPV FILES/Year_4/SEM_2/CMSC 174 - LAB/CMSC_174_Lab6/actionshot")


def create_action_shot(video_path, output_path, num_frames=10, threshold=30):
    """
    Create an action shot from a video by compositing multiple frames.

    Parameters:
        video_path (str): Path to input video file
        output_path (str): Path to save output image
        num_frames (int): Number of frames to composite (default: 10)
        alpha (float): Blending weight for new frames (0-1, default: 0.5)
        threshold (int): Motion threshold to detect moving objects (default: 30)
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Read the first frame as background
    ret, background = cap.read()
    if not ret:
        print("Error: Could not read video frames")
        return

    # Convert background to grayscale for motion detection
    background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    # background_gray = cv2.GaussianBlur(background_gray, (21, 21), 0)

    # Initialize the action shot with the first frame
    action_shot = background.copy()

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(1, total_frames // num_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        # Process current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute difference between current frame and background
        frame_diff = cv2.absdiff(background_gray, gray)
        _, thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

        # Find contours of moving objects
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask for moving objects
        # mask = np.zeros_like(thresh)
        
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                # cv2.drawContours(mask, [contour], 0, 255, -1)
                # Get bounding box of moving object
                x, y, w, h = cv2.boundingRect(contour)

                # Extract moving object
                moving_object = frame[y:y+h, x:x+w]

                # Overlay the moving object onto the action shot
                action_shot[y:y+h, x:x+w] = moving_object

        if frame_count >= num_frames * skip_frames:
            break

    # Save the final action shot
    cv2.imwrite(output_path, action_shot)
    print(f"Action shot saved to {output_path}")

    # Release resources
    cap.release()


# Example usage
if __name__ == "__main__":
    create_action_shot("videos/spike.mp4", "action_shot.jpg", num_frames=4)