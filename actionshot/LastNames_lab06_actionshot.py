import cv2
import numpy as np


def create_action_shot(video_path, output_path, num_frames=10, threshold=30):
    """
    Create an action shot preserving all movement positions without blending.

    Parameters:
        video_path (str): Input video path
        output_path (str): Output image path
        num_frames (int): Number of frames to process
        threshold (int): Motion detection threshold
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    # Initialize background subtractor
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    # Read first frame as base image
    ret, action_shot = cap.read()
    if not ret:
        print("Error: Could not read video frames")
        return

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = max(1, total_frames // num_frames)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_frames != 0:
            continue

        # Get foreground mask using background subtraction
        fg_mask = backSub.apply(frame)

        # Refine the mask
        _, thresh = cv2.threshold(fg_mask, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Create object mask
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros_like(thresh)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(mask, [contour], 0, 255, -1)

        # Only update new regions not previously modified
        update_mask = cv2.bitwise_and(mask, cv2.bitwise_not(action_shot[:, :, 0]))
        action_shot = np.where(update_mask[..., None], frame, action_shot)

        if frame_count >= num_frames * skip_frames:
            break

    cv2.imwrite(output_path, action_shot)
    print(f"Action shot saved to {output_path}")
    cap.release()


# Usage
if __name__ == "__main__":
    create_action_shot("input_video.MOV", "action_shot.jpg", num_frames=6, threshold=25)