import cv2
import os

def video_to_images(video_path, output_dir):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the video frame by frame
    success, frame = video.read()
    count = 0

    while success:
        # Save the current frame as an image file
        image_path = os.path.join(output_dir, f"frame_{count}.jpg")
        cv2.imwrite(image_path, frame)

        # Read the next frame
        success, frame = video.read()
        count += 1

    # Release the video file
    video.release()

# Example usage
video_path = "data/coffee_task.mp4"
output_dir = "data/images"
video_to_images(video_path, output_dir)