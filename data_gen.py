import cv2
import os

# Base path to video files
video_folder = 'reflect/main/few_shot_examples'

# List of all video files
video_files = [f for f in os.listdir(video_folder) if f.endswith(('.mp4', '.avi'))]  # Add more extensions if needed

# Keyframes to capture from each video
event_keyframes = [45, 234, 11, 78, 32, 98, 43, 90, 32, 47]

# Single output folder for all frames
output_folder = 'few_shot_data'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each video file
for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}.")
        continue

    # Process each keyframe
    for i, kf in enumerate(event_keyframes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, kf)
        ret, frame = cap.read()
        if ret:
            # Include video name in the frame filename to ensure uniqueness
            frame_filename = f'{output_folder}/{video_file.split(".")[0]}_frame_{kf}_{i}.png'
            cv2.imwrite(frame_filename, frame)
            print(f'Saved: {frame_filename}')
        else:
            print(f"Failed to retrieve frame at index {kf} from {video_file}")

    # Release the video capture object
    cap.release()

print("All processing complete.")