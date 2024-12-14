import cv2
import numpy as np

# Path to your MKV video file
video_path = 'C:/Users/home/Desktop/Camera_recognition/mvs20231024_144105.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Loop through each frame in the video
while cap.isOpened():
    ret, frame = cap.read()  # Read the frame
    if ret:
        # Convert the frame from BGR to YUV (if your video is not already in YUV format)
        yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)

        # Split the YUV frame into Y, U, and V channels
        y, u, v = cv2.split(yuv_frame)

        # At this point, you can process the Y, U, and V channels as per your project's requirements

        # Example: Show the Y channel
        cv2.imshow('Y Channel', y)
        if cv2.waitKey(25) & 0xFF == ord('q'):  # Press Q on the keyboard to exit
            break
    else:
        break

# When everything done, release the video capture object
cap.release()
cv2.destroyAllWindows()
