import cv2
import numpy as np

def find_pixel_difference(img1, img2):
    # Load the images
    # img1 = cv2.imread(image1)
    # img2 = cv2.imread(image2)

    # Calculate the absolute difference between the images
    diff = cv2.absdiff(img1, img2)

    # Convert the difference image to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to get a binary image
    _, threshold = cv2.threshold(diff_gray, 0, 255, cv2.THRESH_BINARY)

    # Count the number of non-zero pixels in the binary image
    # pixel_diff = np.count_nonzero(threshold)

    # Return the pixel difference
    return diff_gray, threshold


# img1 = 'frame_52.jpg'
# img2 = 'frame_53.jpg'

# diff_gry,thres = find_pixel_difference(img1, img2)

# # print(f'Number of different pixels: {pixel_diff}')
# cv2.imshow('Difference Image', diff_gry)
# cv2.imshow('Threshold Image', thres)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
prev_img = None
cap = cv2.VideoCapture('color.mp4')

# save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if prev_img is not None:
        diff_gry, thres = find_pixel_difference(prev_img, frame)
        out.write(diff_gry)
        cv2.imshow('Difference Image', diff_gry)
        cv2.imshow('Threshold Image', thres)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    prev_img = frame

cap.release()
out.release()

cv2.destroyAllWindows()




