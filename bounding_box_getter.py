import cv2

def draw_bounding_box(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Create a window to display the image
    cv2.namedWindow("Image")
    cv2.imshow("Image", image)

    # Initialize the coordinates of the bounding box
    bbox = []

    def mouse_callback(event, x, y, flags, param):
        nonlocal bbox

        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing the bounding box
            bbox = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing the bounding box
            bbox.append((x, y))

            # Print the coordinates of the bounding box
            print("Bounding Box Coordinates:")
            print("Top Left: ", bbox[0])
            print("Bottom Right: ", bbox[1])

            # Draw the bounding box on the image
            cv2.rectangle(image, bbox[0], bbox[1], (0, 255, 0), 2)
            cv2.imshow("Image", image)

    # Set the mouse callback function
    cv2.setMouseCallback("Image", mouse_callback)

    # Wait for the user to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Path to the image file
image_path = "test_images/frame_15.jpg"

# Call the function to draw the bounding box
draw_bounding_box(image_path)
