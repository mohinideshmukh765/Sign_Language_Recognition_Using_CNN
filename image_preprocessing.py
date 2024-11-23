import os
import cv2
import numpy as np

# Preprocessing function
minValue = 70

def func(frame):    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res

# Directory path
output_directory = "Z"  # Directory to save preprocessed images

# Ensure the output directory exists
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Initialize the camera
cap = cv2.VideoCapture(0)  # Change to 1 or other index if necessary

if not cap.isOpened():
    print("Error: Could not access the camera.")
    exit()

image_count = 0  # To track the number of images saved
max_images = 1000  # Stop after saving 1000 images

# Define the box area (you can adjust these values)
x, y, w, h = 300, 100, 300, 300  # Example box coordinates (x, y, width, height)

# Mouse callback function to select box area (optional)
def select_box(event, x1, y1, flags, param):
    global x, y, w, h, box_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        box_drawing = True
        x, y = x1, y1
    elif event == cv2.EVENT_MOUSEMOVE:
        if box_drawing:
            w, h = x1 - x, y1 - y
    elif event == cv2.EVENT_LBUTTONUP:
        box_drawing = False
        w, h = x1 - x, y1 - y

# Set up mouse callback function
cv2.namedWindow("Capture Box")
cv2.setMouseCallback("Capture Box", select_box)

box_drawing = False

# Make window fullscreen
cv2.namedWindow("Capture Box", cv2.WINDOW_FULLSCREEN)

# Special box dimensions for the preprocessed image (make it smaller)
processed_box_x, processed_box_y, processed_box_w, processed_box_h = 400, 100, 200, 200  # Smaller box for processed image

# Continuously capture and save images
while image_count < max_images:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image")
        break

    # Draw the selection box on the frame
    temp_frame = frame.copy()
    cv2.rectangle(temp_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Crop the frame to the selected box area
    if w > 0 and h > 0:
        cropped_frame = frame[y:y + h, x:x + w]

        # Check if the cropped frame is valid (non-empty)
        if cropped_frame.size != 0:
            # Preprocess the cropped frame
            processed_image = func(cropped_frame)

            # Convert the processed image to 3 channels for display
            processed_image_3channel = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)

            # Resize processed image to fit into the smaller special box area
            processed_image_resized = cv2.resize(processed_image_3channel, (processed_box_w, processed_box_h))

            # Save the preprocessed image with names as integers (0 to 999)
            output_path = os.path.join(output_directory, f"{image_count}.png")
            cv2.imwrite(output_path, processed_image)
            print(f"Saved: {output_path}")
            image_count += 1

            # Check if resized image fits in the special box
            if processed_image_resized.shape[0] == processed_box_h and processed_image_resized.shape[1] == processed_box_w:
                # Show the original frame with the box and the processed image in the smaller box
                temp_frame[processed_box_y:processed_box_y + processed_box_h, processed_box_x:processed_box_x + processed_box_w] = processed_image_resized
                cv2.imshow("Capture Box", temp_frame)  # Show combined frame with original and processed images
            else:
                print("Error: Processed image size does not match box dimensions.")
        else:
            print("Error: Cropped frame is empty.")
    else:
        print("Error: Invalid box dimensions.")

    # Slow down the capture by adding a delay (in milliseconds)
    cv2.waitKey(100)  # Delay of 500 milliseconds (0.5 second)

    # Press 'q' to exit early or 's' to stop the camera and generation
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        print("Stopping the camera and generation.")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print(f"Dataset generation complete. {image_count} images saved.")