import cv2
import numpy as np

print(cv2.__version__)

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create ChArUco board
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
board = cv2.aruco.CharucoBoard((9, 6), 15, 11, dictionary)

# Create parameters for detection
params = cv2.aruco.DetectorParameters()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ChArUco board
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    
    if len(corners) > 0:
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
    
    # Display the frame
    cv2.imshow('ChArUco Board Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
