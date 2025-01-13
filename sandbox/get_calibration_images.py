import gxipy as gx
import cv2
import os, shutil
import time

# set camera parameters
gain = 10

device_manager = gx.DeviceManager()
dev_num, dev_info_list = device_manager.update_device_list()
print(f"Number of enumerated devices is {dev_num}")

# Open camera by serialnumber
sn = dev_info_list[0]['sn']
cam1 = device_manager.open_device_by_sn(sn)

# get resolution parameters
cam1.size = (cam1.Width.get(), cam1.Height.get())
width = cam1.size[0]
height = cam1.size[1]

print((cam1.Width.get(), cam1.Height.get()))

# Get frame rate
frame_rate = cam1.CurrentAcquisitionFrameRate.get()

# Set continuous acquisition
cam1.TriggerMode.set(gx.GxSwitchEntry.OFF)

# hardware trigger
cam1.TriggerSource.set(1)
cam1.LineSelector.set(0)
cam1.LineMode.set(0)
cam1.UserSetSelector.set(1)
cam1.UserSetSave.send_command()

# Set exposure time
cam1.ExposureTime.set(10000.0)

# Set gain (Autogain of niet nodig???)
cam1.Gain.set(gain)

# Set acquisition buffer count
cam1.data_stream[0].set_acquisition_buffer_number(1)

# Start data acquisition
cam1.stream_on()

print(f"Camera {sn} is initialized")

# Create ChArUco board
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
board = cv2.aruco.CharucoBoard((11, 8), 15, 11, dictionary)
board.setLegacyPattern(True) # needed for ChAruco boards from https://calib.io, see: https://forum.opencv.org/t/working-with-commercial-charuco-boards/13624/2

# Create parameters for detection
params = cv2.aruco.DetectorParameters()

saved_images_count = 0
while True:
    frame = cam1.data_stream[0].get_image()
    frame = frame.get_numpy_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    k = cv2.waitKey(5)

    if k == ord('q'):
        break
    elif k == ord('s') and saved_images_count < 3:
        # Save the image
        filename = f"image_{saved_images_count + 1}.png"
        cv2.imwrite(filename, frame)
        print(f"Saved image {saved_images_count + 1}")
        saved_images_count += 1
        
        if saved_images_count == 3:
            print("All 3 images have been saved.")

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect ChArUco board
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    
    if len(corners) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)

    # resize and show frame
    frame = cv2.resize(frame, (int(1920/2),int(1080/2)))
    cv2.imshow(sn, frame)

cv2.destroyAllWindows()