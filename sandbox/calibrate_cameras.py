import numpy as np
import cv2
from cv2 import aruco
import glob
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import os

def create_charuco_board():
    """
    Create a CharUco board for calibration
    """
    squares_x = 11
    squares_y = 8
    square_length = 15  # mm
    marker_length = 11  # mm
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), 
                                   square_length, marker_length, dictionary)
    board.setLegacyPattern(True) # needed for ChAruco boards from https://calib.io, see: https://forum.opencv.org/t/working-with-commercial-charuco-boards/13624/2
    params = cv2.aruco.DetectorParameters()

    return board, params, dictionary

def detect_charuco_points(image_path, board, params, dictionary):
    """
    Detect CharUco board corners in an image
    """

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    
    if len(corners) > 0:
        cv2.aruco.drawDetectedMarkers(image, corners, ids)
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
        
        if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

    # resize and show image
    image = cv2.resize(image, (int(1920/2),int(1080/2)))
    cv2.imshow("image", image)
    cv2.waitKey(1)
    
    return image, charuco_corners, charuco_ids

def stereo_calibrate(board, params, dictionary, left_images, right_images):
    """
    Perform stereo calibration using CharUco board images
    """
    # Lists to store points
    object_points = []
    img_points_left = []
    img_points_right = []
    image_size = None

    # Get object points for the board
    board_points = board.getChessboardCorners()

    # Process all image pairs
    for left_path, right_path in zip(left_images, right_images):
        # Process left image
        left_img, corners_left, ids_left = detect_charuco_points(left_path, board, params, dictionary)
        
        # Process right image
        right_img, corners_right, ids_right = detect_charuco_points(right_path, board, params, dictionary)
        
        # Store image size
        if image_size is None:
            image_size = left_img.shape[:2][::-1]  # width, height
        
        # Check if corners were detected in both images
        if (corners_left is not None and corners_right is not None and 
            ids_left is not None and ids_right is not None):

            # Make sure we have the same points in both images
            common_ids = np.intersect1d(ids_left, ids_right)

            if len(common_ids) > 0:
                # Get indices of common points
                left_idx = np.where(np.isin(ids_left, common_ids))[0]
                right_idx = np.where(np.isin(ids_right, common_ids))[0]

                # Get the corresponding object points
                current_obj_points = board_points[common_ids]
                
                # Store the points
                object_points.append(current_obj_points.astype(np.float32))
                img_points_left.append(corners_left[left_idx].reshape(-1, 2).astype(np.float32))
                img_points_right.append(corners_right[right_idx].reshape(-1, 2).astype(np.float32))
    
    if not object_points:
        raise Exception("No valid image pairs found for calibration!")
    
    # Initial camera matrices
    camera_matrix_left = np.eye(3)
    camera_matrix_right = np.eye(3)
    
    # Distortion coefficients
    dist_coeffs_left = np.zeros((5, 1))
    dist_coeffs_right = np.zeros((5, 1))
    
        # Calibrate individual cameras first
    ret_left, camera_matrix_left, dist_coeffs_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        object_points, img_points_left, image_size, camera_matrix_left, dist_coeffs_left
    )
    
    ret_right, camera_matrix_right, dist_coeffs_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        object_points, img_points_right, image_size, camera_matrix_right, dist_coeffs_right
    )
    
    # Perform stereo calibration
    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    
    ret, camera_matrix_left, dist_coeffs_left, camera_matrix_right, dist_coeffs_right, R, T, E, F = \
        cv2.stereoCalibrate(
            object_points,
            img_points_left,
            img_points_right,
            camera_matrix_left,
            dist_coeffs_left,
            camera_matrix_right,
            dist_coeffs_right,
            image_size,
            None,
            None,
            None,
            criteria=stereocalib_criteria,
            flags=cv2.CALIB_FIX_INTRINSIC
        )
    
    # Compute rectification transforms
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        camera_matrix_left, dist_coeffs_left,
        camera_matrix_right, dist_coeffs_right,
        image_size, R, T
    )


    return {
        'camera_matrix_left': camera_matrix_left,
        'dist_coeffs_left': dist_coeffs_left,
        'camera_matrix_right': camera_matrix_right,
        'dist_coeffs_right': dist_coeffs_right,
        'R': R,
        'T': T,
        'E': E,
        'F': F,
        'R1': R1,
        'R2': R2,
        'P1': P1,
        'P2': P2,
        'Q': Q,
        'roi1': roi1,
        'roi2': roi2
    }

def draw_coordinate_system(ax, position, rotation_matrix, scale=0.1, label=""):
    """
    Draw coordinate system axes at given position with given orientation
    Red = X, Green = Y, Blue = Z
    """
    # Define the axes endpoints in local coordinates
    axes = np.array([[scale, 0, 0],  # X-axis
                    [0, scale, 0],  # Y-axis
                    [0, 0, scale]]) # Z-axis
    
    # Transform endpoints to world coordinates
    axes_world = np.dot(rotation_matrix, axes.T).T + position
    
    # Plot each axis
    colors = ['red', 'green', 'blue']
    # labels = [f'{label} X', f'{label} Y', f'{label} Z'] if label else ['X', 'Y', 'Z']
    
    for i, color in enumerate(colors):
        ax.quiver(position[0], position[1], position[2],
                 axes_world[i,0] - position[0],
                 axes_world[i,1] - position[1],
                 axes_world[i,2] - position[2],
                 color=color, label=None)
        
def draw_board_plane(ax, position, rotation_matrix, board_size, color, alpha=0.3, label=""):
    """
    Draw a colored plane representing the CharucoBoard
    """
    # Define board corners in local coordinates
    half_width = board_size[0] / 2
    half_height = board_size[1] / 2
    corners = np.array([
        [-half_width, -half_height, 0],
        [half_width, -half_height, 0],
        [half_width, half_height, 0],
        [-half_width, half_height, 0]
    ])
    
    # Transform corners to world coordinates
    corners_world = np.dot(rotation_matrix, corners.T).T + position
    
    # Create vertices for plotting
    x = corners_world[:, 0]
    y = corners_world[:, 1]
    z = corners_world[:, 2]
    
    # Create vertices and faces for the plane
    verts = [list(zip(x, y, z))]
    
    # Plot the plane
    plane = art3d.Poly3DCollection(verts)
    plane.set_color(color)
    plane.set_alpha(alpha)
    ax.add_collection3d(plane)
    
    # Add a label point
    if label:
        center = position
        ax.text(center[0], center[1], center[2], label, 
                color='black', fontsize=10, ha='center', va='center')

def visualize_stereo_setup_with_boards(calib_results, board_positions=None, board_size=(28, 20)):
    """
    Create a 3D visualization of the stereo camera setup with coordinate systems
    and CharucoBoard positions
    """
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Left camera is at origin
    left_pos = np.array([0, 0, 0])
    left_rot = np.eye(3)  # Identity matrix as it's our reference frame
    draw_coordinate_system(ax, left_pos, left_rot, scale=20, label="Left")
    
    # Right camera position and orientation
    right_pos = calib_results['T'].ravel()
    right_rot = calib_results['R']
    draw_coordinate_system(ax, right_pos, right_rot, scale=20, label="Right")
    
    # Draw line connecting cameras
    ax.plot([left_pos[0], right_pos[0]], 
            [left_pos[1], right_pos[1]], 
            [left_pos[2], right_pos[2]], 
            'k--')
    
    # Define colors for the boards
    colors = plt.cm.rainbow(np.linspace(0, 1, len(board_positions) if board_positions else 1))
        
    # If we have board positions, plot them
    if board_positions is not None:
        for i, ((rvec, tvec), color) in enumerate(zip(board_positions, colors)):
            # Convert rotation vector to matrix
            board_rot, _ = cv2.Rodrigues(rvec)
            board_pos = tvec.ravel()
            
            # Draw board as a colored plane
            draw_board_plane(ax, board_pos, board_rot, board_size, color, 
                           label=f'Board {i+1}')
    
    # Set equal aspect ratio and labels
    ax.set_aspect('equal')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('Stereo Camera Setup with Coordinate Systems')
    
    # Adjust the view
    ax.view_init(elev=-45, azim=-150, roll=70)
    
    # Add legend
    ax.legend()
    
    return fig, ax

def calculate_board_positions(calib_results, board, params, dictionary, left_images, right_images):
    """
    Calculate CharucoBoard positions relative to the left camera
    """
    board_positions = []
    
    for left_path, right_path in zip(left_images, right_images):
        # Detect board in left image
        left_img, corners_left, ids_left = detect_charuco_points(left_path, board, params, dictionary)
        
        if corners_left is not None and ids_left is not None:
            # Estimate board pose relative to left camera
            success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                corners_left,
                ids_left,
                board,
                calib_results['camera_matrix_left'],
                calib_results['dist_coeffs_left'],
                None,
                None
            )
            
            if success:
                board_positions.append((rvec, tvec))
    
    return board_positions

def main():
    # Create CharUco board
    board, params, dictionary = create_charuco_board()
    
    # Get lists of image pairs
    left_images = sorted(glob.glob('images/camera2/*.png'))  # Adjust path and extension as needed
    right_images = sorted(glob.glob('images/camera3/*.png'))  # Adjust path and extension as needed
    
    if len(left_images) != len(right_images):
        print("Error: Number of left and right images don't match!")
        return
    
    print(f"Found {len(left_images)} image pairs")
    
    # Perform stereo calibration
    print("Performing stereo calibration...")
    calib_results = stereo_calibrate(board, params, dictionary, left_images, right_images)

    # Calculate board positions
    print("Calculating board positions...")
    board_positions = calculate_board_positions(calib_results, board, params, dictionary, left_images, right_images)

    # Save calibration results
    with open('stereo_calibration.pkl', 'wb') as f:
        pickle.dump(calib_results, f)
    
    print("\nCalibration results:")
    print(f"Translation vector (mm): {calib_results['T'].ravel()}")
    print(f"Rotation matrix:\n{calib_results['R']}")
    
    # Get board size automatically
    squares_x, squares_y = board.getChessboardSize()
    square_length = board.getSquareLength()
    board_size = (squares_x * square_length, squares_y * square_length)

    # Visualize stereo setup
    fig, ax = visualize_stereo_setup_with_boards(calib_results, board_positions, board_size)
    plt.savefig('stereo_setup.png')
    plt.show()

if __name__ == "__main__":
    main()