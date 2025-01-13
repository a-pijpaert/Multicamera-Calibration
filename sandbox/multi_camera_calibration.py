import numpy as np
import cv2
from cv2 import aruco
import glob
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

class MultiCameraCalibrator:
    def __init__(self, ref_cam_id, board=None, dictionary=None, params=None, show_detection_flag=None):
        """Initialize calibrator with optional CharucoBoard"""
        self.dictionary = dictionary if dictionary is not None else cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
        self.params = params if params is not None else cv2.aruco.DetectorParameters()
        self.board = board if board is not None else self.create_charuco_board(self.dictionary)
        self.show_detection_flag = show_detection_flag
        self.cameras = {}  # Store camera parameters
        self.board_positions = []  # Store board positions relative to reference camera
        self.ref_cam_id = ref_cam_id

    def create_charuco_board(self, dictionary):
        """Create a CharucoBoard for calibration"""
        squares_x = 11
        squares_y = 8
        square_length = 15  # mm
        marker_length = 11  # mm
        board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)
        board.setLegacyPattern(True) # needed for ChAruco boards from https://calib.io, see: https://forum.opencv.org/t/working-with-commercial-charuco-boards/13624/2
        return board
    
    def get_board_size(self):
        """Get physical size of the board"""
        squares_x, squares_y = self.board.getChessboardSize()
        square_length = self.board.getSquareLength()
        return (squares_x * square_length, squares_y * square_length)
    
    def detect_markers(self, image_path):
        """Detect CharucoBoard corners in an image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.params)
        
        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, self.board)
            
            if charuco_corners is not None and charuco_ids is not None and len(charuco_corners) > 0:
                cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

        # resize and show image
        image = cv2.resize(image, (int(1920/2),int(1080/2)))
        if self.show_detection_flag:
            cv2.imshow("image", image)
            cv2.waitKey(1)
        
        return image, charuco_corners, charuco_ids
    
    def calibrate_cameras(self, image_sets):
        """
        Calibrate multiple cameras simultaneously
        
        Args:
            image_sets: Dictionary with camera IDs as keys and lists of image paths as values
                       First camera in the dict is considered the reference camera
        """
        # Initialize storage for all detected points
        all_corners = {cam_id: [] for cam_id in image_sets.keys()}
        all_ids = {cam_id: [] for cam_id in image_sets.keys()}
        object_points = []
        image_size = None
        
        # # Get reference camera ID (first camera) TODO: change this based on an input
        # ref_cam_id = list(image_sets.keys())[0]
        
        # Process all images
        num_images = len(image_sets[self.ref_cam_id])
        for frame_idx in range(num_images):
            frame_corners = {}
            frame_ids = {}
            valid_frame = True
            
            # Detect markers in all cameras for this frame
            for cam_id, images in image_sets.items():
                if frame_idx >= len(images):
                    valid_frame = False
                    break
                    
                image, corners, ids = self.detect_markers(images[frame_idx])
                
                if corners is None or ids is None:
                    valid_frame = False
                    break
                
                if image_size is None:
                    image_size = image.shape[:2][::-1]
                
                frame_corners[cam_id] = corners
                frame_ids[cam_id] = ids
            
            if not valid_frame:
                continue
            
            # Find common points across all cameras
            common_ids = set(frame_ids[self.ref_cam_id].ravel())
            for cam_id in image_sets.keys():
                if cam_id != self.ref_cam_id:
                    common_ids &= set(frame_ids[cam_id].ravel())
            
            if not common_ids:
                continue
                
            # Get object and image points for common IDs
            common_ids = sorted(list(common_ids))
            board_points = self.board.getChessboardCorners()
            obj_points = board_points[common_ids].astype(np.float32)
            
            # Store points for each camera
            for cam_id in image_sets.keys():
                cam_ids = frame_ids[cam_id].ravel()
                idx = [list(cam_ids).index(id) for id in common_ids]
                all_corners[cam_id].append(frame_corners[cam_id][idx])
                all_ids[cam_id].append(np.array(common_ids))
                if len(object_points) < len(all_corners[self.ref_cam_id]):
                    object_points.append(obj_points)
        
        # Calibrate each camera individually first
        for cam_id in image_sets.keys():
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                object_points,
                all_corners[cam_id],
                image_size,
                None,
                None
            )
            self.cameras[cam_id] = {
                'matrix': mtx,
                'dist_coeffs': dist,
                'rvecs': rvecs,
                'tvecs': tvecs
            }
        
        # Perform stereo calibration between reference camera and each other camera
        ref_matrix = self.cameras[self.ref_cam_id]['matrix']
        ref_dist = self.cameras[self.ref_cam_id]['dist_coeffs']
        
        for cam_id in image_sets.keys():
            self.cameras[cam_id].update({
                'all_corners': all_corners[cam_id],
                'all_ids': all_ids[cam_id]
            })
            if cam_id == self.ref_cam_id:
                # Store identity transform for reference camera
                self.cameras[cam_id].update({
                    'R': np.eye(3),
                    'T': np.zeros((3, 1)),
                    'E': None,
                    'F': None
                })
                continue
                
            ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                object_points,
                all_corners[self.ref_cam_id],
                all_corners[cam_id],
                ref_matrix,
                ref_dist,
                self.cameras[cam_id]['matrix'],
                self.cameras[cam_id]['dist_coeffs'],
                image_size,
                None,
                None,
                None,
                flags=cv2.CALIB_FIX_INTRINSIC
            )
            
            self.cameras[cam_id].update({
                'R': R,
                'T': T,
                'E': E,
                'F': F
            })

    # def draw_coordinate_system(self, ax, position, rotation_matrix, scale=20, label=""):
    #     """
    #     Draw coordinate system axes at given position with given orientation
    #     Red = X, Green = Y, Blue = Z
    #     """
    #     # Define the axes endpoints in local coordinates
    #     axes = np.array([[scale, 0, 0],  # X-axis
    #                     [0, scale, 0],  # Y-axis
    #                     [0, 0, scale]]) # Z-axis
        
    #     # Transform endpoints to world coordinates
    #     axes_world = np.dot(rotation_matrix, axes.T).T + position
        
    #     # Plot each axis
    #     colors = ['red', 'green', 'blue']
    #     # labels = [f'{label} X', f'{label} Y', f'{label} Z'] if label else ['X', 'Y', 'Z']
        
    #     for i, color in enumerate(colors):
    #         ax.quiver(position[0], position[1], position[2],
    #                 axes_world[i,0] - position[0],
    #                 axes_world[i,1] - position[1],
    #                 axes_world[i,2] - position[2],
    #                 color=color, label=label)
    #     if label:
    #         ax.text(
    #             position[0], position[1], position[2],
    #             label,
    #             fontsize=12,
    #             color='black'
    #         )
    
    def draw_board_plane(self, ax, position, rotation_matrix, color, alpha=0.3, label=""):
        """
        Draw a colored plane representing the CharucoBoard
        """
        # Define board corners in local coordinates
        board_size = self.get_board_size()
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
        
    def calculate_board_positions(self):
        """
        Calculate CharucoBoard positions relative to the reference camera
        """
        board_positions = []
        all_corners = self.cameras[self.ref_cam_id]['all_corners']
        all_ids = self.cameras[self.ref_cam_id]['all_ids']

        for corners, ids in zip(all_corners, all_ids):
            if corners is not None and ids is not None:
                # Estimate board pose relative to left camera
                success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                    corners,
                    ids,
                    self.board,
                    self.cameras[self.ref_cam_id]['matrix'],
                    self.cameras[self.ref_cam_id]['dist_coeffs'],
                    None,
                    None
                )
                
                if success:
                    board_positions.append((rvec, tvec))
        
        return board_positions

    def plot_camera(self, ax, position=(0, 0, 0), rotation=np.eye(3), size=1, color='blue'):
        """
        Plot a 3D camera representation.
        
        Args:
            position (tuple): (x, y, z) position of camera center
            rotation (np.ndarray): 3x3 rotation matrix
            size (float): Scale factor for camera size
            color (str): Color of the camera wireframe
        """
        # Camera vertices (basic pyramid shape)
        vertices = np.array([
            [0, 0, 0],  # Camera center
            [1, 1, 2],  # Top right
            [-1, 1, 2], # Top left
            [-1, -1, 2],# Bottom left
            [1, -1, 2], # Bottom right
        ]) * size
        
        vertices = (vertices @ rotation.T) + position
        
        # Define faces using vertex indices
        faces = [
            [vertices[0], vertices[1], vertices[2]],  # Top face
            [vertices[0], vertices[2], vertices[3]],  # Left face
            [vertices[0], vertices[3], vertices[4]],  # Bottom face
            [vertices[0], vertices[4], vertices[1]],  # Right face
            [vertices[1], vertices[2], vertices[3], vertices[4]]  # Back face
        ]
        
        # Plot the camera as a collection of polygons
        camera = Poly3DCollection(faces, alpha=0.3, facecolor=color, edgecolor='black')
        ax.add_collection3d(camera)
        
        # Add coordinate frame arrows
        arrow_length = size * 0.5

        # Transform the coordinate arrows by the rotation matrix
        arrows = np.array([[arrow_length, 0, 0],
                        [0, arrow_length, 0],
                        [0, 0, arrow_length]]) @ rotation.T
        
        ax.quiver(*position, *arrows[0], color='red')
        ax.quiver(*position, *arrows[1], color='green')
        ax.quiver(*position, *arrows[2], color='blue')

    def visualize_setup(self):
        """Visualize the multi-camera setup"""
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot cameras
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.cameras)))
        
        for (cam_id, params), color in zip(self.cameras.items(), colors):
            # Get camera position and orientation
            if cam_id == self.ref_cam_id:
                rotation = np.eye(3)
                position = np.zeros(3)
            else:
                rotation = params['R']
                translation = params['T'].ravel()
                position = -np.dot(rotation.T, translation)
            
            # self.draw_coordinate_system(ax, position, rotation, label=cam_id)
            self.plot_camera(ax, position, np.linalg.inv(rotation), size=20)

        # Get board positions
        board_positions = self.calculate_board_positions()
        
        # Define colors for the boards
        colors = plt.cm.rainbow(np.linspace(0, 1, len(board_positions) if board_positions else 1))
            
        # If we have board positions, plot them
        if board_positions is not None:
            for i, ((rvec, tvec), color) in enumerate(zip(board_positions, colors)):
                # Convert rotation vector to matrix
                board_rot, _ = cv2.Rodrigues(rvec)
                board_pos = tvec.ravel()
                
                # Draw board as a colored plane
                self.draw_board_plane(ax, board_pos, board_rot, color, 
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
        # ax.legend()
        
        return fig, ax
    
    def save_calibration(self, filename='multi_camera_calibration.pkl'):
        """Save calibration results"""
        with open(filename, 'wb') as f:
            pickle.dump(self.cameras, f)
    
    def load_calibration(self, filename='multi_camera_calibration.pkl'):
        """Load calibration results"""
        with open(filename, 'rb') as f:
            self.cameras = pickle.load(f)

def main():
    # Create calibrator
    calibrator = MultiCameraCalibrator(ref_cam_id='cam4', show_detection_flag=False)
    
    # Define image sets for each camera
    image_sets = {
        'cam1': sorted(glob.glob('images/camera1/*.png')),
        'cam2': sorted(glob.glob('images/camera2/*.png')),
        'cam3': sorted(glob.glob('images/camera3/*.png')),  
        'cam4': sorted(glob.glob('images/camera4/*.png')),  # Add more cameras as needed
    }
    
    # Perform calibration
    print("Calibrating cameras...")
    calibrator.calibrate_cameras(image_sets)
    
    # Save results
    calibrator.save_calibration()
    
    # Visualize setup
    print("Creating visualization...")
    fig, ax = calibrator.visualize_setup()
    plt.savefig('multi_camera_setup.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()