import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def plot_camera(ax, position=(0, 0, 0), rotation=np.eye(3), size=1, color='blue'):
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
    

# Example usage
if __name__ == "__main__":
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    rotation = np.array([[ 0.51593067,  0.01950049, -0.85640836],
 [-0.07405101,  0.99701377, -0.02190886],
 [ 0.85342369 , 0.07472136 , 0.51583401]])
    # Create a camera at position (1, 1, 1) with 45-degree rotation around each axis
    plot_camera(
        ax,
        position=(1, 1, 1),
        rotation=rotation,
        size=0.5,
        color='lightblue'
    )
    plt.show()