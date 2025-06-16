from sandbox.multi_camera_calibration import MultiCameraCalibrator
import glob
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Create calibrator
    calibrator = MultiCameraCalibrator(ref_cam_id='cam2', show_detection_flag=False)
    
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
    
    # # Save results
    # calibrator.save_calibration()
    
    # # Visualize setup
    # print("Creating visualization...")
    # fig, ax = calibrator.visualize_setup()
    # plt.savefig('multi_camera_setup.png', dpi=300, bbox_inches='tight')
    # plt.show()