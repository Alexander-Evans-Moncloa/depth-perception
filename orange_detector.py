import cv2
import numpy as np
import torch
import math

# --- Configuration Constants ---

# Camera parameters (IMPORTANT: Calibrate your camera for accuracy!)
# If not calibrated, these are estimates. Frame width is often a rough guess for fx, fy.
# You can get frame_width, frame_height after reading the first frame.
# For now, let's assume a default 640x480 camera resolution.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FX = FRAME_WIDTH  # Approximate focal length in pixels
FY = FRAME_WIDTH  # Approximate focal length in pixels
CX = FRAME_WIDTH / 2
CY = FRAME_HEIGHT / 2
CAMERA_MATRIX = np.array([[FX, 0, CX],
                          [0, FY, CY],
                          [0, 0, 1]], dtype=np.float32)
# Assuming no lens distortion, or you've undistorted the image
DIST_COEFFS = np.zeros((4, 1), dtype=np.float32)

# Object properties
# NOTE: KNOWN_OBJECT_REAL_WIDTH_M should be the average real-world width
# of the object specified by TARGET_CLASS_NAME in meters.
KNOWN_OBJECT_REAL_WIDTH_M = 0.08  # Example: Average diameter of an orange (8 cm)
TARGET_CLASS_NAME = 'cell phone'    # COCO dataset class name for the target object

# Fixed Axis properties
FIXED_AXIS_Z_OFFSET_M = 0.50  # 50 cm in front of the camera
AXIS_DRAW_LENGTH_M = 0.10     # Length of each axis arm for drawing (10 cm)

# Range check for the target object
RANGE_MIN_M = 0.15
RANGE_MAX_M = 0.25

# --- Helper Functions ---

def load_yolo_model(model_name='yolov5s'):
    """
    Loads a YOLOv5 model from torch.hub.
    Args:
        model_name (str): Name of the YOLOv5 model (e.g., 'yolov5s', 'yolov5m').
    Returns:
        torch.hub model.
    """
    try:
        # Attempt to load the model from torch.hub, prioritizing GPU if available
        model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        print(f"YOLOv5 model '{model_name}' loaded successfully.")
        # Check if the model is on GPU
        if next(model.parameters()).is_cuda:
            print("Model is running on GPU.")
        else:
            print("Model is running on CPU.")
        # To list available classes from the loaded model (for reference)
        # print("Available classes:", model.names)
        return model
    except Exception as e:
        print(f"Error loading YOLOv5 model: {e}")
        print("Please ensure you have an internet connection and PyTorch/YOLOv5 installed correctly.")
        print("Try: pip install yolov5")
        return None

def detect_objects(model, frame):
    """
    Performs object detection on a given frame using the YOLOv5 model.
    Args:
        model: The loaded YOLOv5 model.
        frame: The input image/frame (NumPy array).
    Returns:
        A list of detections. Each detection is a dictionary with
        'label', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'.
        Returns None if detection fails.
    """
    if model is None:
        return None
    try:
        # Ensure frame is in the correct format (e.g., BGR) and convert to tensor if needed by model
        # YOLOv5 model(frame) handles conversion internally for common formats like NumPy arrays
        results = model(frame)
        detections = []
        # results.xyxy[0] contains bounding boxes [xmin, ymin, xmax, ymax, confidence, class]
        # Move results to CPU before converting to NumPy
        for det in results.xyxy[0].cpu().numpy():
            xmin, ymin, xmax, ymax, confidence, class_id = det
            label = model.names[int(class_id)]
            detections.append({
                'label': label,
                'confidence': float(confidence),
                'xmin': int(xmin),
                'ymin': int(ymin),
                'xmax': int(xmax),
                'ymax': int(ymax)
            })
        return detections
    except Exception as e:
        print(f"Error during object detection: {e}")
        return None

def estimate_object_depth(bbox_pixel_width, known_object_real_width_m, focal_length_px):
    """
    Estimates the depth of an object based on its apparent size in pixels and known real-world size.
    Formula: Depth = (Focal_Length_px * Real_Object_Width_m) / Object_Width_px
    Args:
        bbox_pixel_width (float): Width of the object's bounding box in pixels.
        known_object_real_width_m (float): Known real-world width of the object in meters.
        focal_length_px (float): Camera's focal length in pixels (typically fx).
    Returns:
        float: Estimated depth in meters. Returns None if bbox_pixel_width is zero or negative.
    """
    if bbox_pixel_width <= 0:
        print("Warning: Bounding box pixel width is zero or negative, cannot estimate depth.")
        return None
    return (focal_length_px * known_object_real_width_m) / bbox_pixel_width

def get_3d_coordinates_camera_frame(u, v, depth_m, camera_matrix):
    """
    Converts 2D image coordinates (u, v) and depth to 3D coordinates in the camera frame.
    Xc = (u - cx) * depth / fx
    Yc = (v - cy) * depth / fy
    Zc = depth
    Args:
        u (float): x-coordinate in the image (pixel).
        v (float): y-coordinate in the image (pixel).
        depth_m (float): Depth of the point in meters.
        camera_matrix (np.array): The 3x3 camera intrinsic matrix.
    Returns:
        np.array: 3D coordinates (X, Y, Z) in meters in the camera frame.
                  Returns None if depth_m is None.
    """
    if depth_m is None:
        return None

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # Ensure fx and fy are not zero to avoid division errors
    if fx == 0 or fy == 0:
         print("Error: Focal length (fx or fy) is zero in camera matrix, cannot calculate 3D coordinates.")
         return None

    X_cam = (u - cx) * depth_m / fx
    Y_cam = (v - cy) * depth_m / fy
    Z_cam = depth_m
    return np.array([X_cam, Y_cam, Z_cam])

def draw_fixed_axis(frame, camera_matrix, dist_coeffs, origin_z_offset_m, axis_length_m):
    """
    Draws a fixed X,Y,Z axis on the frame.
    The axis origin is defined in camera coordinates (0, 0, origin_z_offset_m).
    Args:
        frame: The image (NumPy array) to draw on.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Camera distortion coefficients.
        origin_z_offset_m (float): Z-distance of the axis origin from the camera.
        axis_length_m (float): Length of each axis arm to draw.
    """
    # Define axis points in 3D (camera coordinate system)
    # Origin of the fixed axis
    axis_origin_3d_cam = np.array([0.0, 0.0, origin_z_offset_m], dtype=np.float32)
    # X, Y, Z axis endpoints relative to camera origin
    # Remember: In OpenCV camera coords: X right, Y down, Z forward
    x_axis_end_3d_cam = axis_origin_3d_cam + np.array([axis_length_m, 0, 0], dtype=np.float32)
    y_axis_end_3d_cam = axis_origin_3d_cam + np.array([0, axis_length_m, 0], dtype=np.float32) # Y points down
    z_axis_end_3d_cam = axis_origin_3d_cam + np.array([0, 0, axis_length_m], dtype=np.float32) # Z points further

    axis_points_3d = np.array([
        axis_origin_3d_cam,
        x_axis_end_3d_cam,
        y_axis_end_3d_cam,
        z_axis_end_3d_cam
    ])

    # Project 3D points to 2D image plane
    # rvec and tvec are zero because the 3D points are already in camera coordinates
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)

    try:
        # Ensure camera matrix is valid before projection
        if camera_matrix.shape != (3, 3) or np.linalg.det(camera_matrix) == 0:
             print("Error: Invalid camera matrix for projection.")
             return

        image_points, _ = cv2.projectPoints(axis_points_3d, rvec, tvec, camera_matrix, dist_coeffs)
        image_points = image_points.reshape(-1, 2).astype(int) # Reshape and convert to int

        origin_2d = tuple(image_points[0])
        x_axis_2d = tuple(image_points[1])
        y_axis_2d = tuple(image_points[2])
        z_axis_2d = tuple(image_points[3])

        # Draw lines
        cv2.line(frame, origin_2d, x_axis_2d, (0, 0, 255), 2)  # X-axis: Red
        cv2.line(frame, origin_2d, y_axis_2d, (0, 255, 0), 2)  # Y-axis: Green
        cv2.line(frame, origin_2d, z_axis_2d, (255, 0, 0), 2)  # Z-axis: Blue

        # Draw labels
        cv2.putText(frame, "X_fixed", x_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "Y_fixed", y_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "Z_fixed", z_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(frame, "O_fixed", origin_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    except Exception as e:
        print(f"Error projecting or drawing axis: {e}")

def calculate_angles_and_distance(coords_rel_to_fixed_axis):
    """
    Calculates azimuth, elevation, and distance from relative coordinates.
    Args:
        coords_rel_to_fixed_axis (np.array): [x, y, z] coordinates relative to the fixed axis origin.
    Returns:
        tuple: (azimuth_deg, elevation_deg, distance_m)
               Returns (None, None, None) if input is invalid.
    """
    if coords_rel_to_fixed_axis is None or len(coords_rel_to_fixed_axis) != 3:
        return None, None, None

    x, y, z = coords_rel_to_fixed_axis

    # Azimuthal angle (theta1)
    # atan2(y, x) returns angle in radians between -pi and pi.
    # We want it in [0, 360) degrees.
    azimuth_rad = math.atan2(y, x)
    azimuth_deg = math.degrees(azimuth_rad)
    azimuth_deg = math.fmod(azimuth_deg + 360, 360) # Ensure positive [0, 360)

    # Distance (Euclidean)
    distance_m = math.sqrt(x**2 + y**2 + z**2)

    # Elevation angle (theta2)
    # arccos(z / distance)
    # Argument to arccos must be in [-1, 1]
    if distance_m == 0: # Avoid division by zero and invalid arccos argument
        elevation_deg = 0 # Or handle as undefined, e.g., 90 if z=0 and distance=0
    else:
        # In camera frame, Y is down, Z is forward. Elevation is angle from XY plane towards Z axis.
        # Using atan2(z, sqrt(x^2 + y^2)) is more robust than acos(z/distance) for elevation
        # This gives angle from horizontal plane (XY) towards Z axis.
        # For angle from Z axis towards XY plane, use atan2(sqrt(x^2+y^2), z)
        # Let's use angle from XY plane (elevation above/below horizontal)
        horizontal_distance = math.sqrt(x**2 + y**2)
        if horizontal_distance == 0 and z == 0:
             elevation_deg = 0
        elif horizontal_distance == 0: # Object is directly on Z axis
             elevation_deg = math.degrees(math.atan2(z, 0)) # Will be +90 or -90
        else:
             elevation_rad = math.atan2(z, horizontal_distance)
             elevation_deg = math.degrees(elevation_rad)


    return azimuth_deg, elevation_deg, distance_m

# --- Main Script ---
def main():
    """
    Main function to run the object detection and pose estimation script.
    """
    global CAMERA_MATRIX, CX, CY, FX, FY, FRAME_WIDTH, FRAME_HEIGHT # Added FRAME_WIDTH and FRAME_HEIGHT

    # 1. Load YOLOv5 model
    yolo_model = load_yolo_model()
    if yolo_model is None:
        return

    # 2. Initialize Camera
    cap = cv2.VideoCapture(0) # 0 for default camera
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    # Update camera parameters if actual frame size differs from defaults
    ret, frame_test = cap.read()
    if ret:
        actual_height, actual_width = frame_test.shape[:2]
        # Check if actual frame size is different from initial default constants
        if actual_width != FRAME_WIDTH or actual_height != FRAME_HEIGHT:
            print(f"Camera resolution detected: {actual_width}x{actual_height}. Updating camera matrix.")
            # Update the global constants
            FRAME_WIDTH = actual_width
            FRAME_HEIGHT = actual_height
            FX = FRAME_WIDTH
            FY = FRAME_WIDTH # Often fx is close to fy, and width is a common estimate
            CX = FRAME_WIDTH / 2
            CY = FRAME_HEIGHT / 2
            CAMERA_MATRIX = np.array([[FX, 0, CX],
                                      [0, FY, CY],
                                      [0, 0, 1]], dtype=np.float32)
    else:
        print("Error: Cannot read frame to get actual camera resolution.")
        cap.release()
        return


    print(f"\n--- Starting Detection Loop for '{TARGET_CLASS_NAME}' ---")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Create a copy of the frame for drawing, to keep original clean if needed
        display_frame = frame.copy()

        # 3. Draw the fixed X,Y,Z axis
        draw_fixed_axis(display_frame, CAMERA_MATRIX, DIST_COEFFS,
                        FIXED_AXIS_Z_OFFSET_M, AXIS_DRAW_LENGTH_M)

        # 4. Perform Object Detection
        detections = detect_objects(yolo_model, frame) # Use original frame for detection

        info_text_lines = [] # To store text for screen display

        target_object_found = False

        if detections:
            # Find the first (or largest) target object
            target_object = None
            largest_area = 0
            for det in detections:
                if det['label'] == TARGET_CLASS_NAME:
                    w = det['xmax'] - det['xmin']
                    h = det['ymax'] - det['ymin']
                    area = w * h
                    if area > largest_area: # Prioritize larger (likely closer or better detected) objects
                        largest_area = area
                        target_object = det

            if target_object:
                target_object_found = True
                # A. Draw bounding box
                xmin, ymin, xmax, ymax = target_object['xmin'], target_object['ymin'], target_object['xmax'], target_object['ymax']
                cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2) # Yellow box
                cv2.putText(display_frame, f"{target_object['label']} ({target_object['confidence']:.2f})",
                            (xmin, ymin - 10 if ymin > 10 else ymin + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # B. Estimate target object 3D position in CAMERA coordinates
                bbox_pixel_width = xmax - xmin
                # Using fx for depth estimation as it's often more stable or use (fx+fy)/2
                object_depth_cam_m = estimate_object_depth(bbox_pixel_width, KNOWN_OBJECT_REAL_WIDTH_M, CAMERA_MATRIX[0,0])

                if object_depth_cam_m is not None:
                    # Center of the bounding box in image coordinates
                    u_object_center = (xmin + xmax) / 2
                    v_object_center = (ymin + ymax) / 2

                    # Get (Xc, Yc, Zc) of target object in CAMERA frame
                    object_coords_cam = get_3d_coordinates_camera_frame(u_object_center, v_object_center,
                                                                        object_depth_cam_m, CAMERA_MATRIX)

                    if object_coords_cam is not None:
                        Xc, Yc, Zc_object_cam = object_coords_cam

                        # C. Calculate target object position RELATIVE to the FIXED AXIS ORIGIN
                        # Fixed axis origin in camera coords is (0, 0, FIXED_AXIS_Z_OFFSET_M)
                        x_rel = Xc - 0.0 # X relative to fixed axis origin
                        y_rel = Yc - 0.0 # Y relative to fixed axis origin
                        z_rel = Zc_object_cam - FIXED_AXIS_Z_OFFSET_M # Z relative to fixed axis origin

                        coords_rel_to_fixed_axis = np.array([x_rel, y_rel, z_rel])

                        # D. Calculate Azimuth, Elevation, and Distance
                        azimuth_deg, elevation_deg, dist_to_fixed_origin_m = \
                            calculate_angles_and_distance(coords_rel_to_fixed_axis)

                        if azimuth_deg is not None:
                            # --- Print to Terminal ---
                            print(f"\n--- '{TARGET_CLASS_NAME}' Detected ---")
                            print(f"  BBox Center (px): ({u_object_center:.1f}, {v_object_center:.1f}), Width (px): {bbox_pixel_width}")
                            print(f"  Est. Depth (cam): {object_depth_cam_m:.2f} m")
                            print(f"  Coords (cam frame): Xc={Xc:.2f}m, Yc={Yc:.2f}m, Zc={Zc_object_cam:.2f}m")
                            print(f"  Coords (rel. to fixed axis O): X_rel={x_rel:.2f}m, Y_rel={y_rel:.2f}m, Z_rel={z_rel:.2f}m")
                            print(f"  Azimuth (from fixed O): {azimuth_deg:.1f} deg")
                            print(f"  Elevation (from fixed O): {elevation_deg:.1f} deg")
                            print(f"  Distance (from fixed O): {dist_to_fixed_origin_m:.2f} m")

                            # --- Prepare text for screen display ---
                            info_text_lines.append(f"{TARGET_CLASS_NAME.capitalize()} Rel Coords (X,Y,Z): ({x_rel:.2f}, {y_rel:.2f}, {z_rel:.2f})m")
                            info_text_lines.append(f"Azimuth: {azimuth_deg:.1f} deg")
                            info_text_lines.append(f"Elevation: {elevation_deg:.1f} deg")
                            info_text_lines.append(f"Dist from Fixed O: {dist_to_fixed_origin_m:.2f}m")

                            # E. Range Check
                            if RANGE_MIN_M <= dist_to_fixed_origin_m <= RANGE_MAX_M:
                                range_text = f"{TARGET_CLASS_NAME.capitalize()} in range"
                                range_color = (0, 255, 0) # Green
                                print(f"  Status: {range_text}")
                            else:
                                range_text = f"{TARGET_CLASS_NAME.capitalize()} out of range"
                                range_color = (0, 0, 255) # Red
                            info_text_lines.append(range_text)

                            # Draw range text with specific color
                            cv2.putText(display_frame, range_text, (10, FRAME_HEIGHT - 100),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, range_color, 2)
                        else:
                            info_text_lines.append("Could not calculate angles/distance.")
                    else:
                        info_text_lines.append(f"Could not get {TARGET_CLASS_NAME} 3D cam coords.")
                else:
                    info_text_lines.append(f"Could not estimate {TARGET_CLASS_NAME} depth.")
            # No target object found in detections
            if not target_object_found:
                 info_text_lines.append(f"No '{TARGET_CLASS_NAME}' detected in this frame.")
        else:
            info_text_lines.append("No objects detected by YOLO.")


        # Display info text on screen
        y_offset = 20
        for i, line in enumerate(info_text_lines):
            cv2.putText(display_frame, line, (10, y_offset + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1) # Cyan text


        # 5. Display the resulting frame
        cv2.imshow('Object Detection and Pose Estimation', display_frame)

        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # 6. Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Script finished.")

if __name__ == '__main__':
    main()
