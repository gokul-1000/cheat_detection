import cv2
import dlib
import numpy as np
import os
import time
import win32gui  # type: ignore
import win32process  # type: ignore
import psutil  # type: ignore

from flask import Flask, render_template, Response, jsonify
import threading

app = Flask(__name__)

# --- Dlib Face and Landmark Initialization ---
detector = dlib.get_frontal_face_detector()
# Ensure shape_predictor_68_face_landmarks.dat is in the same directory
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# --- YOLO Object Detection Initialization ---
# Using yolov4 for better accuracy, but requires larger files and more processing power.
YOLO_MODEL_NAME = "yolov4" 

# Global variables for YOLO (loaded once to avoid re-loading per frame)
yolo_net = None
yolo_classes = []
yolo_output_layers = []
yolo_colors = None

def load_yolo():
    """
    Loads the YOLO neural network model, including weights, configuration, and class names.
    Ensures model files exist in the current directory.
    Raises FileNotFoundError if any required file is missing.
    """
    global yolo_net, yolo_classes, yolo_output_layers, yolo_colors
    print(f"Loading {YOLO_MODEL_NAME} model...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Paths for YOLO model files.
    weights_path = os.path.join(current_dir, f"{YOLO_MODEL_NAME}.weights") 
    config_path = os.path.join(current_dir, f"{YOLO_MODEL_NAME}.cfg")    
    names_path = os.path.join(current_dir, "coco.names.txt") 

    # Verify file existence before attempting to load
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Error: YOLO weights file not found at {weights_path}. Please download {YOLO_MODEL_NAME}.weights.")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Error: YOLO config file not found at {config_path}. Please download {YOLO_MODEL_NAME}.cfg.")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Error: YOLO names file not found at {names_path}.")

    try:
        # Load YOLO model from files
        net = cv2.dnn.readNet(weights_path, config_path)
        # Optimize for CPU performance (can be changed to DNN_TARGET_CUDA if GPU is available)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"{YOLO_MODEL_NAME} net loaded successfully!")
    except cv2.error as e:
        print(f"OpenCV Error during {YOLO_MODEL_NAME} net loading: {e}")
        print("This often means corrupted or incompatible .weights/.cfg files. Please check downloads or model compatibility.")
        raise

    classes = []
    try:
        # Load class names (e.g., "person", "cell phone")
        with open(names_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        print("coco.names loaded successfully!")
    except Exception as e:
        print(f"Error loading coco.names: {e}. Check file permissions or corruption.")
        raise

    # Determine YOLO output layers
    layer_names = net.getLayerNames()
    unconnected_layers = net.getUnconnectedOutLayers()
    if isinstance(unconnected_layers, np.ndarray):
        output_layers = [layer_names[i - 1] for i in unconnected_layers.flatten()]
    else: 
        output_layers = [layer_names[i[0] - 1] for i in unconnected_layers]
    print(f"{YOLO_MODEL_NAME} output layers determined successfully!")

    # Store loaded model and data in global variables
    yolo_net = net
    yolo_classes = classes
    yolo_output_layers = output_layers
    # Generate random colors for bounding boxes for each class
    yolo_colors = np.random.uniform(0, 255, size=(len(yolo_classes), 3))

def detect_objects(img, net, output_layers, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Performs object detection on an image using the loaded YOLO network.
    Applies Non-Maximum Suppression (NMS) to filter redundant bounding boxes.
    
    Args:
        img (np.array): The input image frame.
        net (cv2.dnn.Net): The loaded YOLO neural network.
        output_layers (list): Names of the YOLO output layers.
        confidence_threshold (float): Minimum confidence to consider a detection.
        nms_threshold (float): IoU threshold for Non-Maximum Suppression.

    Returns:
        tuple: A tuple containing lists of bounding boxes, confidences, and class IDs.
    """
    height, width, _ = img.shape
    # Create a 4D blob from the image for network input
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    # Perform forward pass to get detections
    outputs = net.forward(output_layers)

    boxes = []
    confs = []
    class_ids = []

    # Process each detection output
    for output in outputs:
        for detect in output:
            scores = detect[5:] # Class scores are from index 5 onwards
            class_id = np.argmax(scores) # Get the class with the highest score
            conf = scores[class_id] # Get the confidence for that class
            if conf > confidence_threshold:
                # Calculate bounding box coordinates
                center_x = int(detect[0] * width)
                center_y = int(detect[1] * height)
                w = int(detect[2] * width)
                h = int(detect[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confs.append(float(conf))
                class_ids.append(class_id)
    
    # Apply Non-Maximum Suppression to remove redundant overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confs, confidence_threshold, nms_threshold)
    if len(indices) > 0:
        indices = indices.flatten() # Flatten the indices array
        return [boxes[i] for i in indices], [confs[i] for i in indices], [class_ids[i] for i in indices]
    return [], [], []

# --- Cheating Detection Parameters ---
LOOK_AWAY_THRESHOLD_SEC = 1.5      # Time before 'look away' is considered a cheating incident
NO_FACE_THRESHOLD_SEC = 2.0        # Time before 'no face' is considered a cheating incident
MULTIPLE_FACES_THRESHOLD_SEC = 1.0 # Time before 'multiple faces' is considered a cheating incident

YAW_THRESHOLD = 20  # Angle threshold for head yaw (left/right)
PITCH_THRESHOLD = 15 # Angle threshold for head pitch (up/down)

PHONE_DETECTION_CONFIDENCE = 0.6  # Confidence threshold for detecting a "cell phone"
PERSON_DETECTION_CONFIDENCE = 0.5 # Confidence threshold for detecting a "person"

MAX_STRIKES = 3 # Number of warnings before test is permanently blocked.
# (1st, 2nd, 3rd cheating incidents issue a warning, 4th incident blocks)

# --- OS Level Monitoring Configuration (Windows-specific) ---
ALLOWED_WINDOW_TITLES = [
    "My Mock Online Exam", # The title of your HTML page
    "flask", "python.exe", "powershell.exe", "cmd.exe", # Development/System Processes
    "Pycharm", "Code" # IDEs (if relevant during exam)
]

ALLOWED_PROCESS_NAMES = [
    "chrome.exe", "firefox.exe", "msedge.exe", "python.exe",
    "YourTestApp.exe" # Replace with actual exam software executable name if any
]

APP_SWITCH_THRESHOLD_SEC = 2.0 # Time threshold for app switching to be considered cheating

# --- State Variables (Global for persistent state across frames) ---
# Dlib timers for continuous violations
last_look_away_time = None
last_no_face_time = None
last_multiple_faces_time = None
# OS monitoring timer
last_app_switch_time = None 

# Global status for Flask to provide to frontend via JSON
current_overall_status = "STATUS: INITIALIZING..."
current_overall_color = (128, 128, 128) # Grey (RGB)
current_alert_sound = "none" # "light", "severe", "none"

strike_count = 0 # Tracks the number of cheating incidents (strikes)
_last_strike_timestamp = 0 # Timestamp of the last time a strike was issued
_strike_cooldown_duration = 5 # Seconds to wait before another strike can be issued for a continuous violation
_last_cheating_reason_for_strike = "" # Stores the specific cheating reason that last triggered a strike

# Lock for thread-safe access to global status variables
lock = threading.Lock() 

# --- OS Level Monitoring Function (Windows-specific) ---
def get_active_window_info():
    """
    Retrieves the title and process name of the currently active foreground window.
    This function is Windows-specific due to `win32gui` and `win32process`.
    """
    try:
        hwnd = win32gui.GetForegroundWindow() # Get handle of the foreground window
        if hwnd:
            window_title = win32gui.GetWindowText(hwnd) # Get window title
            thread_id, process_id = win32process.GetWindowThreadProcessId(hwnd) # Get PID
            process_name = "N/A"
            try:
                process = psutil.Process(process_id) # Get process details using psutil
                process_name = process.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # Handle cases where process is not found or access denied
                pass
            return window_title, process_name
        return None, None # No foreground window found
    except Exception as e:
        # Log or print error, but don't crash the application
        # print(f"Error getting active window info: {e}")
        return None, None

# --- Incident Logging ---
LOG_FILE_NAME = "cheating_incidents.log"

def log_incident(incident_type, details=""):
    """
    Logs a cheating incident to a file and prints to console.
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] INCIDENT: {incident_type}"
    if details:
        log_message += f" - Details: {details}"
    
    print(f"LOGGED: {log_message}")
    try:
        with open(LOG_FILE_NAME, "a") as f:
            f.write(log_message + "\n")
    except Exception as e:
        print(f"Error writing to log file {LOG_FILE_NAME}: {e}")

# --- Head Pose Estimation and Gaze Direction ---
# 3D model points of a generic face (standard points for head pose estimation)
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),            # Nose tip
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),    # Left eye left corner
    (225.0, 170.0, -135.0),     # Right eye right corner
    (-150.0, -150.0, -125.0),   # Left mouth corner
    (150.0, -150.0, -125.0)     # Right mouth corner
], dtype="double")

def get_head_pose(landmarks, frame_size):
    """
    Estimates head pose (pitch, yaw, roll) using Dlib landmarks and solvePnP.
    
    Args:
        landmarks (dlib.full_object_detection): 68 facial landmarks detected by Dlib.
        frame_size (tuple): (height, width) of the video frame.

    Returns:
        tuple: pitch, yaw, roll angles, rotation/translation vectors, camera matrix, dist coeffs.
    """
    # 2D image points from Dlib facial landmarks (specific points chosen for pose estimation)
    image_points = np.array([
        (landmarks.part(30).x, landmarks.part(30).y), # Nose tip
        (landmarks.part(8).x, landmarks.part(8).y),   # Chin
        (landmarks.part(36).x, landmarks.part(36).y), # Left eye left corner
        (landmarks.part(45).x, landmarks.part(45).y), # Right eye right corner
        (landmarks.part(48).x, landmarks.part(48).y), # Left mouth corner
        (landmarks.part(54).x, landmarks.part(54).y)  # Right mouth corner
    ], dtype="double")

    # Camera matrix calculation (approximated for standard webcam)
    focal_length = frame_size[1] # Assuming focal length is roughly frame width
    center = (frame_size[1]/2, frame_size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

    # Solve for rotation and translation vectors
    (success, rotation_vector, translation_vector) = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # Convert rotation vector to rotation matrix and then to Euler angles (pitch, yaw, roll)
    rmat, jacobian = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rmat, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix, camera_matrix, dist_coeffs)[6]
    
    pitch = euler_angles[0]
    yaw = euler_angles[1]
    roll = euler_angles[2]

    return pitch, yaw, roll, rotation_vector, translation_vector, camera_matrix, dist_coeffs

def get_gaze_direction(pitch, yaw):
    """
    Determines gaze direction based on estimated head pose angles.
    """
    gaze_direction = "Forward"
    is_gaze_threat = False

    if yaw > YAW_THRESHOLD:
        gaze_direction = "Left"
        is_gaze_threat = True
    elif yaw < -YAW_THRESHOLD:
        gaze_direction = "Right"
        is_gaze_threat = True

    # Pitch (up/down) is generally less critical for "looking away"
    if pitch > PITCH_THRESHOLD:
        if gaze_direction != "Forward":
            gaze_direction += "/Down (Allowed)"
        else:
            gaze_direction = "Down (Allowed)"
    elif pitch < -PITCH_THRESHOLD:
        if gaze_direction != "Forward":
            gaze_direction += "/Up (Allowed)"
        else:
            gaze_direction = "Up (Allowed)"

    return gaze_direction, is_gaze_threat

# Global webcam object
cap = None 
# Global variable to store the latest processed frame for streaming
output_frame = None 

def generate_frames():
    """
    Generator function to continuously capture webcam frames,
    perform cheating detection, and yield processed frames as JPEG bytes.
    """
    global last_look_away_time, last_no_face_time, last_multiple_faces_time, last_app_switch_time
    global current_overall_status, current_overall_color, current_alert_sound
    global strike_count, _last_strike_timestamp, _last_cheating_reason_for_strike
    global output_frame 

    # Handle webcam initialization failure
    if cap is None or not cap.isOpened():
        print("Error: Webcam not initialized or opened.")
        with lock: # Acquire lock before modifying globals
            current_overall_status = "ERROR: WEBCAM NOT FOUND!"
            current_overall_color = (0, 0, 255) # Red for critical error
            current_alert_sound = "severe"
        # Yield a blank frame to prevent browser from breaking, then exit generator
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes() + b'\r\n')
        time.sleep(1) # Prevent busy-waiting if camera is truly unavailable
        return # Exit the generator if webcam is not available

    print("Starting frame generation for webcam...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, attempting to re-initialize webcam...")
            if cap is not None:
                cap.release()
            time.sleep(1) # Give camera a moment to reset
            start_webcam() # Attempt to re-initialize
            # If re-initialization fails, `cap` will be None, and the next loop iteration will handle it
            continue

        current_time = time.time()
        height, width, _ = frame.shape
        frame_size = (height, width)

        # --- OS-level Monitoring for Tab/Application Switching ---
        tab_cheating_detected_os = False
        active_window_title, active_process_name = get_active_window_info()

        if active_window_title and active_process_name:
            is_allowed_window = any(allowed_title_part.lower() in active_window_title.lower() for allowed_title_part in ALLOWED_WINDOW_TITLES)
            is_allowed_process = any(allowed_process_exe.lower() == active_process_name.lower() for allowed_process_exe in ALLOWED_PROCESS_NAMES)
            is_currently_allowed = is_allowed_window and is_allowed_process

            if not is_currently_allowed:
                if last_app_switch_time is None:
                    last_app_switch_time = current_time 
                
                # App switch detected if duration exceeds threshold
                if (current_time - last_app_switch_time) > APP_SWITCH_THRESHOLD_SEC:
                    tab_cheating_detected_os = True
            else:
                last_app_switch_time = None # Reset timer if back to allowed app

        # --- Dlib Face and Gaze Detection ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        num_faces_dlib = len(faces)

        current_dlib_status_text = ""
        current_dlib_status_color = (0, 255, 0) # Default: Green (OK)
        gaze_status_text = "Gaze: N/A"
        gaze_status_color = (255, 255, 0) # Default: Yellow (Info)

        # Logic for No Face, Multiple Faces, Gaze Away based on thresholds
        if num_faces_dlib == 0:
            if last_no_face_time is None:
                last_no_face_time = current_time
            if (current_time - last_no_face_time) > NO_FACE_THRESHOLD_SEC:
                current_dlib_status_text = "No Face Found"
                current_dlib_status_color = (0, 0, 255) # Red (Cheating)
            else:
                current_dlib_status_text = "Warning: No Face (Briefly)"
                current_dlib_status_color = (0, 165, 255) # Orange (Warning)
            # Reset other timers as only one Dlib state can be active
            last_look_away_time = None
            last_multiple_faces_time = None
            
        elif num_faces_dlib > 1:
            if last_multiple_faces_time is None:
                last_multiple_faces_time = current_time
            if (current_time - last_multiple_faces_time) > MULTIPLE_FACES_THRESHOLD_SEC:
                current_dlib_status_text = "Multiple Faces Detected"
                current_dlib_status_color = (0, 0, 255) # Red (Cheating)
            else:
                current_dlib_status_text = "Warning: Multiple Faces (Briefly)"
                current_dlib_status_color = (0, 165, 255) # Orange (Warning)
            # Reset other timers
            last_look_away_time = None
            last_no_face_time = None
            
        else: # Exactly one face detected by Dlib
            last_no_face_time = None
            last_multiple_faces_time = None

            face = faces[0]
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) # Green box for face

            landmarks = predictor(gray, face)

            try:
                pitch, yaw, roll, rot_vec, trans_vec, cam_mat, dist_coeffs = get_head_pose(landmarks, frame_size)
                current_gaze_direction, is_gaze_threat = get_gaze_direction(pitch, yaw)

                # Draw gaze line (from nose tip)
                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rot_vec, trans_vec, cam_mat, dist_coeffs)
                p1 = ( int(landmarks.part(30).x), int(landmarks.part(30).y))
                p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                cv2.line(frame, p1, p2, (255,0,0), 2) # Blue line for gaze direction

                gaze_status_text = f"Gaze: {current_gaze_direction} (P: {pitch[0]:.1f}, Y: {yaw[0]:.1f})"
                gaze_status_color = (0, 0, 255) if is_gaze_threat else (255, 255, 0) # Red if threat, else yellow

                if is_gaze_threat:
                    if last_look_away_time is None:
                        last_look_away_time = current_time
                    if (current_time - last_look_away_time) > LOOK_AWAY_THRESHOLD_SEC:
                        current_dlib_status_text = f"Gaze: {current_gaze_direction.replace('(Allowed)', '').strip()}"
                        current_dlib_status_color = (0, 0, 255) # Red (Cheating)
                    else:
                        current_dlib_status_text = f"Warning: Gaze {current_gaze_direction.replace('(Allowed)', '').strip()} (Briefly)"
                        current_dlib_status_color = (0, 165, 255) # Orange (Warning)
                else:
                    last_look_away_time = None # Reset timer if gaze is forward
                    if current_dlib_status_text == "": # Only if no other Dlib issue is present
                        current_dlib_status_text = "Gaze: Forward (OK)"
                        current_dlib_status_color = (0, 255, 0) # Green (OK)
            except Exception as e:
                gaze_status_text = "Gaze: Error"
                gaze_status_color = (0, 165, 255) # Orange (Warning)

        # --- YOLO Object and Person Detection ---
        phone_detected = False
        num_persons_yolo = 0
        yolo_status_text = "YOLO: No Objects"
        yolo_status_color = (255, 255, 0) # Default: Yellow (Info)

        if yolo_net is not None:
            yolo_boxes, yolo_confs, yolo_class_ids = detect_objects(
                frame, yolo_net, yolo_output_layers, 
                confidence_threshold=0.5,
                nms_threshold=0.4
            )
            
            for i, (box, conf, class_id) in enumerate(zip(yolo_boxes, yolo_confs, yolo_class_ids)):
                x, y, w, h = box
                label = str(yolo_classes[class_id])
                color = yolo_colors[class_id % len(yolo_colors)]
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if label == "cell phone" and conf > PHONE_DETECTION_CONFIDENCE: 
                    phone_detected = True
                    yolo_status_text = "YOLO: PHONE DETECTED!"
                    yolo_status_color = (0, 0, 255) # Red (Cheating)
                elif label == "person":
                    num_persons_yolo += 1
            
            # If multiple persons detected by YOLO (and no phone)
            if num_persons_yolo > 1 and not phone_detected: 
                yolo_status_text = f"YOLO: {num_persons_yolo} Persons Detected!"
                yolo_status_color = (0, 165, 255) # Orange (Warning)
        else:
            yolo_status_text = "YOLO: Model not loaded!"
            yolo_status_color = (0, 165, 255) # Orange (Warning)


        # --- Strike Management Logic ---
        # Determine if a cheating incident is severe enough to warrant a strike
        is_severe_cheating_detected_this_frame = False
        current_cheating_reason_type = "" # Used for cooldown logging

        if tab_cheating_detected_os:
            is_severe_cheating_detected_this_frame = True
            current_cheating_reason_type = "App Switch"
        elif phone_detected:
            is_severe_cheating_detected_this_frame = True
            current_cheating_reason_type = "Phone Use"
        elif current_dlib_status_color == (0, 0, 255): # Any red status from Dlib indicates severe cheating
            is_severe_cheating_detected_this_frame = True
            if "No Face" in current_dlib_status_text:
                current_cheating_reason_type = "No Face"
            elif "Multiple Faces" in current_dlib_status_text:
                current_cheating_reason_type = "Multiple Faces"
            elif "Gaze" in current_dlib_status_text:
                current_cheating_reason_type = "Gaze Away"
        
        # Logic to increment strike_count only if a new or sufficiently prolonged cheating event occurs
        if is_severe_cheating_detected_this_frame:
            # Check for cooldown or a change in cheating reason type
            if (current_time - _last_strike_timestamp > _strike_cooldown_duration) or \
               (current_cheating_reason_type and current_cheating_reason_type != _last_cheating_reason_for_strike):
                
                if strike_count < MAX_STRIKES + 1: # Allow incrementing up to MAX_STRIKES + 1 (for final block)
                    strike_count += 1
                    _last_strike_timestamp = current_time
                    _last_cheating_reason_for_strike = current_cheating_reason_type
                    log_incident(f"Strike {strike_count}: {current_cheating_reason_type}", details=current_overall_status)
        else:
            # If no severe cheating, reset the last logged reason to allow new types of strikes
            _last_cheating_reason_for_strike = "" # Clear reason
            # Optionally, gradually reduce strike count if 'all clear' for a long time
            # if current_time - _last_strike_timestamp > 60 and strike_count > 0:
            #     strike_count = max(0, strike_count - 1) # Reduce by 1 every 60s if all clear

        # --- Combine Overall Status (for display and frontend communication) ---
        final_status_text = ""
        final_status_color = (0, 255, 0) # Default: Green (All Clear)
        alert_to_play = "none"

        # Determine overall status based on strike count and current detections
        if strike_count > MAX_STRIKES: # Test is permanently blocked
            final_status_text = "TEST BLOCKED: MAXIMUM STRIKES REACHED!"
            final_status_color = (0, 0, 255) # Red
            alert_to_play = "severe"
        elif is_severe_cheating_detected_this_frame: # Severe cheating detected, but not yet blocked
            # This covers App Switch, Phone, No Face, Multiple Faces, Gaze Away (when red)
            final_status_text = f"CHEATING DETECTED: {current_cheating_reason_type}!"
            final_status_color = (0, 0, 255) # Red
            alert_to_play = "severe"
        elif current_dlib_status_color == (0, 165, 255) or num_persons_yolo > 1:
            # This covers Dlib warning states (brief no face/multiple faces/gaze away)
            # or multiple persons by YOLO (warning, not cheating yet)
            final_status_text = f"WARNING: {current_dlib_status_text.replace('(Briefly)', '').strip() or yolo_status_text.replace('YOLO: ', '')}!"
            final_status_color = (0, 165, 255) # Orange
            alert_to_play = "light"
        else:
            final_status_text = "STATUS: ALL CLEAR!"
            final_status_color = (0, 255, 0) # Green
            alert_to_play = "none"

        # Update global status variables for the jsonify endpoint
        with lock:
            current_overall_status = final_status_text
            current_overall_color = [c/255.0 for c in final_status_color] # Normalize to 0-1 for JS
            current_alert_sound = alert_to_play

        # Display all status messages on the frame itself (for visual feedback)
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        FONT_SCALE = 0.7
        FONT_THICKNESS = 2

        cv2.putText(frame, final_status_text, (10, 30), FONT, FONT_SCALE, final_status_color, FONT_THICKNESS)
        cv2.putText(frame, yolo_status_text, (10, 60), FONT, FONT_SCALE, yolo_status_color, FONT_THICKNESS)
        cv2.putText(frame, f"Faces (Dlib): {num_faces_dlib}", (10, 90), FONT, FONT_SCALE, (255, 255, 0), FONT_THICKNESS)
        cv2.putText(frame, gaze_status_text, (10, 120), FONT, FONT_SCALE, gaze_status_color, FONT_THICKNESS)
        cv2.putText(frame, f"Strikes: {strike_count}/{MAX_STRIKES}", (10, 150), FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS)

        # Encode frame as JPEG for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            print("Failed to encode frame.")
            continue
        
        frame_bytes = buffer.tobytes()

        # Update global output_frame (if needed for other purposes, though not used by current Flask route directly)
        with lock:
            output_frame = frame_bytes

        # Yield the frame for HTTP streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def start_webcam():
    """
    Initializes the webcam capture object.
    """
    global cap
    if cap is None or not cap.isOpened(): # Check if webcam is already open
        print("Initializing webcam...")
        cap = cv2.VideoCapture(0) # 0 for default webcam
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            cap = None # Set to None to indicate failure
        else:
            print("Webcam initialized successfully!")

# Flask Routes
@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Supplies frames from the webcam."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status_feed')
def status_feed():
    """JSON endpoint to get current detection status, strike count, and alert sound."""
    global current_overall_status, current_overall_color, current_alert_sound, strike_count
    with lock: # Ensure thread-safe access to global variables
        return jsonify(
            status=current_overall_status,
            color=current_overall_color, # Color as [R, G, B] normalized
            alert_sound=current_alert_sound,
            strike_count=strike_count,
            max_strikes=MAX_STRIKES # Send max strikes to frontend
        )

# Startup sequence: Load YOLO model and initialize webcam when the app starts
if __name__ == '__main__':
    load_yolo() 
    start_webcam() 
    # Run the Flask app
    # threaded=True allows multiple requests (e.g., video feed and status feed) to be handled concurrently
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True) 
