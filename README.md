# cheat_detection
Project Explanation: Advanced Cheating Detection System
This project is a sophisticated web-based proctoring system designed to monitor students during online exams and detect potential cheating behaviors in real-time. It uses a combination of computer vision, object detection, and operating system-level monitoring. The system provides escalating warnings to the student before permanently blocking the test if cheating persists.

Core Technologies Used
Python Flask (Backend): A lightweight web framework for building the server-side application, handling video streams, running AI models, and managing detection logic.
OpenCV (Python Library): Used for general image processing, drawing annotations on video frames, and serving as the foundation for other vision tasks.
Dlib (Python Library): A machine learning library used specifically for real-time face detection and facial landmark estimation. This enables head pose estimation for gaze tracking.
YOLO (You Only Look Once) (via OpenCV DNN module): An advanced real-time object detection algorithm. Here, it's used to detect specific objects like "cell phone" and "person" in the video stream.
win32gui, win32process, psutil (Python Libraries - Windows-specific): Used for operating system-level monitoring, allowing the system to detect if the user switches out of the allowed application window or runs unpermitted processes.
HTML, CSS, JavaScript (Frontend): Standard web technologies for building the user interface, displaying the live video feed, status updates, and interactive warning/blocking modals.
Material Design (CSS): A design system developed by Google, applied to the frontend to ensure a clean, modern, responsive, and intuitive user experience with consistent aesthetics.
Backend (app.py) - The Brains of the Operation
The app.py Flask application is the core engine. It manages video input from the webcam, processes it with AI models, applies cheating detection rules, and serves the results to the web frontend.

1. Initialization and Global State
detector, predictor (Dlib): These are initialized once globally. detector finds faces, and predictor maps 68 key points (landmarks) on a detected face (eyes, nose, mouth outline).
YOLO Globals (yolo_net, yolo_classes, etc.): The YOLO model, its class names (coco.names.txt), and output layers are loaded once via the load_yolo() function when the application starts. This is crucial for performance, as loading these large files repeatedly would be inefficient. Error handling is included to check for missing model files.
Webcam (cap): The webcam is initialized once via start_webcam() globally.
Cheating Detection Timers (last_look_away_time, etc.): These None variables track the last time a specific violation (e.g., no face, looking away) was detected. They are used with _THRESHOLD_SEC constants to determine if a brief anomaly becomes a sustained, strike-worthy incident.
Overall Status (current_overall_status, current_overall_color, current_alert_sound): These global variables hold the current state of the proctoring. They are updated frequently and sent to the frontend.
Strike Logic Globals (strike_count, MAX_STRIKES, _last_strike_timestamp, _strike_cooldown_duration, _last_cheating_reason_for_strike):
strike_count: The most critical variable, tracking the number of "strikes" (warnings) the student has accumulated.
MAX_STRIKES: Set to 3, meaning the 1st, 2nd, and 3rd severe cheating incidents trigger a warning. The 4th incident triggers a permanent block.
_last_strike_timestamp, _strike_cooldown_duration, _last_cheating_reason_for_strike: These implement a "cooldown" mechanism. If a student is continuously violating a rule (e.g., keeps looking away), the system won't issue a new strike every single frame. A new strike is issued only if:
The _strike_cooldown_duration (e.g., 5 seconds) has passed since the last strike, OR
A different type of severe cheating starts (e.g., they stop looking away, but then immediately pick up a phone). This prevents an immediate cascade of strikes for a single, prolonged violation.
threading.Lock: This lock is used to prevent "race conditions" when multiple threads (like the video streaming thread and the status update thread) try to access or modify the global state variables (current_overall_status, strike_count, etc.) simultaneously.
2. Cheating Detection Modules
Dlib-based Face & Gaze Detection (get_head_pose, get_gaze_direction):
Face Detection: detector(gray) identifies rectangular regions containing faces.
Landmark Prediction: predictor(gray, face) locates 68 specific points on the face, crucial for head pose.
Head Pose Estimation (get_head_pose):
Uses a technique called Perspective-n-Point (PnP). It takes known 3D model points of a generic face and their corresponding 2D projections (the Dlib landmarks on the captured frame) to calculate the 3D orientation (rotation) and position (translation) of the head relative to the camera.
It then extracts Euler angles (pitch, yaw, roll).
Pitch: Up/down head tilt.
Yaw: Left/right head turn (most relevant for "looking away").
Roll: Tilting head side to side (less critical for gaze).
Gaze Direction (get_gaze_direction): Based on the YAW_THRESHOLD and PITCH_THRESHOLD, it determines if the student is looking "Forward," "Left," "Right," "Up," or "Down." Looking Left or Right is considered a "gaze threat."
YOLO-based Object Detection (detect_objects):
Model Loading (load_yolo): Loads pre-trained YOLO weights and configuration. yolov4 is a general-purpose object detection model trained on the COCO dataset, which includes objects like "cell phone" and "person."
Detection Process:
The frame is converted into a blob (a format suitable for neural networks).
The blob is fed into the yolo_net for a "forward pass."
The network outputs are processed to find bounding boxes, confidence scores, and class IDs for detected objects.
Non-Maximum Suppression (NMS) is applied to remove redundant overlapping bounding boxes, ensuring each object is detected only once.
Specific Detections: The system specifically checks for:
"cell phone": If detected with a confidence above PHONE_DETECTION_CONFIDENCE, it's a severe cheating incident.
"person": If more than one "person" is detected (num_persons_yolo > 1), it's a warning, indicating a potential helper or unauthorized presence.
OS-Level Monitoring (get_active_window_info):
This is a Windows-specific feature using win32gui, win32process, and psutil.
It checks the GetForegroundWindow() (the active window) to get its title and the associated process name.
It compares these against ALLOWED_WINDOW_TITLES and ALLOWED_PROCESS_NAMES.
If the user switches to an disallowed application or window for longer than APP_SWITCH_THRESHOLD_SEC, it's flagged as a severe cheating incident.
3. The generate_frames() Loop
This is the heart of the real-time processing:

Frame Capture: It continuously reads frames from the webcam (cap.read()).
Detection Execution: For each frame, it runs:
OS-level monitoring.
Dlib face and gaze detection.
YOLO object detection.
Strike Management: This is where the core logic for warnings and blocking resides:
is_severe_cheating_detected_this_frame: A boolean flag that becomes True if any of the critical cheating conditions (app switch, phone, no face, multiple faces, sustained gaze away) are met.
The strike_count is incremented only if is_severe_cheating_detected_this_frame is True AND the cooldown period has passed OR the type of cheating has changed. This is key for the 3-strike warning system.
log_incident() is called whenever a new strike is issued.
Overall Status Update: Based on strike_count and current detection states, current_overall_status, current_overall_color, and current_alert_sound are updated.
If strike_count > MAX_STRIKES, the status becomes "TEST BLOCKED."
If severe cheating is detected but strike_count is still within MAX_STRIKES, it's "CHEATING DETECTED."
If minor warnings (brief anomalies, multiple persons by YOLO) are present, it's "WARNING."
Otherwise, it's "ALL CLEAR."
Frame Annotation: Detection results (bounding boxes, labels, status text, gaze lines, strike count) are drawn directly onto the frame using OpenCV functions.
Encoding and Yielding: The frame is encoded into JPEG bytes (cv2.imencode) and yielded. Flask's /video_feed endpoint then streams these bytes to the browser as a multipart/x-mixed-replace response, effectively creating a live video feed.
4. Flask Endpoints
/: Renders index.html, which is the main frontend page.
/video_feed: This route is responsible for streaming the webcam feed. It calls generate_frames(), which is a generator function. Flask automatically streams the JPEG segments yielded by generate_frames() to the client.
/status_feed: This is a JSON API endpoint. The frontend periodically polls this endpoint to get the latest current_overall_status (text), current_overall_color (RGB values), current_alert_sound (which sound to play), strike_count, and MAX_STRIKES. This allows the frontend to update its UI without reloading the page or relying solely on the video feed.
Frontend (index.html and style.css) - The User Experience
The frontend is built with standard web technologies, but with a strong emphasis on Google's Material Design guidelines for a modern and intuitive look.

1. HTML Structure (index.html)
Semantic Layout: Uses <header>, <main>, <section>, <aside>, <footer> for clear structure.
container: A central div that holds the main content, designed to center content and manage responsiveness.
exam-section: Contains the mock exam questions. It can be visually disabled if the test is blocked.
detection-section: Houses the live proctoring elements (webcam feed, status display, proctoring tips, strike counter).
Video Feed (<img id="videoFeed">): An <img> tag whose src points to the /video_feed endpoint, allowing the browser to display the MJPEG stream from the backend.
Status Display (<div id="statusDisplay">): Shows the current status text from the backend.
Strike Counter (<span id="strikeCount">, <span id="maxStrikes">): Displays the current strike count out of the maximum allowed.
Audio Elements: <audio> tags for lightAlertSound and severeAlertSound to play auditory warnings.
Modals (testBlockedModal, warningModal):
testBlockedModal: The critical modal shown when the test is permanently blocked.
warningModal: The new modal shown for each warning (strike 1, 2, and 3).
Both use modal-overlay and modal-content for consistent styling and behavior.
Snackbar (snackbar): A small, transient message bar for brief notifications that don't interrupt the user flow significantly.
2. Styling (style.css) - Material Design Implementation
Color Palette (:root variables): Defines Material Design inspired primary, secondary, error, warning, success, and surface colors using CSS variables. This ensures consistent color usage throughout the UI.
Typography: Specifies the Roboto font with appropriate weights and font sizes for different headings (h1 to h3) and body text, following Material Design's typographic scale.
Layout and Spacing:
Uses flexbox for main layout (.container) to enable responsiveness.
Employs consistent spacing values (multiples of 8px) for padding, margin, and gap to adhere to the 8dp grid system.
Elevation (Shadows):
box-shadow is extensively used on elements like header, .exam-section, .detection-section, .question, and especially modal-content.
Different shadow values (dp 2, dp 4, dp 16) simulate varying levels of elevation and indicate hierarchy.
Transitions on box-shadow create subtle, engaging hover effects.
Responsive Design:
flex-wrap on .container allows sections to stack vertically on smaller screens.
min-width on sections ensures they don't become too narrow.
@media (max-width: 768px) queries adjust layout and font sizes for mobile devices, ensuring the app looks good and is usable on all screen sizes.
Interactive Elements:
Buttons: Styled with md-button classes, including hover and active states that mimic Material Design's ripple and lift effects.
Radio Buttons: accent-color is used for a Material-like appearance.
Status Box: Dynamically changes background-color and color based on the status received from the backend (green, orange, red, grey, yellow).
Modals (modal-overlay, modal-content):
position: fixed, background: rgba(...) for the overlay effect.
opacity and visibility with transition for smooth fade-in/fade-out.
transform: scale(...) with a cubic-bezier transition for the characteristic Material Design "grow" animation when modals appear.
Specific styles for warning (orange heading) and blocked (red heading) modals.
Accessibility:
Uses semantic HTML (<header>, <main>, role="status", aria-live="polite").
pointer-events: none and opacity are used for exam-section.disabled to visually and functionally disable the test.
cursor: not-allowed for disabled radio buttons.
Webcam Blocked Overlay: The video-container.blocked::before pseudo-element provides a clear, centered overlay message when the webcam is unavailable or blocked.
3. JavaScript (index.html - <script>)
DOM Element References: All necessary HTML elements are referenced using document.getElementById().
Utility Functions (playSound, showSnackbar, showWarningModal, hideWarningModal, showTestBlockedModal, hideTestBlockedModal): Encapsulate common UI behaviors.
updateStatus() - The Core Frontend Logic:
Polling: It uses fetch('/status_feed') to send an AJAX request to the backend every 1000 milliseconds (1 second) via setInterval.
Data Processing: When data is received:
Updates the statusDisplay text and applies the correct CSS class for coloring (e.g., status-green, status-red).
Updates the strikeCountSpan and maxStrikesSpan.
Plays alert sounds based on data.alert_sound ("light" or "severe"), with a lastPlayedSound check to prevent continuous sound playback.
Webcam Status Handling: If data.status indicates "WEBCAM NOT FOUND" or "WEBCAM BLOCKED," it adds the blocked class to videoContainer and stops the videoFeed.src to visually show the camera is inactive. It restarts the stream if the status becomes clear again.
Crucial Modal Logic:
Permanent Block: If data.strike_count exceeds data.max_strikes (i.e., it's the 4th incident), showTestBlockedModal() is called. isTestBlocked flag prevents repeated blocking actions.
Warning: If data.strike_count has just increased (i.e., data.strike_count > lastReportedStrikeCount) and is still within data.max_strikes (1, 2, or 3), showWarningModal() is triggered. isWarningModalOpen prevents stacking modals.
Clear State: If the status becomes "ALL CLEAR" and the test is not already blocked, hideWarningModal() is called to dismiss any active warning pop-up.
lastReportedStrikeCount is updated to track the state for the next polling cycle.
Error Handling: Includes .catch() for fetch calls to gracefully handle network or server errors.
Event Listeners: Attached to modal buttons (closeModalBtn, closeWarningModalBtn) to manage their visibility.
How They Work Together (The Flow)
User Opens Page: The browser loads index.html.
Frontend Initialization: JavaScript immediately starts two key background processes:
The <img> tag attempts to load the video stream from /video_feed.
The updateStatus() function starts polling /status_feed every second.
Backend Processing:
The Flask app.py starts, loads YOLO models, and initializes the webcam.
The generate_frames() generator begins its infinite loop, capturing video, running Dlib and YOLO detections, performing OS checks, and updating the global status variables (current_overall_status, strike_count, etc.) based on its findings.
It continuously encodes and streams frames to /video_feed.
It continuously updates the global state variables.
Frontend Updates:
The browser displays the live video stream from /video_feed.
Each second, updateStatus() fetches the latest status JSON from /status_feed.
Based on the status, color, alert_sound, and crucially, the strike_count from the backend, the frontend dynamically:
Changes the color and text of the statusDisplay.
Plays a "light" or "severe" warning sound.
Updates the Strikes: X/3 counter.
Triggers Modals:
If strike_count increments to 1, 2, or 3, the warningModal pops up, displaying the current strike number.
If strike_count goes to 4 (or higher), the testBlockedModal pops up, permanently disabling the exam elements and stopping the video feed.
User Interaction: The student must acknowledge the warning/blocked modals by clicking a button to continue or understand the block.
Setup and Running Recap
To run this:

Python 3.x and pip are required.
Install Libraries: pip install Flask opencv-python dlib numpy psutil pywin32 (Note: pywin32 is for Windows; OS monitoring won't work on Linux/macOS without alternative libraries).
Download Models: Obtain shape_predictor_68_face_landmarks.dat, yolov4.weights, yolov4.cfg, and coco.names.txt. Place these in the same directory as app.py.
Create Directory Structure:
your_project_folder/
├── app.py
├── shape_predictor_68_face_landmarks.dat
├── yolov4.weights
├── yolov4.cfg
├── coco.names.txt
├── templates/
│   └── index.html
└── static/
    ├── style.css
    ├── warning_light.mp3
    └── warning_severe.mp3
Run: Navigate to your_project_folder in your terminal and execute python app.py.
Access: Open your browser to http://127.0.0.1:5000/.
Future Improvements and Considerations
Persistent Strike Count: For a real-world application, strike_count should be stored in a database for each student's test session, not just in memory. This ensures that if the server restarts or the student refreshes the page, their strike count is preserved.
Backend Validation: While frontend logic is good, the backend should also independently track and validate strike counts and test block status to prevent client-side manipulation.
Authentication and Test Sessions: Integrate proper user authentication and robust test session management (e.g., student ID, test ID).
More Robust Cheating Detection: Explore advanced techniques like:
Head pose stability analysis.
Eye blink rate for attentiveness.
Audio monitoring for speech or external noises.
Keyboard and mouse activity patterns.
Browser lockdown features (if building a dedicated exam browser).
Error Reporting: More sophisticated error logging and reporting for debugging.
Scalability: For many concurrent users, consider a more scalable Flask deployment (e.g., Gunicorn + Nginx) and potentially distributed AI processing.
This detailed breakdown should give you a solid understanding of how every piece of your project works together to achieve the real-time cheating detection and warning system!
