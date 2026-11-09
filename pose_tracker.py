import cv2
import mediapipe as mp
import math
import time
import numpy as np
import csv
import os
import re
from scipy.io import savemat
from collections import deque


# ============================================================
# SETTINGS
# ============================================================

CAMERA_INDEX = 0

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Smaller sway box
BOX_WIDTH_PIXELS = 100
BOX_HEIGHT_PIXELS = 100

# Trial timing
CALIBRATION_DURATION_SECONDS = 8
COUNTDOWN_DURATION_SECONDS = 3
RECORDING_DURATION_SECONDS = 60

# Activity thresholds
LEG_STANDING_MAX_SPEED = 65
LEG_RUNNING_MIN_SPEED = 220

# Displacement legend
LEGEND_VISIBLE_EVENTS = 4
LEGEND_SCROLL_DELAY_SECONDS = 0.0

MIN_VISIBILITY = 0.50

# Data collection
DATA_FOLDER = "movement_data"


def safe_filename(text):
    """Convert a participant name into a safe filename."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())
    return cleaned.strip("_") or "unknown_participant"


def next_trial_number(participant_name):
    """Return the next unused trial number for a participant."""
    os.makedirs(DATA_FOLDER, exist_ok=True)

    participant_file = safe_filename(participant_name)
    pattern = re.compile(
        r"^" + re.escape(participant_file) + r"_trial_(\d+)\.mat$",
        re.IGNORECASE
    )

    existing_numbers = []

    for filename in os.listdir(DATA_FOLDER):
        match = pattern.match(filename)
        if match:
            existing_numbers.append(int(match.group(1)))

    if not existing_numbers:
        return 1

    return max(existing_numbers) + 1


# ============================================================
# BODY LANDMARKS
# ============================================================

# Arms, hands, and head are excluded.
TRACKED_LANDMARKS = {
    11, 12,        # shoulders
    23, 24,        # hips
    25, 26,        # knees
    27, 28,        # ankles
    29, 30, 31, 32 # feet
}

TRACKED_CONNECTIONS = {
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (27, 31),
    (28, 32)
}

LEG_SPEED_LANDMARKS = [25, 26, 27, 28]


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_point(landmarks, index, width, height):
    landmark = landmarks[index]

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    return x, y, landmark.visibility


def visible(landmarks, index):
    return landmarks[index].visibility >= MIN_VISIBILITY


def distance(point_a, point_b):
    return math.sqrt(
        (point_b[0] - point_a[0]) ** 2 +
        (point_b[1] - point_a[1]) ** 2
    )


def midpoint(point_a, point_b):
    return (
        int((point_a[0] + point_b[0]) / 2),
        int((point_a[1] + point_b[1]) / 2)
    )


def calculate_pelvic_tilt(left_hip, right_hip):
    """
    Level hips = 0 degrees.
    """

    dx = right_hip[0] - left_hip[0]
    dy = right_hip[1] - left_hip[1]

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None

    angle = math.degrees(math.atan2(-dy, dx))

    # Convert 180 degrees to 0 degrees
    angle = ((angle + 90) % 180) - 90

    if abs(angle) < 0.5:
        angle = 0.0

    return angle


def determine_activity(leg_speed):
    if leg_speed <= LEG_STANDING_MAX_SPEED:
        return "STANDING"

    elif leg_speed < LEG_RUNNING_MIN_SPEED:
        return "WALKING"

    else:
        return "RUNNING"


def activity_color(activity):
    if activity == "STANDING":
        return (255, 150, 0)

    elif activity == "WALKING":
        return (0, 255, 0)

    elif activity == "RUNNING":
        return (0, 165, 255)

    else:
        return (150, 150, 150)


def sidebar_text(sidebar, text, y, color=(255, 255, 255), size=0.43):
    cv2.putText(
        sidebar,
        text,
        (15, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        1,
        cv2.LINE_AA
    )


def export_session_data(history, events, participant_name, trial_number):
    """Save one labeled trial as CSV and MATLAB MAT data."""
    if not history:
        print("No movement data was collected; nothing was exported.")
        return

    os.makedirs(DATA_FOLDER, exist_ok=True)

    participant_file = safe_filename(participant_name)
    base_name = "%s_trial_%03d" % (participant_file, trial_number)
    csv_path = os.path.join(DATA_FOLDER, base_name + ".csv")
    mat_path = os.path.join(DATA_FOLDER, base_name + ".mat")

    # The first seven values are already used by the playback display.
    # The additional values are features for machine learning.
    headers = [
        "time_s", "waist_x_px", "waist_y_px", "inside_box",
        "sway_displacement_px", "pelvic_tilt_deg", "activity",
        "leg_speed_px_s", "thigh_left_px", "thigh_right_px",
        "shin_left_px", "shin_right_px"
    ]

    numeric_rows = []
    activity_labels = []

    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["participant_name", "trial_number"] + headers)

        for record in history:
            # Older records contain only the original seven fields.
            extra = list(record[7:]) + [np.nan] * 5
            row = list(record[:7]) + extra[:5]
            writer.writerow([participant_name, trial_number] + row)

            numeric_rows.append([
                float(row[0]), float(row[1]), float(row[2]),
                float(row[3]), float(row[4]), float(row[5]),
                float(row[7]), float(row[8]), float(row[9]),
                float(row[10]), float(row[11])
            ])
            activity_labels.append(str(row[6]))

    event_rows = []
    for event in events:
        event_rows.append([
            float(event["number"]), float(event["time"]),
            float(event["x"]), float(event["y"]),
            float(event["distance"])
        ])

    if not event_rows:
        event_rows = np.empty((0, 5), dtype=float)

    savemat(mat_path, {
        "participant_name": participant_name,
        "trial_number": trial_number,
        "features": np.asarray(numeric_rows, dtype=float),
        "feature_names": np.asarray([
            "time_s", "waist_x_px", "waist_y_px", "inside_box",
            "sway_displacement_px", "pelvic_tilt_deg",
            "leg_speed_px_s", "thigh_left_px", "thigh_right_px",
            "shin_left_px", "shin_right_px"
        ], dtype=object),
        "activity_labels": np.asarray(activity_labels, dtype=object),
        "displacement_events": event_rows
    })

    print("Exported trial data:")
    print("  CSV:", os.path.abspath(csv_path))
    print("  MAT:", os.path.abspath(mat_path))


# ============================================================
# PELVIC TRACKER
# ============================================================

def draw_pelvic_tracker(canvas, center, tilt_degrees):
    """
    Draws a hip-like line with a center circle.
    """

    center_x, center_y = center

    angle = math.radians(-tilt_degrees)
    half_length = 65

    def rotate_point(point):
        x, y = point

        rotated_x = (
            x * math.cos(angle) -
            y * math.sin(angle)
        )

        rotated_y = (
            x * math.sin(angle) +
            y * math.cos(angle)
        )

        return (
            int(center_x + rotated_x),
            int(center_y + rotated_y)
        )

    left_point = rotate_point((-half_length, 0))
    right_point = rotate_point((half_length, 0))

    # Hip line
    cv2.line(
        canvas,
        left_point,
        right_point,
        (255, 100, 255),
        7,
        cv2.LINE_AA
    )

    # Hip endpoints
    cv2.circle(
        canvas,
        left_point,
        9,
        (255, 180, 255),
        -1
    )

    cv2.circle(
        canvas,
        right_point,
        9,
        (255, 180, 255),
        -1
    )

    # Waist-center circle
    cv2.circle(
        canvas,
        (center_x, center_y),
        14,
        (255, 0, 255),
        -1
    )

    cv2.circle(
        canvas,
        (center_x, center_y),
        14,
        (255, 255, 255),
        2
    )


# ============================================================
# DISPLACEMENT MARKERS
# ============================================================

DISPLACEMENT_COLORS = [
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Yellow
    (255, 150, 0),   # Blue
    (0, 165, 255),   # Orange
    (255, 255, 0),   # Cyan
    (0, 255, 0)      # Green
]


def draw_displacement_events(
    frame,
    events,
    box_center,
    convert_point,
    current_playback_time
):
    """
    Displays clean markers for completed
    out-and-return sway events.
    """

    if box_center is None:
        return

    box_center_display = convert_point(
        box_center[0],
        box_center[1]
    )

    for event in events:

        if event["time"] > current_playback_time:
            continue

        color = DISPLACEMENT_COLORS[
            (event["number"] - 1) %
            len(DISPLACEMENT_COLORS)
        ]

        event_point = convert_point(
            event["x"],
            event["y"]
        )

        # Displacement measurement line
        cv2.line(
            frame,
            box_center_display,
            event_point,
            color,
            2
        )

        # Colored event marker
        cv2.circle(
            frame,
            event_point,
            12,
            color,
            -1
        )

        # White outline
        cv2.circle(
            frame,
            event_point,
            12,
            (255, 255, 255),
            2
        )

        # Event number
        cv2.putText(
            frame,
            "D%d" % event["number"],
            (event_point[0] - 10, event_point[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )


def draw_displacement_legend(
    frame,
    events,
    current_playback_time
):
    """
    Shows four displacement events at a time.
    The legend scrolls immediately as new events appear.
    """

    panel_left = 650
    panel_top = 300
    panel_right = 880
    panel_bottom = 535

    cv2.rectangle(
        frame,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (25, 25, 25),
        -1
    )

    cv2.rectangle(
        frame,
        (panel_left, panel_top),
        (panel_right, panel_bottom),
        (100, 100, 100),
        2
    )

    cv2.putText(
        frame,
        "SWAY EVENTS",
        (panel_left + 35, panel_top + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    visible_events = [
        event
        for event in events
        if event["time"] <= current_playback_time
    ]

    if len(visible_events) == 0:

        cv2.putText(
            frame,
            "No events yet",
            (panel_left + 45, panel_top + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )

        return

    maximum_start = max(
        0,
        len(visible_events) - LEGEND_VISIBLE_EVENTS
    )

    scroll_start = 0

    for possible_start in range(
        1,
        maximum_start + 1
    ):

        event_that_triggers_scroll = visible_events[
            possible_start + LEGEND_VISIBLE_EVENTS - 1
        ]

        if current_playback_time >= (
            event_that_triggers_scroll["time"] +
            LEGEND_SCROLL_DELAY_SECONDS
        ):

            scroll_start = possible_start

    events_to_show = visible_events[
        scroll_start:
        scroll_start + LEGEND_VISIBLE_EVENTS
    ]

    y = panel_top + 65

    for event in events_to_show:

        color = DISPLACEMENT_COLORS[
            (event["number"] - 1) %
            len(DISPLACEMENT_COLORS)
        ]

        cv2.circle(
            frame,
            (panel_left + 25, y - 5),
            9,
            color,
            -1
        )

        cv2.putText(
            frame,
            "D%d: %.1f px"
            % (
                event["number"],
                event["distance"]
            ),
            (panel_left + 45, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            event["activity"],
            (panel_left + 45, y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            activity_color(event["activity"]),
            1,
            cv2.LINE_AA
        )

        y += 42

    if len(visible_events) > LEGEND_VISIBLE_EVENTS:

        list_status = "%d-%d of %d" % (
            scroll_start + 1,
            min(
                scroll_start + LEGEND_VISIBLE_EVENTS,
                len(visible_events)
            ),
            len(visible_events)
        )

        cv2.putText(
            frame,
            list_status,
            (panel_left + 155, panel_bottom - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )


# ============================================================
# ACTIVITY TIMELINE
# ============================================================

def draw_activity_timeline(playback, history, playback_time):

    if len(history) < 2:
        return

    timeline_left = 30
    timeline_right = 530
    timeline_y = 565

    total_time = history[-1][0]

    if total_time <= 0:
        return

    cv2.putText(
        playback,
        "Activity history",
        (timeline_left, timeline_y - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    for i in range(1, len(history)):

        previous_record = history[i - 1]
        current_record = history[i]

        previous_time = previous_record[0]
        current_time = current_record[0]

        x1 = int(
            timeline_left +
            (previous_time / total_time) *
            (timeline_right - timeline_left)
        )

        x2 = int(
            timeline_left +
            (current_time / total_time) *
            (timeline_right - timeline_left)
        )

        color = activity_color(
            current_record[6]
        )

        cv2.line(
            playback,
            (x1, timeline_y),
            (x2, timeline_y),
            color,
            12
        )

    current_x = int(
        timeline_left +
        (playback_time / total_time) *
        (timeline_right - timeline_left)
    )

    cv2.line(
        playback,
        (current_x, timeline_y - 15),
        (current_x, timeline_y + 15),
        (255, 255, 255),
        2
    )

    cv2.putText(
        playback,
        "Blue = standing",
        (550, 550),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 150, 0),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        playback,
        "Green = walking",
        (550, 570),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        playback,
        "Orange = running",
        (550, 590),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 165, 255),
        1,
        cv2.LINE_AA
    )


# ============================================================
# PLAYBACK WINDOW
# ============================================================

def make_playback_frame(
    history,
    events,
    box_center,
    playback_index
):
    playback_width = 900
    playback_height = 650

    playback = np.zeros(
        (playback_height, playback_width, 3),
        dtype=np.uint8
    )

    if box_center is None:
        return playback

    # Reserve the left side for the sway box.
    # Reserve the right side for the information panels.
    playback_camera_right = 610
    playback_camera_bottom = 500
    margin = 45

    available_width = (
        playback_camera_right - 2 * margin
    )

    available_height = (
        playback_camera_bottom - 2 * margin
    )

    scale_x = available_width / CAMERA_WIDTH
    scale_y = available_height / CAMERA_HEIGHT

    scale = min(scale_x, scale_y)

    offset_x = margin

    offset_y = (
        playback_camera_bottom -
        CAMERA_HEIGHT * scale
    ) / 2

    def convert_point(point_x, point_y):
        return (
            int(offset_x + point_x * scale),
            int(offset_y + point_y * scale)
        )

    # Camera-frame border
    cv2.rectangle(
        playback,
        convert_point(0, 0),
        convert_point(CAMERA_WIDTH, CAMERA_HEIGHT),
        (100, 100, 100),
        2
    )

    # Sway box
    box_left = box_center[0] - BOX_WIDTH_PIXELS / 2
    box_right = box_center[0] + BOX_WIDTH_PIXELS / 2
    box_top = box_center[1] - BOX_HEIGHT_PIXELS / 2
    box_bottom = box_center[1] + BOX_HEIGHT_PIXELS / 2

    cv2.rectangle(
        playback,
        convert_point(box_left, box_top),
        convert_point(box_right, box_bottom),
        (255, 255, 255),
        3
    )

    # Box center
    cv2.circle(
        playback,
        convert_point(box_center[0], box_center[1]),
        6,
        (255, 255, 255),
        -1
    )

    visible_history = history[:playback_index + 1]

    current_playback_time = 0

    if len(visible_history) > 0:

        current_record = visible_history[-1]

        current_x = current_record[1]
        current_y = current_record[2]
        current_displacement = current_record[4]
        current_tilt = current_record[5]
        current_activity = current_record[6]
        current_playback_time = current_record[0]

        current_point = convert_point(
            current_x,
            current_y
        )

        # Purple waist dot
        cv2.circle(
            playback,
            current_point,
            10,
            (255, 0, 255),
            -1
        )

        cv2.circle(
            playback,
            current_point,
            10,
            (255, 255, 255),
            2
        )

        # Displacement markers
        draw_displacement_events(
            playback,
            events,
            box_center,
            convert_point,
            current_playback_time
        )

        # Pelvic tracker panel
        tracker_left = 650
        tracker_top = 35
        tracker_right = 880
        tracker_bottom = 270

        cv2.rectangle(
            playback,
            (tracker_left, tracker_top),
            (tracker_right, tracker_bottom),
            (25, 25, 25),
            -1
        )

        cv2.rectangle(
            playback,
            (tracker_left, tracker_top),
            (tracker_right, tracker_bottom),
            (100, 100, 100),
            2
        )

        cv2.putText(
            playback,
            "PELVIC TRACKER",
            (tracker_left + 28, tracker_top + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA
        )

        pelvic_center = (
            int((tracker_left + tracker_right) / 2),
            tracker_top + 105
        )

        draw_pelvic_tracker(
            playback,
            pelvic_center,
            current_tilt
        )

        cv2.putText(
            playback,
            "Tilt: %.1f deg" % current_tilt,
            (tracker_left + 48, tracker_top + 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            playback,
            "Activity: %s" % current_activity,
            (tracker_left + 25, tracker_top + 195),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            activity_color(current_activity),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            playback,
            "Displacement: %.1f px"
            % current_displacement,
            (tracker_left + 18, tracker_top + 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    # Scrolling legend
    draw_displacement_legend(
        playback,
        events,
        current_playback_time
    )

    # Activity timeline
    draw_activity_timeline(
        playback,
        history,
        current_playback_time
    )

    # Title
    cv2.putText(
        playback,
        "SWAY BOX PLAYBACK",
        (30, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

    # Timing and maximum displacement
    if len(history) > 0:

        total_time = history[-1][0]

        cv2.putText(
            playback,
            "Playback: %.1f / %.1f sec"
            % (current_playback_time, total_time),
            (30, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        maximum_displacement = max(
            record[4]
            for record in history
        )

        cv2.putText(
            playback,
            "Maximum displacement: %.1f px"
            % maximum_displacement,
            (300, 625),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    return playback


def close_displacement_event(
    events,
    event_active,
    event_time,
    event_x,
    event_y,
    event_distance,
    event_activity
):
    if event_active and event_x is not None:

        events.append(
            {
                "number": len(events) + 1,
                "time": event_time,
                "x": event_x,
                "y": event_y,
                "distance": event_distance,
                "activity": event_activity
            }
        )

    return (
        False,
        None,
        None,
        None,
        0.0,
        ""
    )


# ============================================================
# CAMERA SETUP
# ============================================================

camera = cv2.VideoCapture(CAMERA_INDEX)

if not camera.isOpened():
    print("Could not open the camera.")
    print("Try changing CAMERA_INDEX from 0 to 1.")
    raise SystemExit

camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

print("Movement data collection mode")
participant_name = input("Enter participant name for this trial: ").strip()
if not participant_name:
    participant_name = "Unknown"

trial_number = next_trial_number(participant_name)
data_exported = False


# ============================================================
# WINDOW SETUP
# ============================================================

WINDOW_NAME = "Windows Pose and Pelvic Motion Tracker"
PLAYBACK_WINDOW_NAME = "Sway Box Playback"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    WINDOW_NAME,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)


# ============================================================
# TRIAL VARIABLES
# ============================================================

trial_state = "CALIBRATING"

calibration_start = None
calibration_samples = []

box_center = None
box_ready = False

countdown_start = None
recording_start = None

# Each record contains:
# time, waist_x, waist_y, inside_box,
# displacement, pelvic_tilt, activity
session_history = []

# Completed out-and-return displacement events
displacement_events = []

# Current displacement event
event_active = False
event_time = None
event_x = None
event_y = None
event_distance = 0.0
event_activity = ""

playback_active = False
playback_index = 0

previous_leg_points = {}
previous_leg_time = None
leg_speed_history = deque(maxlen=8)

activity = "WAITING"
pelvic_tilt_value = None
leg_speed = 0
sway_displacement = 0
sway_status = "CALIBRATING"

fullscreen = True


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = camera.read()

    if not success:
        print("Could not read from the camera.")
        break

    frame = cv2.flip(frame, 1)

    frame_height, frame_width = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    current_time = time.time()

    pelvic_tilt_value = None
    leg_speed = 0
    sway_displacement = 0

    thigh_left = None
    thigh_right = None
    shin_left = None
    shin_right = None


    # Finish after one minute
    if (
        trial_state == "RECORDING" and
        recording_start is not None and
        current_time - recording_start >=
        RECORDING_DURATION_SECONDS
    ):

        trial_state = "FINISHED"

        (
            event_active,
            event_time,
            event_x,
            event_y,
            event_distance,
            event_activity
        ) = close_displacement_event(
            displacement_events,
            event_active,
            event_time,
            event_x,
            event_y,
            event_distance,
            event_activity
        )

        if not data_exported:
            export_session_data(
                session_history,
                displacement_events,
                participant_name,
                trial_number
            )
            data_exported = True
            trial_number += 1


    # ========================================================
    # POSE DETECTION
    # ========================================================

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark


        # ----------------------------------------------------
        # LEG MOVEMENT SPEED
        # ----------------------------------------------------

        current_leg_points = {}
        current_leg_speeds = []

        for index in LEG_SPEED_LANDMARKS:

            if visible(landmarks, index):

                point_data = get_point(
                    landmarks,
                    index,
                    frame_width,
                    frame_height
                )

                current_point = (
                    point_data[0],
                    point_data[1]
                )

                current_leg_points[index] = current_point

                if (
                    index in previous_leg_points and
                    previous_leg_time is not None
                ):

                    elapsed_time = (
                        current_time - previous_leg_time
                    )

                    if elapsed_time > 0:

                        movement = distance(
                            previous_leg_points[index],
                            current_point
                        )

                        point_speed = movement / elapsed_time
                        current_leg_speeds.append(point_speed)

        if len(current_leg_speeds) > 0:

            leg_speed = sum(
                current_leg_speeds
            ) / len(current_leg_speeds)

            leg_speed_history.append(leg_speed)

        previous_leg_points = current_leg_points
        previous_leg_time = current_time

        if len(leg_speed_history) > 0:

            average_leg_speed = sum(
                leg_speed_history
            ) / len(leg_speed_history)

            activity = determine_activity(
                average_leg_speed
            )

        else:
            activity = "DETECTING LEGS"


        # ----------------------------------------------------
        # DRAW TORSO, HIPS, LEGS, AND FEET
        # ----------------------------------------------------

        for connection in TRACKED_CONNECTIONS:

            start_index, end_index = connection

            if not visible(landmarks, start_index):
                continue

            if not visible(landmarks, end_index):
                continue

            start = get_point(
                landmarks,
                start_index,
                frame_width,
                frame_height
            )

            end = get_point(
                landmarks,
                end_index,
                frame_width,
                frame_height
            )

            cv2.line(
                frame,
                (start[0], start[1]),
                (end[0], end[1]),
                (0, 255, 0),
                3
            )

        for index in TRACKED_LANDMARKS:

            if not visible(landmarks, index):
                continue

            point = get_point(
                landmarks,
                index,
                frame_width,
                frame_height
            )

            cv2.circle(
                frame,
                (point[0], point[1]),
                5,
                (0, 255, 0),
                -1
            )


        # ----------------------------------------------------
        # WAIST AND PELVIC TRACKING
        # ----------------------------------------------------

        hips_visible = (
            visible(landmarks, 23) and
            visible(landmarks, 24)
        )

        if hips_visible:

            left_hip_data = get_point(
                landmarks,
                23,
                frame_width,
                frame_height
            )

            right_hip_data = get_point(
                landmarks,
                24,
                frame_width,
                frame_height
            )

            left_hip = (
                left_hip_data[0],
                left_hip_data[1]
            )

            right_hip = (
                right_hip_data[0],
                right_hip_data[1]
            )

            waist = midpoint(
                left_hip,
                right_hip
            )

            pelvic_tilt_value = calculate_pelvic_tilt(
                left_hip,
                right_hip
            )

            # Purple waist dot
            cv2.circle(
                frame,
                waist,
                10,
                (255, 0, 255),
                -1
            )

            # Cyan hip line
            cv2.line(
                frame,
                left_hip,
                right_hip,
                (255, 255, 0),
                3
            )


            # ------------------------------------------------
            # CALIBRATION
            # ------------------------------------------------

            if trial_state == "CALIBRATING":

                if calibration_start is None:
                    calibration_start = current_time

                calibration_elapsed = (
                    current_time - calibration_start
                )

                calibration_samples.append(waist)

                if calibration_elapsed >= CALIBRATION_DURATION_SECONDS:

                    average_x = sum(
                        point[0]
                        for point in calibration_samples
                    ) / len(calibration_samples)

                    average_y = sum(
                        point[1]
                        for point in calibration_samples
                    ) / len(calibration_samples)

                    box_center = (
                        int(average_x),
                        int(average_y)
                    )

                    box_ready = True
                    trial_state = "COUNTDOWN"
                    countdown_start = current_time


            # ------------------------------------------------
            # COUNTDOWN
            # ------------------------------------------------

            if trial_state == "COUNTDOWN":

                countdown_elapsed = (
                    current_time - countdown_start
                )

                if countdown_elapsed >= COUNTDOWN_DURATION_SECONDS:

                    trial_state = "RECORDING"
                    recording_start = current_time

                    session_history = []
                    displacement_events = []

                    event_active = False
                    event_time = None
                    event_x = None
                    event_y = None
                    event_distance = 0.0
                    event_activity = ""


            # ------------------------------------------------
            # SWAY BOX AND RECORDING
            # ------------------------------------------------

            if box_ready and box_center is not None:

                box_left = int(
                    box_center[0] - BOX_WIDTH_PIXELS / 2
                )

                box_right = int(
                    box_center[0] + BOX_WIDTH_PIXELS / 2
                )

                box_top = int(
                    box_center[1] - BOX_HEIGHT_PIXELS / 2
                )

                box_bottom = int(
                    box_center[1] + BOX_HEIGHT_PIXELS / 2
                )

                inside_box = (
                    box_left <= waist[0] <= box_right and
                    box_top <= waist[1] <= box_bottom
                )

                sway_displacement = distance(
                    waist,
                    box_center
                )

                if inside_box:

                    sway_status = "LOW SWAY"
                    box_color = (0, 255, 0)

                else:

                    sway_status = "HIGH SWAY"
                    box_color = (0, 0, 255)


                # Record the trial
                if trial_state == "RECORDING":

                    recording_elapsed = (
                        current_time - recording_start
                    )

                    tilt_for_history = (
                        pelvic_tilt_value
                        if pelvic_tilt_value is not None
                        else 0.0
                    )

                    session_history.append(
                        (
                            recording_elapsed,
                            waist[0],
                            waist[1],
                            inside_box,
                            sway_displacement,
                            tilt_for_history,
                            activity,
                            leg_speed,
                            thigh_left if thigh_left is not None else np.nan,
                            thigh_right if thigh_right is not None else np.nan,
                            shin_left if shin_left is not None else np.nan,
                            shin_right if shin_right is not None else np.nan
                        )
                    )


                    # Start displacement event
                    if not inside_box and not event_active:

                        event_active = True
                        event_time = recording_elapsed
                        event_x = waist[0]
                        event_y = waist[1]
                        event_distance = sway_displacement
                        event_activity = activity


                    # Track maximum displacement
                    elif not inside_box and event_active:

                        if sway_displacement > event_distance:

                            event_time = recording_elapsed
                            event_x = waist[0]
                            event_y = waist[1]
                            event_distance = sway_displacement
                            event_activity = activity


                    # Finish displacement event once back inside
                    elif inside_box and event_active:

                        (
                            event_active,
                            event_time,
                            event_x,
                            event_y,
                            event_distance,
                            event_activity
                        ) = close_displacement_event(
                            displacement_events,
                            event_active,
                            event_time,
                            event_x,
                            event_y,
                            event_distance,
                            event_activity
                        )


                # Draw live sway box
                cv2.rectangle(
                    frame,
                    (box_left, box_top),
                    (box_right, box_bottom),
                    box_color,
                    3
                )

                # Box center
                cv2.circle(
                    frame,
                    box_center,
                    6,
                    (255, 255, 255),
                    -1
                )

                # Purple waist dot
                cv2.circle(
                    frame,
                    waist,
                    10,
                    (255, 0, 255),
                    -1
                )

        else:

            sway_status = "WAIST NOT VISIBLE"


    else:

        activity = "NO PERSON DETECTED"
        sway_status = "NO POSE"

        previous_leg_points = {}
        previous_leg_time = None
        leg_speed_history.clear()


    # ========================================================
    # LEG SEGMENT LENGTHS
    # ========================================================

    if results.pose_landmarks:

        landmarks = results.pose_landmarks.landmark

        required_points = [23, 24, 25, 26, 27, 28]

        if all(
            visible(landmarks, index)
            for index in required_points
        ):

            left_hip = get_point(
                landmarks,
                23,
                frame_width,
                frame_height
            )

            right_hip = get_point(
                landmarks,
                24,
                frame_width,
                frame_height
            )

            left_knee = get_point(
                landmarks,
                25,
                frame_width,
                frame_height
            )

            right_knee = get_point(
                landmarks,
                26,
                frame_width,
                frame_height
            )

            left_ankle = get_point(
                landmarks,
                27,
                frame_width,
                frame_height
            )

            right_ankle = get_point(
                landmarks,
                28,
                frame_width,
                frame_height
            )

            thigh_left = distance(
                left_hip,
                left_knee
            )

            thigh_right = distance(
                right_hip,
                right_knee
            )

            shin_left = distance(
                left_knee,
                left_ankle
            )

            shin_right = distance(
                right_knee,
                right_ankle
            )


    # ========================================================
    # TRIAL MESSAGE
    # ========================================================

    trial_message = ""

    if trial_state == "CALIBRATING":

        if calibration_start is None:

            trial_message = "Stand still to calibrate"

        else:

            calibration_elapsed = (
                current_time - calibration_start
            )

            remaining = max(
                0,
                CALIBRATION_DURATION_SECONDS -
                calibration_elapsed
            )

            trial_message = (
                "Calibrating: %.1f sec"
                % remaining
            )


    elif trial_state == "COUNTDOWN":

        countdown_elapsed = (
            current_time - countdown_start
        )

        countdown_number = max(
            1,
            math.ceil(
                COUNTDOWN_DURATION_SECONDS -
                countdown_elapsed
            )
        )

        trial_message = (
            "STARTING: %d"
            % countdown_number
        )

        cv2.putText(
            frame,
            str(countdown_number),
            (frame_width // 2 - 35, frame_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            3,
            (0, 255, 255),
            5,
            cv2.LINE_AA
        )


    elif trial_state == "RECORDING":

        recording_elapsed = (
            current_time - recording_start
        )

        remaining = max(
            0,
            RECORDING_DURATION_SECONDS -
            recording_elapsed
        )

        minutes = int(remaining // 60)
        seconds = int(remaining % 60)

        trial_message = (
            "RECORDING: %02d:%02d"
            % (minutes, seconds)
        )

        cv2.putText(
            frame,
            "GO",
            (frame_width // 2 - 35, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )


    elif trial_state == "FINISHED":

        trial_message = "FINISHED"

        cv2.putText(
            frame,
            "FINISHED",
            (frame_width // 2 - 90, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 255, 255),
            3,
            cv2.LINE_AA
        )


    # ========================================================
    # SIDEBAR
    # ========================================================

    sidebar_width = 380

    sidebar = np.zeros(
        (frame_height, sidebar_width, 3),
        dtype=np.uint8
    )

    y = 24

    sidebar_text(
        sidebar,
        "POSE TRACKER",
        y,
        (0, 255, 255),
        0.62
    )

    y += 27

    sidebar_text(
        sidebar,
        "Trial: " + trial_state,
        y
    )

    y += 21

    sidebar_text(
        sidebar,
        "Activity: " + activity,
        y
    )

    y += 21

    sidebar_text(
        sidebar,
        "Leg speed: %.1f px/s" % leg_speed,
        y
    )

    y += 29

    sidebar_text(
        sidebar,
        "PELVIC MEASUREMENT",
        y,
        (0, 255, 255),
        0.45
    )

    y += 21

    if pelvic_tilt_value is not None:

        sidebar_text(
            sidebar,
            "Pelvic tilt: %.1f deg"
            % pelvic_tilt_value,
            y
        )

    else:

        sidebar_text(
            sidebar,
            "Pelvic tilt: unavailable",
            y
        )

    y += 29

    sidebar_text(
        sidebar,
        "SWAY MEASUREMENT",
        y,
        (0, 255, 255),
        0.45
    )

    y += 21

    sidebar_text(
        sidebar,
        "Status: " + sway_status,
        y
    )

    y += 21

    sidebar_text(
        sidebar,
        "Center distance: %.1f px"
        % sway_displacement,
        y
    )

    y += 21

    sidebar_text(
        sidebar,
        "Box: %d x %d px"
        % (
            BOX_WIDTH_PIXELS,
            BOX_HEIGHT_PIXELS
        ),
        y
    )

    y += 29

    sidebar_text(
        sidebar,
        "LEG SEGMENTS",
        y,
        (0, 255, 255),
        0.45
    )

    y += 21

    if thigh_left is not None:

        sidebar_text(
            sidebar,
            "Thigh L/R: %.1f / %.1f px"
            % (
                thigh_left,
                thigh_right
            ),
            y,
            size=0.40
        )

        y += 20

        sidebar_text(
            sidebar,
            "Shin L/R: %.1f / %.1f px"
            % (
                shin_left,
                shin_right
            ),
            y,
            size=0.40
        )

    else:

        sidebar_text(
            sidebar,
            "Leg measurements unavailable",
            y,
            size=0.40
        )

    y += 29

    sidebar_text(
        sidebar,
        trial_message,
        y,
        (0, 165, 255),
        0.42
    )

    y += 29

    sidebar_text(
        sidebar,
        "CONTROLS",
        y,
        (0, 255, 255),
        0.45
    )

    y += 20

    sidebar_text(
        sidebar,
        "R = new trial",
        y,
        size=0.40
    )

    y += 19

    sidebar_text(
        sidebar,
        "S = playback history",
        y,
        size=0.40
    )

    y += 19

    sidebar_text(
        sidebar,
        "F = fullscreen",
        y,
        size=0.40
    )

    y += 19

    sidebar_text(
        sidebar,
        "ESC = quit",
        y,
        size=0.40
    )


    # ========================================================
    # LIVE DISPLAY
    # ========================================================

    combined_display = np.hstack(
        (frame, sidebar)
    )

    cv2.imshow(
        WINDOW_NAME,
        combined_display
    )


    # ========================================================
    # PLAYBACK DISPLAY
    # ========================================================

    if playback_active and len(session_history) > 0:

        playback_frame = make_playback_frame(
            session_history,
            displacement_events,
            box_center,
            playback_index
        )

        cv2.imshow(
            PLAYBACK_WINDOW_NAME,
            playback_frame
        )

        if playback_index < len(session_history) - 1:
            playback_index += 1

    elif not playback_active:

        try:
            cv2.destroyWindow(
                PLAYBACK_WINDOW_NAME
            )
        except cv2.error:
            pass


    # ========================================================
    # KEYBOARD CONTROLS
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # Quit
    if key == 27:
        break


    # New trial
    elif key == ord("r"):

        trial_state = "CALIBRATING"

        calibration_start = None
        calibration_samples = []

        box_center = None
        box_ready = False

        countdown_start = None
        recording_start = None
        data_exported = False

        session_history = []
        displacement_events = []

        event_active = False
        event_time = None
        event_x = None
        event_y = None
        event_distance = 0.0
        event_activity = ""

        playback_active = False
        playback_index = 0

        previous_leg_points = {}
        previous_leg_time = None

        leg_speed_history.clear()

        try:
            cv2.destroyWindow(
                PLAYBACK_WINDOW_NAME
            )
        except cv2.error:
            pass


    # Playback history
    elif key == ord("s"):

        if (
            trial_state == "FINISHED" and
            len(session_history) > 0
        ):

            playback_active = not playback_active

            if playback_active:

                playback_index = 0

            else:

                try:
                    cv2.destroyWindow(
                        PLAYBACK_WINDOW_NAME
                    )
                except cv2.error:
                    pass


    # Fullscreen
    elif key == ord("f"):

        if fullscreen:

            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_NORMAL
            )

            fullscreen = False

        else:

            cv2.setWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN
            )

            fullscreen = True


   

# ============================================================
# CLEANUP
# ============================================================

camera.release()
pose.close()
cv2.destroyAllWindows()
