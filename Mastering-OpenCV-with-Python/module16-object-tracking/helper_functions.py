import cv2
import numpy as np

tracker_dict = dict(
    BOOSTING=cv2.legacy.TrackerBoosting_create(),
    CSRT=cv2.legacy.TrackerCSRT_create(),
    KCF=cv2.legacy.TrackerKCF_create(),
    MEDIANFLOW=cv2.legacy.TrackerMedianFlow_create(),
    MIL=cv2.legacy.TrackerMIL_create(),
    MOSSE=cv2.legacy.TrackerMOSSE_create(),
    TLD=cv2.legacy.TrackerTLD_create(),
)

resolution_dict = {
    '360p': (480, 360),
    '480p': (858, 480),
    '720p': (1280, 720),
    '1080p': (1920, 1080)
}


class VideoSpec:
    # Constructor.
    def __init__(self, video_filename, resolution, bbox):
        self.video_filename = video_filename
        self.res = resolution
        self.bbox = bbox


def draw_bounding_box(frame, bbox, ok, color=(0, 255, 255), thickness=2):
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, color, thickness)
    else:
        cv2.putText(frame, "Tracking failure detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


def draw_banner_text(frame, text, banner_height_percent=0.08, font_scale=1.5, font_thickness=2,
                     text_alignment="center", text_color=(0, 255, 0)):
    # Draw a black filled banner across the top of the image frame.
    # percent: banner height as a percentage of the frame height.
    banner_height = int(banner_height_percent * frame.shape[0])
    cv2.rectangle(frame, (0, 0), (frame.shape[1], banner_height), (0, 0, 0), thickness=-1)

    # Draw text on banner
    width = frame.shape[0]
    alignment_dict = dict(left=width // 4, center=width // 2, right=width * 3 // 4)
    left_offset = alignment_dict[text_alignment]
    location = (left_offset, banner_height - 10)
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_PLAIN, font_scale, text_color,
                font_thickness, cv2.LINE_AA)


def draw_text(frame, text, location=(20, 20), font_scale=1, color=(50, 170, 50), font_thickness=2):
    cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                font_thickness, cv2.LINE_AA)


def get_trackers(tracker_names, tracker_dict):
    tracker_objects = []
    for tracker in tracker_names:
        tracker_objects.append(tracker_dict[tracker])

    return tracker_objects


def initialize_trackers(tracker_objects, frame, bbox):
    for tracker in tracker_objects:
        tracker.init(frame, bbox)


def get_tracker_results(tracker_objects, frame, tracker_names):
    n = len(tracker_objects)
    init_frames_list = [frame.copy() for i in range(n)]
    final_frames_list = []

    for i in range(n):
        ok, result = update_tracker(tracker_objects[i], init_frames_list[i], tracker_names[i])
        final_frames_list.append(result)

    return final_frames_list


def update_tracker(tracker, frame, tracker_type):
    timer = cv2.getTickCount()

    # Update tracker.
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS).
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

    # Draw bounding box.
    draw_bounding_box(frame, bbox, ok)

    # Display tracker type on frame.
    draw_banner_text(frame, tracker_type + ' Tracker' + ', FPS : ' + str(int(fps)))

    return ok, frame


def get_output_video_dims(tracker_names, resolution_specs):
    width, height = resolution_specs
    n = len(tracker_names)
    if n == 1:
        return width, height
    if n == 2:
        return width * 2, height
    if n == 4:
        return width * 2, height * 2
    if n == 6:
        return width * 3, height * 2
    if n == 8:
        return width * 4, height * 2


def align_frames(frames_list):
    n = len(frames_list)

    if n == 1:
        return frames_list[0]

    if n == 2:
        return np.hstack([frames_list[0], frames_list[1]])

    if n == 4:
        top = np.hstack([frames_list[0], frames_list[1]])
        bottom = np.hstack([frames_list[2], frames_list[3]])
        return np.vstack([top, bottom])

    if n == 6:
        top = np.hstack([frames_list[0], frames_list[1], frames_list[2]])
        bottom = np.hstack([frames_list[3], frames_list[4], frames_list[5]])
        return np.vstack([top, bottom])

    if n == 8:
        top = np.hstack([frames_list[0], frames_list[1], frames_list[2], frames_list[3]])
        bottom = np.hstack([frames_list[4], frames_list[5], frames_list[6], frames_list[7]])
        return np.vstack([top, bottom])