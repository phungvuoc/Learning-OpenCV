import cv2
import sys
from helper_functions import *

def run_tracker(tracker_names, video_spec, video_output_file_name):

    # Create the video capture object.
    video_cap = cv2.VideoCapture(video_spec.video_filename)

    # Confirm video file can be opened.
    if video_cap.isOpened():
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    else:
        print("Could not open video")
        sys.exit()
    # Set up video writer object for mp4.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps_write = fps  # or other desired value
    resolution_specs = resolution_dict.get(video_spec.res)
    output_video_dim = get_output_video_dims(tracker_names, resolution_specs)
    video_out = cv2.VideoWriter(video_output_file_name, fourcc, fps_write, output_video_dim)

    # Read the first frame.
    ok, frame = video_cap.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    # Resize the image frame to the specified resolution.
    frame = cv2.resize(frame, resolution_specs, interpolation=cv2.INTER_AREA)

    # Get the list of tracker objects.
    tracker_objects = get_trackers(tracker_names, tracker_dict)

    # Initialize trackers.
    initialize_trackers(tracker_objects, frame, video_spec.bbox)

    # -----------------------
    # Process video frames.
    # -----------------------
    print('Processing frames, please wait ...')
    while True:

        ok, frame = video_cap.read()
        if not ok:
            break

        # Resize the frame to the specified resolution.
        frame = cv2.resize(frame, resolution_specs, interpolation=cv2.INTER_AREA)

        # Retrieve the results for each tracker.
        frames_list = get_tracker_results(tracker_objects, frame, tracker_names)

        # Compose the final results in a multi-view layout.
        result = align_frames(frames_list)

        video_out.write(result)

    video_cap.release()
    video_out.release()
    print('Processing completed.')


if __name__ == "__main__":

    input_video = './race_car.mp4'
    trackers = ['BOOSTING', 'CSRT']
    video_output_prefix = 'debug_1x2'
    video_output_file_name = 'tracking_analysis_output_videos/' + video_output_prefix + '.mp4'
    video_obj = VideoSpec(input_video, '480p', (370, 225, 180, 80))

    run_tracker(trackers, video_obj, video_output_file_name)
