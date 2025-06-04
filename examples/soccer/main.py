import argparse
from enum import Enum
from typing import Iterator, List

import os
import cv2
import io
import json
import moviepy.editor as mp
import sounddevice as sd
import soundfile as sf
import numpy as np
import queue
import threading
import time
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from examples.soccer.stt_service import transcribe_audio_chunk
from examples.soccer.translation_service import translate_text
from examples.soccer.tts_service import synthesize_speech
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

# Global Audio Playback Queue and Thread
audio_playback_queue = queue.Queue()
playback_active = True # Flag to stop the thread gracefully

def audio_playback_worker():
    global playback_active
    print("Audio playback worker started.")
    while playback_active:
        try:
            # Wait for up to 1 second for an item
            item = audio_playback_queue.get(timeout=1)
            if item is None: # Sentinel to stop the worker
                print("Audio playback worker received sentinel. Stopping.")
                playback_active = False # Ensure loop terminates
                audio_playback_queue.task_done() # Signal task completion for sentinel
                break

            audio_data_bytes, lang_code, text_for_log = item

            # Guard against None audio_data_bytes if it somehow gets queued
            if audio_data_bytes is None:
                print(f"Skipping None audio data for {lang_code}")
                audio_playback_queue.task_done()
                continue

            print(f"Playing audio for {lang_code}: '{text_for_log[:30]}...'")
            data, samplerate = sf.read(io.BytesIO(audio_data_bytes))
            sd.play(data, samplerate=samplerate)
            sd.wait()  # Wait for this segment to finish
            print(f"Finished playing audio for {lang_code}: '{text_for_log[:30]}...'")
            audio_playback_queue.task_done()
        except queue.Empty:
            continue # No audio to play, loop back
        except Exception as e:
            print(f"Audio playback error: {e}")
            if not audio_playback_queue.empty():
                try:
                    audio_playback_queue.task_done() # Must be called if get() was successful
                except ValueError: # If task_done() called too many times
                    pass
    print("Audio playback worker finished.")

def process_audio_segment_thread(audio_data_bytes, sample_rate, segment_start_time, target_languages_list, stt_language_code: str):
    if not audio_data_bytes:
        return

    print(f"Thread for segment {segment_start_time:.2f}s: Starting STT for lang {stt_language_code}.")
    transcript = transcribe_audio_chunk(audio_data_bytes, sample_rate, stt_language_code)
    print(f"Thread for segment {segment_start_time:.2f}s: STT result: '{transcript[:30]}...'")

    texts_to_synthesize_and_queue = []
    if transcript:
        texts_to_synthesize_and_queue.append({'text': transcript, 'lang': stt_language_code, 'original_timestamp': segment_start_time})

        if target_languages_list:
            for lang_code in target_languages_list:
                print(f"Thread for segment {segment_start_time:.2f}s: Translating to {lang_code} from {stt_language_code}.")
                translated_text = translate_text(transcript, lang_code, stt_language_code)
                print(f"Thread for segment {segment_start_time:.2f}s: Translation to {lang_code} result: '{translated_text[:30]}...'")
                if translated_text and translated_text != transcript:
                    tts_lang_code = lang_code
                    if lang_code == "es": tts_lang_code = "es-ES"
                    elif lang_code == "fr": tts_lang_code = "fr-FR"
                    elif lang_code == "de": tts_lang_code = "de-DE"
                    # Add more mappings as needed
                    texts_to_synthesize_and_queue.append({'text': translated_text, 'lang': tts_lang_code, 'original_timestamp': segment_start_time})

    for item_to_synth in texts_to_synthesize_and_queue:
        text_to_say = item_to_synth['text']
        lang_for_tts = item_to_synth['lang']
        print(f"Thread for segment {segment_start_time:.2f}s: Synthesizing TTS for lang {lang_for_tts}, text: '{text_to_say[:30]}...'")
        synthesized_audio = synthesize_speech(text_to_say, language_code=lang_for_tts)
        if synthesized_audio:
            print(f"Thread for segment {segment_start_time:.2f}s: TTS for lang {lang_for_tts} successful. Queueing for playback.")
            # Queue: (audio_bytes, language_code_for_logging, original_text_for_logging)
            audio_playback_queue.put((synthesized_audio, lang_for_tts, text_to_say))
        else:
            print(f"Thread for segment {segment_start_time:.2f}s: TTS for lang {lang_for_tts} failed.")

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        overlap_filter_strategy=sv.OverlapFilter.NONE,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)

        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


def main(source_video_path: str, target_video_path: str, device: str, mode: Mode, target_languages_list: List[str], stt_language_code: str) -> None:
    video_clip = mp.VideoFileClip(source_video_path)
    audio_clip = video_clip.audio if hasattr(video_clip, 'audio') else None
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    # Start the playback worker thread
    playback_thread = threading.Thread(target=audio_playback_worker, daemon=True)
    playback_thread.start()
    active_audio_threads = [] # To keep track of threads if needed for joining later

    player_detection_model = None
    pitch_detection_model = None
    # Initialize other models to None as needed
    # ball_detection_model = None
    # etc.

    if mode == Mode.PLAYER_DETECTION:
        player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    elif mode == Mode.PITCH_DETECTION:
        pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    # Add other modes as needed, following the pattern
    # elif mode == Mode.BALL_DETECTION:
    #     ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)

    with sv.VideoSink(target_video_path, video_info) as sink:
        frame_iterator = video_clip.iter_frames(fps=video_info.fps, with_times=True)
        for frame_timestamp, frame_rgb in tqdm(frame_iterator, total=int(video_clip.duration * video_info.fps), desc="Processing video"):
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Audio Subclip Processing
            if audio_clip:
                frame_duration = 1.0 / video_info.fps # Duration of one video frame
                segment_start_time = frame_timestamp
                # Ensure segment_end_time does not exceed audio_clip duration
                segment_end_time = min(segment_start_time + frame_duration, audio_clip.duration)

                if segment_start_time < segment_end_time: # Check for valid segment
                    # Extract audio subclip for the current frame's duration
                    audio_subclip = audio_clip.subclip(segment_start_time, segment_end_time)

                    # Convert audio subclip to bytes (PCM 16-bit)
                    # MoviePy's to_soundarray gives raw PCM data.
                    # Default sample rate for many videos is 44100 or 48000.
                    # Google STT often prefers LINEAR16, which is typically signed 16-bit PCM.
                    # audio_subclip.to_soundarray() returns float32/float64 by default, needs conversion.
                    # For simplicity, let's assume a common sample rate and convert.
                    # A more robust solution would get the actual sample rate from audio_clip.fps.

                    # Using a fixed sample rate for now, this should be improved
                    # by getting it from audio_clip.fps if available and valid.
                    assumed_sample_rate = 44100 # Or audio_clip.fps if it's an integer
                    if audio_subclip.fps:
                        assumed_sample_rate = int(audio_subclip.fps)

                    # Convert audio to a byte stream suitable for STT
                    # to_soundarray gives numpy array, usually float. Convert to int16.
                    sound_array = audio_subclip.to_soundarray(fps=assumed_sample_rate, nbytes=2, quantize=True) # nbytes=2 for 16-bit
                    audio_data_bytes = sound_array.tobytes()

                    if audio_data_bytes:
                        transcript = transcribe_audio_chunk(audio_data_bytes, assumed_sample_rate)

                        # Create and start a new thread for processing this audio segment
                        # Pass necessary parameters: audio_data_bytes, assumed_sample_rate, segment_start_time, target_languages_list
                        audio_thread = threading.Thread(
                            target=process_audio_segment_thread,
                            args=(audio_data_bytes, assumed_sample_rate, segment_start_time, target_languages_list, stt_language_code), # Add stt_language_code
                            daemon=True # Daemon threads will exit when the main program exits
                        )
                        audio_thread.start()
                        active_audio_threads.append(audio_thread) # Optional: if you want to join them later

            annotated_frame = frame_bgr.copy()

            # --- Embedded video processing logic for the selected mode ---
            if mode == Mode.PLAYER_DETECTION and player_detection_model:
                result = player_detection_model(annotated_frame, imgsz=1280, verbose=False)[0]
                detections = sv.Detections.from_ultralytics(result)
                annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
                annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
            elif mode == Mode.PITCH_DETECTION and pitch_detection_model:
                result = pitch_detection_model(annotated_frame, verbose=False)[0]
                keypoints = sv.KeyPoints.from_ultralytics(result)
                annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
                    annotated_frame, keypoints, CONFIG.labels)
            # Add other modes as needed, ensuring models and annotators are correctly used

            sink.write_frame(annotated_frame)
        # cv2.destroyAllWindows() # Not needed for headless

    print("Video processing finished. Waiting for audio tasks to complete...")

    # Optional: Wait for processor threads to finish (if not daemon or want to ensure all tasks sent to queue)
    # for t in active_audio_threads:
    #    t.join(timeout=5) # Wait for a bit for threads to finish sending to queue

    # Signal playback queue that no more items are coming from new processor threads
    # Now wait for the playback queue to empty
    while not audio_playback_queue.empty():
        print(f"Waiting for {audio_playback_queue.qsize()} audio segments to play...")
        time.sleep(1)

    print("All queued audio has been played.")

    # Stop the playback worker thread
    global playback_active
    audio_playback_queue.put(None) # Send sentinel to stop worker
    playback_active = False # Explicitly set flag
    playback_thread.join(timeout=5) # Wait for playback thread to finish
    print("Playback thread joined. Exiting.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument(
        '--mode',
        type=lambda x: Mode[x.upper()], # Convert string to Mode enum member
        default=Mode.PLAYER_DETECTION,  # Default to Enum member
        choices=list(Mode)              # Show Enum members as choices
    )
    parser.add_argument(
        '--target_languages',
        type=str,
        default='[]', # Default to an empty list string
        help='JSON string of target language codes (e.g., \'["es", "fr", "de"]\')'
    )
    parser.add_argument(
        '--stt_language',
        type=str,
        default='en-US',
        help='Language code for Speech-to-Text (e.g., "en-US", "es-ES", "fr-FR").'
    )
    args = parser.parse_args()

    parsed_target_languages = []
    try:
        parsed_target_languages = json.loads(args.target_languages)
        if not isinstance(parsed_target_languages, list):
            print("Warning: target_languages is not a list. Disabling translation.")
            parsed_target_languages = []
    except json.JSONDecodeError:
        print("Warning: Invalid JSON string for target_languages. Disabling translation.")
        parsed_target_languages = []

    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        target_languages_list=parsed_target_languages, # Pass the parsed list
        stt_language_code=args.stt_language # Add this
    )
