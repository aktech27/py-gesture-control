import cv2 as cv
import mediapipe as mp
from py_gesture_control.utils.logger import logger

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


MODEL_PATH = "src/models/hand_landmarker.task"
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

hand_land_res = None


def print_result(
    result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int
):
    if result.hand_landmarks:
        # logger.info(f"Hands detected: {len(result.hand_landmarks)}")
        global hand_land_res
        hand_land_res = result.hand_landmarks
    else:
        hand_land_res = None
        # logger.info("No hands")


def main() -> None:
    logger.info("Starting webcam feed...")

    hand_landmark_options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        num_hands=6,
    )
    with HandLandmarker.create_from_options(hand_landmark_options) as landmarker:
        cap = cv.VideoCapture(0, cv.CAP_DSHOW)

        if not cap.isOpened():
            logger.error("No camera detected")
            return

        logger.info("Camera opened. Press 'q' to exit...")

        while True:
            success, frame = cap.read()
            if not success:
                logger.warning("Can't receive frame (stream end?). Exiting...")
                break

            frame = cv.flip(frame, 1)

            rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(cv.getTickCount() / cv.getTickFrequency() * 1000)

            # Async detection
            landmarker.detect_async(mp_image, timestamp_ms)

            if hand_land_res:
                h, w, _ = frame.shape
                for hand in hand_land_res:
                    for start_idx, end_idx in HAND_CONNECTIONS:
                        p1 = hand[start_idx]
                        p2 = hand[end_idx]

                        x1, y1 = int(p1.x * w), int(p1.y * h)
                        x2, y2 = int(p2.x * w), int(p2.y * h)

                        cv.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    for lm in hand:
                        x = int(lm.x * w)
                        y = int(lm.y * h)

                        cv.circle(frame, (x, y), 4, (0, 255, 0), -1)

            cv.imshow("Webcam Feed", frame)

            # Exit if 'q' is pressed
            if cv.waitKey(1) & 0xFF == ord("q"):
                logger.info("Exiting...")
                break

        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
