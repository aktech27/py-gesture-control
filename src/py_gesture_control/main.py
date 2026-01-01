import cv2 as cv
from py_gesture_control.utils.logger import logger

def main() -> None:
    logger.info("Starting webcam feed...")

    cap = cv.VideoCapture(0, cv.CAP_DSHOW)

    if not cap.isOpened():
        logger.error("No camera detected")
        return

    logger.info("Camera opened. Press 'q' to exit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Can't receive frame (stream end?). Exiting...")
            break

        frame = cv.flip(frame, 1)

        cv.imshow("Webcam Feed", frame)

        # Exit if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            logger.info("Exiting...")
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
