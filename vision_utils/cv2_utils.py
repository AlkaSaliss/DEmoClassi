from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2


# Instantiate a dlib
detector = dlib.get_frontal_face_detector()

vs = VideoStream(src=0).start()
time.sleep(2)

if __name__ == '__main__':

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=224, height=224)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detects faces in the grayscale image
        rects = detector(gray, 0)

        # if faces found
        if len(rects) > 0:
            for rect in rects:
                # get and plot bounding box for each face
                (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
                cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH), (0, 0, 255), 1)
            # Show the frame with the bounding boxes
        cv2.imshow('Frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    vs.stop()
