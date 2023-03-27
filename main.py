import cv2
from PIL import Image
import utils


class Camera:
    def __init__(self):
        self.green = [0, 255, 0]  # green in BGR colorspace
        self.cap = cv2.VideoCapture(0)
        self.cap.set(10, 160)
        self.cap.set(3, 1920)
        self.cap.set(4, 1080)
        self.scale = 3
        self.wP = 210 * self.scale
        self.hP = 297 * self.scale

    def capture_frame(self):
        ret, frame = self.cap.read()
        hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lowerLimit, upperLimit = utils.get_limits(color=self.green)
        mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
        mask_ = Image.fromarray(mask)
        bbox = mask_.getbbox()
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            roi = frame[y1:y2, x1:x2]
            imgContours, conts = utils.getContours(roi, minArea=50000, filter=4)
            if len(conts) != 0:
                biggest = conts[0][2]
                imgWarp = utils.warpImg(roi, biggest, self.wP, self.hP)
                imgContours2, conts2 = utils.getContours(imgWarp, minArea=2000, filter=4, cThr=[50, 50], draw=False)
                if len(conts2) != 0:
                    for obj in conts2:
                        cv2.polylines(imgContours2, [obj[2]], True, (0, 255, 0), 2)
                        nPoints = utils.reorder(obj[2])
                        nW = round((utils.findDis(nPoints[0][0] // self.scale, nPoints[1][0] // self.scale) / 10), 1)
                        nH = round((utils.findDis(nPoints[0][0] // self.scale, nPoints[2][0] // self.scale) / 10), 1)
                        cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                        (nPoints[1][0][0], nPoints[1][0][1]),
                                        (255, 0, 255), 3, 8, 0, 0.05)
                        cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]),
                                        (nPoints[2][0][0], nPoints[2][0][1]),
                                        (255, 0, 255), 3, 8, 0, 0.05)
                        x, y, w, h = obj[3]
                        cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                                    (255, 0, 255), 2)
                        cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                                    (255, 0, 255), 2)

