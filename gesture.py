import sys
from time import sleep

import cv2
import numpy as np
import math
from PyQt4 import QtGui, QtCore
from enum import Enum

from PyQt4.QtCore import QObject, pyqtSlot
from PyQt4.QtCore import pyqtSignal
from PyQt4.QtCore import QThread
from PyQt4.QtGui import QWidget

Gestures = Enum('Gestures', 'ONE TWO THREE FOUR FIVE')
gesture = QtCore.pyqtSignal(Gestures)
'''
class AThread(QtCore.QThread):
    # Signals
    Gestures = Enum('Gestures', 'ONE TWO THREE FOUR FIVE')
    gesture = QtCore.pyqtSignal(Gestures)

    def run(self):'''

def run_gestures():
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, img = cap.read()
        cv2.rectangle(img,(300,300),(100,100),(0,255,0),0)
        crop_img = img[100:300, 100:300]
        grey = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        value = (35, 35)
        blurred = cv2.GaussianBlur(grey, value, 0)
        _, thresh1 = cv2.threshold(blurred, 127, 255,
                                   cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        #cv2.imshow('Thresholded', thresh1)

        (version, _, _) = cv2.__version__.split('.')

        if version is '3':
            image, contours, hierarchy = cv2.findContours(thresh1.copy(), \
                   cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        elif version is '2':
            contours, hierarchy = cv2.findContours(thresh1.copy(),cv2.RETR_TREE, \
                   cv2.CHAIN_APPROX_NONE)

        cnt = max(contours, key = lambda x: cv2.contourArea(x))

        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(crop_img,(x,y),(x+w,y+h),(0,0,255),0)
        hull = cv2.convexHull(cnt)
        drawing = np.zeros(crop_img.shape,np.uint8)
        cv2.drawContours(drawing,[cnt],0,(0,255,0),0)
        cv2.drawContours(drawing,[hull],0,(0,0,255),0)
        hull = cv2.convexHull(cnt,returnPoints = False)
        defects = cv2.convexityDefects(cnt,hull)
        count_defects = 0
        cv2.drawContours(thresh1, contours, -1, (0,255,0), 3)
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_img,far,1,[0,0,255],-1)
            #dist = cv2.pointPolygonTest(cnt,far,True)
            cv2.line(crop_img,start,end,[0,255,0],2)
            #cv2.circle(crop_img,far,5,[0,0,255],-1)
        current_gesture = count_defects
        if count_defects == Gestures.ONE.value:
            cv2.putText(img,"closed fist", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
            #gesture.emit(self.Gestures.CLOSED)

        elif count_defects == 2:
            str = "Two fingers"
            cv2.putText(img, str, (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        elif count_defects == 3:
            cv2.putText(img,"three fingers", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        elif count_defects == 4:
            cv2.putText(img,"four fingers", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        else:
            cv2.putText(img,"five fingers", (50,50),\
                        cv2.FONT_HERSHEY_SIMPLEX, 2, 2)
        #cv2.imshow('drawing', drawing)
        #cv2.imshow('end', crop_img)
        cv2.imshow('Gesture', img)
        all_img = np.hstack((drawing, crop_img))
        #cv2.imshow('Contours', all_img)
        k = cv2.waitKey(10)
        if k == 27:
            break




class Worker(QtCore.QThread):
    def __init__(self, thread_name, finished_slot, function, *args, **kwargs):
        QtCore.QThread.__init__(self)

        self._thread_name = thread_name
        self._function = function
        self._args = args
        self._kwargs = kwargs

        self._finished_slot = finished_slot

    def run(self):
        self._function(*self._args, **self._kwargs)

        self._finished_slot()

        return


class Signals(QObject):
    update = pyqtSignal(int)
    enable_button = pyqtSignal(bool)

class Window(QWidget):
    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args, **kwargs)

        self.button = QtGui.QPushButton("Run", self)
        self.button.clicked.connect(self.onButton)

        self.progress = QtGui.QProgressBar(self)
        self.progress.setTextVisible(False)

        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.progress)
        self.layout.addStretch()

        self.worker_thread = QThread()
        self.worker_thread.run = run_gestures #self.worker
        self.worker_thread.should_close = False

        self.signals = Signals()
        self.signals.update.connect(self.progress.setValue)
        self.signals.enable_button.connect(self.button.setEnabled)

        self.setLayout(self.layout)
        self.show()
        self.resize(self.size().width(), 0)

    # Override
    def closeEvent(self, e):
        self.worker_thread.should_close = True
        self.worker_thread.wait()

    @pyqtSlot()
    def onButton(self):
        self.button.setDisabled(True)
        self.worker_thread.start()

    # Worker thread, no direct GUI updates!
    def worker(self):
        for i in range(1):
            if self.worker_thread.should_close:
                break
            self.signals.update.emit(i)
            sleep(0.1)
        self.signals.enable_button.emit(True)


def main():
    app = QtGui.QApplication(sys.argv)
    win = Window()
    sys.exit(app.exec_())

if __name__ == '__main__':
    #main()
    run_gestures()