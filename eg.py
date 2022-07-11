from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import cv2
app = QApplication(sys.argv)

x = QWidget()
x.y = QLabel(x)
x.y.setText("a0")

#x.setContentsMargins(0,0,0,0)
#x.y.setGeometry(QRect(0, 0, 980, 673))


img = cv2.imread("examples/test_00.jpg")
height, width, bytesPerComponent = img.shape
print(img.shape)

bytesPerLine = 3 * width

cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)

QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
pixmap = QPixmap.fromImage(QImg)
x.y.setPixmap(pixmap)

x.show()

app.exec_()