from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import sys
import cv2
'''
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
'''

'''
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
 
    def initUI(self):
        # 设置窗口标题和大小
        self.setWindowTitle('TestWindow')
        self.resize(400, 300)
 
        self.collec_btn = QPushButton('打开新窗口', self)
 
        layout = QVBoxLayout()
        layout.addWidget(self.collec_btn)
        self.setLayout(layout)
 
        self.show()
 
 
class NewWindow(QWidget):
    def __init__(self):
        super(NewWindow, self).__init__()
        self.setWindowTitle('新窗口')
        self.resize(280, 230)

 
if __name__ == '__main__':
 
    app = QApplication(sys.argv)

    window = MainWindow()
    newWin = NewWindow()
    window.show()
    window.collec_btn.clicked.connect(newWin.show)

    sys.exit(app.exec_())
'''

from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

print("START UI")

class firstWindow(QWidget):
  def __init__(self, *args, **kwargs):
    super(firstWindow, self).__init__(*args, **kwargs)
    self.image_path = './examples/test.jpg'
    self.setWindowTitle('Detect objects. Please input custom labels.')
    
    self.customLabel = QLineEdit()
    
    img = cv2.imread("examples/test.jpg")
    self.height, self.width, self.bytesPerComponent = img.shape
    self.resize(self.width, self.height)
    
    ## TEST
    self.collec_btn = QPushButton('打开新窗口', self)
    layout = QVBoxLayout()
    layout.addWidget(self.collec_btn)
    layout.addWidget(self.collec_btn)
    self.setLayout(layout)
    
  def showpath(self):
    print(self.image_path)



class resultWindow(QMainWindow):
  def __init__(self, parent=None):
    super(resultWindow, self).__init__(parent)
    self.setWindowTitle('Detected objects. Zero-shot ends.')
    img = cv2.imread("examples/test.jpg")
    self.height, self.width, self.bytesPerComponent = img.shape
    self.resize(self.width, self.height)
    self.lb = QLabel()
    self.lb.setText("a0")
    self.parentt = parent
    self.parentt.image_path="ll"
    self.parentt.showpath()
    

app = QApplication(sys.argv)
win_1 = firstWindow()

def showshow():
    print("OK")
    win_2 = resultWindow(win_1)
    win_2.show()
win_1.collec_btn.clicked.connect(showshow)
win_1.show()
sys.exit(app.exec_())