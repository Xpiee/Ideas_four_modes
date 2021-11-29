from PyQt5 import QtGui, QtCore
from PyQt5 import QtWidgets 
import sys
import numpy as np


class ProbDisplay(QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):
        super(ProbDisplay, self).__init__(*args, **kwargs)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.width_hint_size = 600
        self.height_hint_size = 50    
        self.low = 0.5 
        self.high = 0.5
        # self.update_cogload(0.5, 0.5)

    def sizeHint(self):
        return QtCore.QSize(self.width_hint_size,self.height_hint_size)

    def update_cogload(self, low, high):
        self.low = low 
        self.high = high
        self.update()
        

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        # painter.setPen(QtGui.QPen(QtCore.black,  1))
        # self.gradient.setStart(0, painter.device().height())
        brush1 = QtGui.QBrush(QtCore.Qt.blue)
        brush2 = QtGui.QBrush(QtCore.Qt.red)
        width1 = int(painter.device().width() * self.low)
        width2 = int(painter.device().width() * self.high)
        rect1 = QtCore.QRect(0, 0, width1, painter.device().height())
        rect2 = QtCore.QRect(width1, 0, painter.device().width(), painter.device().height())
        painter.fillRect(rect1, brush1)
        painter.fillRect(rect2, brush2)
        # painter.drawRect(0, 0, self.frameGeometry().width(), self.frameGeometry().height())
        painter.end()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ProbDisplay()
    window.show()
    sys.exit(app.exec())
