from PyQt5 import QtGui, QtCore
from PyQt5 import QtWidgets 
import sys
import numpy as np


class CogloadGradientDisplay(QtWidgets.QWidget):

    def __init__(self, *args, **kwargs):
        super(CogloadGradientDisplay, self).__init__(*args, **kwargs)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.MinimumExpanding,
            QtWidgets.QSizePolicy.MinimumExpanding
        )
        self.width_hint_size = 50
        self.width_hint_size = 600
        self.MAXLEVEL = 6
        self.blue = 2
        self.red = 4
        self.color = QtGui.QColor(int(self.red /self.MAXLEVEL) * 255, 0, int(self.red /self.MAXLEVEL))
        self.gradient = self.gradient = QtGui.QLinearGradient(QtCore.QPointF(0, self.width_hint_size), QtCore.QPointF(0, 0))
        self.gradient.setColorAt(0.0, QtCore.Qt.blue)
        self.gradient.setColorAt(1., QtCore.Qt.red)
        self.update_cogload(0)

    def sizeHint(self):
        return QtCore.QSize(50,600)

    def update_cogload(self, cog_load):
        height = self.geometry().height()
        incremenet = 1 if cog_load == 1 else -1
        self.red = min(max(self.red + incremenet, 0), self.MAXLEVEL)
        self.blue = self.MAXLEVEL - self.red 
        scale_factor_blue =  height * (3 - self.blue) / 6.
        scale_factor_red = height * (self.red - 3.) / 6.
        self.gradient = QtGui.QLinearGradient(QtCore.QPointF(0, height+ scale_factor_blue), QtCore.QPointF(0, scale_factor_red))
        self.gradient.setColorAt(0.0, QtCore.Qt.blue)
        self.gradient.setColorAt(1., QtCore.Qt.red)
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        # painter.setPen(QtGui.QPen(QtCore.black,  1))
        # self.gradient.setStart(0, painter.device().height())
        brush = QtGui.QBrush(self.gradient)
        rect = QtCore.QRect(0, 0, painter.device().width(), painter.device().height())
        painter.fillRect(rect, brush)
        # painter.drawRect(0, 0, self.frameGeometry().width(), self.frameGeometry().height())
        painter.end()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = CogloadGradientDisplay()
    window.show()
    sys.exit(app.exec())
