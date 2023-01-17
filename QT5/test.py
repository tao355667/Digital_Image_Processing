#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
ZetCode PyQt5 tutorial
In this example, we select a file with a
QFileDialog and display its contents
in a QTextEdit.
Author: Jan Bodnar
Website: zetcode.com
Last edited: August 2017
"""

from PyQt5.QtWidgets import (QMainWindow, QTextEdit,
                             QAction, QFileDialog, QApplication)
from PyQt5.QtGui import QIcon
import sys


class Example(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.textEdit = QTextEdit()
        self.setCentralWidget(self.textEdit)
        self.statusBar()

        ActNew = QAction(QIcon('open.png'), 'New', self)
        ActNew.setShortcut('Ctrl+O')
        ActNew.setStatusTip('Open new File')
        ActNew.triggered.connect(self.showDialog)

        ActOpen = QAction(QIcon('open.png'), 'Open', self)
        ActOpen.setShortcut('Ctrl+I')
        ActOpen.setStatusTip('Open new File')
        ActOpen.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        fileText = menubar.addMenu('文件菜单')
        fileText.addAction(ActNew)
        fileText.addAction(ActOpen)

        AppText = menubar.addMenu('&编辑')

        self.setGeometry(300, 300, 750, 600)
        self.setWindowTitle('Huatec AI')
        self.show()

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')

        if fname[0]:
            f = open(fname[0], 'r')

            with f:
                data = f.read()
                self.textEdit.setText(data)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = Example()
    sys.exit(app.exec_())
