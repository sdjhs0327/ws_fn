import sys
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.uic import loadUi
import numpy as np


class ScatterPlotApp(QMainWindow):
    def __init__(self):
        super(ScatterPlotApp, self).__init__()
        loadUi("C:/Users/GCC/git/ws_fn/contents/untitled.ui", self)  # .ui 파일 로드

        # UI 요소 연결
        self.plotButton.clicked.connect(self.plot_scatter)  # 버튼 클릭 시 실행
        self.clearButton.clicked.connect(self.clear_inputs)  # 입력 초기화 버튼

    def plot_scatter(self):
        try:
            # 입력값 가져오기
            x_values = list(map(float, self.xInput.text().split(',')))
            y_values = list(map(float, self.yInput.text().split(',')))

            if len(x_values) != len(y_values):
                self.statusLabel.setText("X와 Y의 길이가 일치하지 않습니다.")
                return

            # 산점도 그리기
            plt.scatter(x_values, y_values)
            plt.title("Scatter Plot")
            plt.xlabel("X values")
            plt.ylabel("Y values")
            plt.grid(True)
            plt.show()

        except ValueError:
            self.statusLabel.setText("숫자를 올바르게 입력하세요. (쉼표로 구분)")

    def clear_inputs(self):
        self.xInput.clear()
        self.yInput.clear()
        self.statusLabel.setText("")

# 앱 실행
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = ScatterPlotApp()
    mainWindow.show()
    sys.exit(app.exec_())