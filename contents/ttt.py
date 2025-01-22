from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QLineEdit, QPushButton, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ScatterPlotApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("사용자 입력 산점도")
        self.setGeometry(100, 100, 600, 400)

        self.initUI()

    def initUI(self):
        # 중앙 위젯과 레이아웃 설정
        central_widget = QWidget()
        layout = QVBoxLayout()

        # 입력 안내 레이블
        self.input_label = QLabel("5개의 숫자 포인트를 입력하세요 (예: 1,2,3,4,5):")
        layout.addWidget(self.input_label)

        # 사용자 입력 필드
        self.input_field = QLineEdit()
        layout.addWidget(self.input_field)

        # 버튼: 그래프 그리기
        self.plot_button = QPushButton("산점도 그리기")
        self.plot_button.clicked.connect(self.plot_scatter)
        layout.addWidget(self.plot_button)

        # Matplotlib 캔버스
        self.canvas = MatplotlibCanvas(self, width=5, height=4)
        layout.addWidget(self.canvas)

        # 중앙 위젯에 레이아웃 설정
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def plot_scatter(self):
        # 사용자 입력값 가져오기
        user_input = self.input_field.text()
        try:
            # 입력값을 숫자 리스트로 변환
            points = list(map(float, user_input.split(',')))
            if len(points) != 5:
                QMessageBox.warning(self, "입력 오류", "정확히 5개의 숫자를 입력해야 합니다!")
                return
            
            # 산점도 그리기
            self.canvas.plot_scatter(points)

        except ValueError:
            QMessageBox.warning(self, "입력 오류", "올바른 숫자를 입력해주세요!")

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)

    def plot_scatter(self, points):
        self.axes.clear()
        x = range(1, len(points) + 1)  # X 좌표: 1, 2, 3, ...
        y = points  # Y 좌표: 사용자 입력 값
        self.axes.scatter(x, y, color="blue", label="Points")
        self.axes.set_title("사용자 입력 산점도")
        self.axes.set_xlabel("Index")
        self.axes.set_ylabel("Value")
        self.axes.legend()
        self.draw()

if __name__ == "__main__":
    app = QApplication([])
    window = ScatterPlotApp()
    window.show()
    app.exec_()
