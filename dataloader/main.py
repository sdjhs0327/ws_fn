## 야후 파이낸스 패키지
## C:\Users\GCC\git\ws_fn\.venv\Lib\site-packages\qt5_applications\Qt\bin
## pyuic5 -x dataloader/main.ui -o dataloader/main_ui.py
## pyinstaller --onefile dataloader/main.py
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
## PyQt5 패키지
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.uic import loadUi
import sys
import time
from main_ui import Ui_MainWindow

class Dataloader(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Dataloader, self).__init__()
        self.setupUi(self)  # setupUi 메서드를 호출하여 UI 설정
        self.nowdate = datetime.now().strftime('%Y%m%d')
        
        self.pathInput.setReadOnly(True)
        self.pathInput.setText("C:")
        
        # UI 요소 연결
        self.downloadButton.clicked.connect(self.dataloader)  # 데이터 다운로드
        self.selectPathButton.clicked.connect(self.select_save_path)  # 경로 선택 버튼
        
    def select_save_path(self):
        # QFileDialog로 저장 경로 선택
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory", "C:")
        if folder:  # 사용자가 선택한 경우
            self.pathInput.setText(folder)
        else:  # 사용자가 취소한 경우
            self.statusLabel.setText("Status: 저장 경로 선택이 취소되었습니다.")
    
    def dataloader(self):
        try:
            # 입력값 가져오기
            tickers = self.tickerInput.text().split(',')
            save_path = self.pathInput.text()
            if save_path[-1] == '/':
                save_path = save_path[:-1]
            if len(tickers) <= 0:
                self.statusLabel.setText("Status: 정확한 <Ticker>를 입력해주세요.")
                return
            # 데이터 다운로드
            dataset = yf.download(tickers, ignore_tz=True, auto_adjust=True)
            dataset = dataset['Close']
            # 엑셀로 저장
            dataset.to_excel(save_path + f"/{tickers[0]}_{self.nowdate}.xlsx")
            
            self.statusLabel.setText("Status: 다운로드 완료")
        except ValueError:
            self.statusLabel.setText("Status: 입력값을 확인해주세요.")
            
# 앱 실행
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = Dataloader()
    mainWindow.show()
    sys.exit(app.exec_())