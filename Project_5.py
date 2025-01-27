import sys
from ximea import xiapi
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QLineEdit, QLabel, QTextEdit
from PyQt5.QtCore import pyqtSignal, QObject
from PyQt5.QtGui import QIntValidator
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import gc
from collections import deque

class CameraApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        # Otvorí sa hlavné okno
        self.init_ui()
        # Nastaví sa kamera
        self.init_camera()
        # Definujú sa gobálne premenné a premenné sledujúce stav (True/False)
        self.drawing_line = False
        self.line_points = []
        self.line_drawn = False
        self.dragging = True
        self.crosshair_pos = [640, 480] # Definuje stred interferenčného obrazca
        self.var_distance = 1 # Tieto premenné súvisia s maximami na grafe
        self.var_prominence = 0
        self.var_threshold = 0
        self.var_gap = 0
        self.var_maxima = 4
        self.scaling = None # Definuje škálovanie interferenčného obrazca (px -> mm)
        self.data = None # Snímka z kamery
        self.intensity_frame = None # Kópia snímky, z ktorej sa počíta intenzita
        self.var_avg = 1 # min = 1, max = 10, definuje dĺžku zoznamu snímkov self.frame_buffer
        self.frame_buffer = [] # Zbiera snímky na priemerovanie

    def init_ui(self):
        # Definuje sa veľkosť okna vzhľadomm na rozlíšenie obrazovky
        screen = QtWidgets.QApplication.primaryScreen()
        screen_size = screen.size()
        screen_width, screen_height = screen_size.width(), screen_size.height()

        window_width = int(screen_width)
        window_height = int(screen_height * 0.8)

        self.setWindowTitle('XIMEA Camera Viewer')
        self.setGeometry(0, 50, window_width, window_height)

        # Zobrazenie z kamery
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setFixedSize(int(window_width * 0.61), window_height - 50)
        self.image_label.mousePressEvent = self.get_mouse_click # Na obraz kamery sa dá klikať a registrovať polohu klikov

        # Tlačítka
        self.capture_button = QtWidgets.QPushButton('Capture', self)
        self.capture_button.clicked.connect(self.capture_image)

        self.scale_button = QtWidgets.QPushButton('Scale', self)
        self.scale_button.clicked.connect(self.scale_image)

        self.intensity_button = QtWidgets.QPushButton('Intensity', self)
        self.intensity_button.clicked.connect(self.intensity_calc)
        self.intensity_button.setEnabled(False) # Najskôr treba zadefinovať mierku, až potom sa dá kliknúť na toto tlačítko

        # Inputové polia
        self.var_prominence_input = QLineEdit(self)
        self.var_prominence_input.setText("0")
        self.var_prominence_input.setEnabled(False)

        self.var_distance_input = QLineEdit(self)
        self.var_distance_input.setText("1")
        self.var_distance_input.setEnabled(False)

        self.var_threshold_input = QLineEdit(self)
        self.var_threshold_input.setText("0")
        self.var_threshold_input.setEnabled(False)

        self.var_gap_input = QLineEdit(self)
        self.var_gap_input.setText("0")
        self.var_gap_input.setEnabled(False)

        self.var_maxima_input = QLineEdit(self)
        self.var_maxima_input.setText("4")
        self.var_maxima_input.setEnabled(False)

        self.var_avg_input = QLineEdit(self)
        self.var_avg_input.setText("1")
        self.var_avg_input.setValidator(QIntValidator(1, 100, self))
        self.var_avg_input.setEnabled(False)

        # Updatuje premenné cez inputové polia
        self.var_distance_input.textChanged.connect(lambda: self.update_variable("var_distance"))
        self.var_prominence_input.textChanged.connect(lambda: self.update_variable("var_prominence"))
        self.var_threshold_input.textChanged.connect(lambda: self.update_variable("var_threshold"))
        self.var_gap_input.textChanged.connect(lambda: self.update_variable("var_gap"))
        self.var_maxima_input.textChanged.connect(lambda: self.update_variable("var_maxima"))
        self.var_avg_input.textChanged.connect(lambda: self.update_variable("var_avg"))

        # Súčasťou okna sú grafy
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

        # Vypíše print(), aby sa zobrazovali priamo v okne programu
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sys.stdout = EmittingStream()
        sys.stdout.text_written.connect(self.log_output.append)

        # Rozdelí layout horizontálne, potom pridávame vertikálne časti (celková štruktúra je v stĺpcoch)
        main_layout = QtWidgets.QHBoxLayout()

        # Najviac vľavo je obraz kamery
        image_layout = QtWidgets.QVBoxLayout()
        image_layout.addWidget(self.image_label)

        # V strede sú grafy a výsledky
        plot_layout = QtWidgets.QVBoxLayout()
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, stretch=3)
        plot_layout.addWidget(QtWidgets.QLabel("Console Output:"))
        plot_layout.addWidget(self.log_output, stretch=2)

        # Vpravo sú tlačítka a inputy
        button_layout = QtWidgets.QVBoxLayout()
        button_layout.addWidget(self.capture_button)
        button_layout.addWidget(self.scale_button)
        button_layout.addWidget(self.intensity_button)
        button_layout.addSpacing(20)
        button_layout.addWidget(QLabel("Distance [px]:")) # Parametre pre vyhľadávanie píkov
        button_layout.addWidget(self.var_distance_input)
        button_layout.addWidget(QLabel("Prominence:"))
        button_layout.addWidget(self.var_prominence_input)
        button_layout.addWidget(QLabel("Threshold:"))
        button_layout.addWidget(self.var_threshold_input)
        button_layout.addWidget(QLabel("Gap [px]:"))
        button_layout.addWidget(self.var_gap_input)
        button_layout.addWidget(QLabel("Maxima:"))
        button_layout.addWidget(self.var_maxima_input)
        button_layout.addWidget(QLabel("Avg:"))
        button_layout.addWidget(self.var_avg_input)
        button_layout.addStretch()

        # Tu sa zoradia z ľava do prava 
        main_layout.addLayout(image_layout)
        main_layout.addLayout(plot_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

        # Hlavné okno sa updatuje 30 krát za sekundu
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)    

    def update_variable(self, variable_name): # Updatuje premenné cez input na základe náznu inputu a názvu premenej
        text = getattr(self, f'{variable_name}_input').text()
        try:
            setattr(self, variable_name, int(text))  # Berie sa len integer input
            print(f"Updated {variable_name}: {getattr(self, variable_name)}")
        except ValueError:
            print(f"Only integer input available for {variable_name}") 

    def init_camera(self):
        self.cam = xiapi.Camera()
        self.cam.open_device()
        self.cam.set_imgdataformat('XI_RGB24')
        self.cam.set_exposure(5000) # Toto treba nastaviť podľa sveteľných podmienok, inak môže byť problém nájsť interferenčné maximá
        self.cam.set_wb_kr(5)
        self.cam.set_wb_kg(2.5)
        self.cam.set_wb_kb(5)
        self.cam.start_acquisition()
        self.img = xiapi.Image()

    def update_frame(self):
        self.cam.get_image(self.img)
        self.data = self.img.get_image_data_numpy(invert_rgb_order=True) # Výstup z kamery treba invertovať, inak sa zobrazí B->R a R->B
        self.frame_buffer.append(self.data.copy())
        if len(self.frame_buffer) == self.var_avg: # Priemerovanie obrazu z viacerých snímkov na redukciu šumu
            self.intensity_frame = sum(self.frame_buffer) / len(self.frame_buffer)
            self.frame_buffer = deque(maxlen=self.var_avg) # Aby táto premenná nebola príliš veľká
            self.frame_buffer.clear()
            gc.collect() # Vyhadzuje prebytočné dáta

        if self.data is not None: # Ak máme obraz z kamery, nakreslí sa kríž, ktorý určuje stred interferenčného obrazca
            x, y = self.crosshair_pos
            cv2.line(self.data, (x, 0), (x, self.data.shape[0]), color=(0, 255, 0), thickness=3)
            cv2.line(self.data, (0, y), (self.data.shape[1], y), color=(0, 255, 0), thickness=3)

        if self.line_drawn and len(self.line_points) == 2: # Kreslenie mierky sa vykoná po kliknutí dvoch bodov na obrázku
            pt1, pt2 = self.line_points
            cv2.line(self.data, pt1, pt2, (255, 0, 0), 3)

        q_img = QtGui.QImage(self.data.data, self.data.shape[1], self.data.shape[0], self.data.strides[0], QtGui.QImage.Format_RGB888) # Formát, ktorý dokáže správne zobraziť XI_RGB24
        scaled_q_img = q_img.scaled(self.image_label.size(), QtCore.Qt.KeepAspectRatio)
        self.image_label.setPixmap(QtGui.QPixmap.fromImage(scaled_q_img))

    def capture_image(self): # Pomocou tlačítka "Capture" vieme uložiť fotku (bez kríža a mierky)
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Save Image', '', 'PNG Files (*.png);;JPG Files (*.jpg);;TIFF Files (*.tiff)')[0]
        if filename:
            self.cam.get_image(self.img)
            data = self.img.get_image_data_numpy()
            cv2.imwrite(filename, data)

    def scale_image(self): # Zapína a vypína kreslenie mierky
        if self.drawing_line:
            self.line_points = []
            self.drawing_line = False
            self.line_drawn = False
        else:
            self.line_points = []
            self.drawing_line = True

    def equation_a(self, x_values, distance, R): # Funkcia na výpočet polomeru zakrivenia (A)
        deltas = []
        for i in range(len(x_values)):
            term1 = (R - np.sqrt(R**2 - (1 - distance / (distance + R / 2))**2 * x_values[i]**2))
            term2 = (1 + np.sqrt(x_values[i]**2 + (distance + R / 2)**2) / (distance + R / 2))
            term3 = (distance * np.sqrt(x_values[i]**2 + (distance + R / 2)**2)) / (distance + R / 2)
            delta_i = term1 * term2 + term3
            deltas.append(delta_i)
        return deltas
    
    def equation_b(self, j, deltas, distance, wavelength): # Druhá funkcia na výpočet polomeru (B), ale iným spôsobom. Pri správnom polomere (A) aj (B) dajú rovnaký výsledok
        return deltas[0] + (j+1) * wavelength - distance

    def intensity_image(self): # Metóda na výpočet súradníc intenzity, píkov a bodov vo funkciách (A) a (B)
        # Maximá sa vyrátajú z intenzity horizontálnych pixelov
        crosshair_y = self.crosshair_pos[1]
        row_range = range(crosshair_y - 7, crosshair_y + 8)  # priemerovanie obrazu z viacero horizontálnych čiar
        rows = [self.intensity_frame[y, :, 0].tolist() for y in row_range] # pod intenzitou sa myslí červená farba
        rows_array = np.array(rows)
        average_row = np.mean(rows_array, axis=0)
        row_1 = average_row.tolist()

        window_size = 3
        # Vypočíta priemernú hodnotu z najbližších hodnôt kvôli redukcii šumu
        row = [sum(row_1[i:i+window_size]) / window_size for i in range(len(row_1) - window_size + 1)]
        x_vals = [(i - self.crosshair_pos[0]) for i in range(len(row))] # Nastavíme polohu kríža do nuly
        peaks, _ = find_peaks(row, distance=self.var_distance, height=self.var_threshold, prominence=self.var_prominence)
        average_x = []
        if len(peaks) >= 2:
            right_peaks = []
            left_peaks = []
            len_right_peaks = []
            len_left_peaks = []
            right_gap_peaks = []
            right_len_gap_peaks = []
            left_gap_peaks = []
            left_len_gap_peaks = []

            # Nájdené píky sa vytriedia podľa polohy voči krížu
            for i in range(len(peaks)):
                if self.crosshair_pos[0] > peaks[i]:
                    left_peaks.append(peaks[i])
                    len_left_peaks.append(i)
                if self.crosshair_pos[0] < peaks[i]:
                    right_peaks.append(peaks[i])
                    len_right_peaks.append(i)
                if self.crosshair_pos[0] < peaks[i] and peaks[i] < self.crosshair_pos[0] + self.var_gap:
                    right_gap_peaks.append(peaks[i])
                    right_len_gap_peaks.append(i)
                if self.crosshair_pos[0] > peaks[i] and peaks[i] > self.crosshair_pos[0] - self.var_gap:
                    left_gap_peaks.append(peaks[i])
                    left_len_gap_peaks.append(i)

            right_gap_size = len(right_gap_peaks) # Počet píkov v "gap" - píky v gap sa nerátajú, často sú to artefakty alebo píky s evidentne zlou polohou; pre určenie polomeru nie sú podstatné
            left_gap_size = len(left_gap_peaks)
            if right_gap_size > 0:
                right_peaks = right_peaks[right_gap_size:]
            if left_gap_size > 0:
                left_peaks = left_peaks[:-left_gap_size]

            # Vyráta sa priemerná vzdialenosť píku od stredu doprava a doľava, pre presnejší výsledok
            for i in range(min(len(left_peaks), len(right_peaks), self.var_maxima)):
                average_x.append(((right_peaks[i] - self.crosshair_pos[0]) + (self.crosshair_pos[0] - left_peaks[-i-1])) / 2)
            print(f"The program is considering {min(len(left_peaks), len(right_peaks), self.var_maxima)} pairs of peaks.")
            print(f"Average peak pair distance in [px]: {average_x}")
            for i in range(min(len(left_peaks), len(right_peaks), self.var_maxima)):
                average_x[i] = average_x[i] * self.scaling
            formatted_avgx_list = []
            for x_i in average_x:
                y_i = x_i * 1e3
                rounded_y_i = round(y_i, 2)
                formatted_avgx_list.append(str(rounded_y_i))
            formatted_avgx_str = ", ".join(formatted_avgx_list)
            print(f"Average peak pair distance in [mm]: [{formatted_avgx_str}]")

            all_peaks = left_peaks + right_peaks # tieto píky sa nakreslia

            x_c = x_vals
            y_c = row
            x_int_coordinates = [x_vals[i] for i in all_peaks]
            y_int_coordinates = [row[i] for i in all_peaks]

        else:
            print("Intensity profile can't be plotted, insufficient number of peaks.")

        # premenné na výpočet polomeru zakrivenia
        distance = 263 * 0.001
        wavelength = 650 / 1e9
        radius = 0.01
        goodness = []

        if len(average_x) >= 1:
        # vyráta sa "goodness" parameter pre daný polomer - niečo ako v metóde najmenších štvorcov
            while radius <= 20:
                deltas = self.equation_a(average_x, distance, radius)
                distances_1 = [d - distance for d in deltas]
                distances_2 = [deltas[0] - distance]
                distances_2.extend([self.equation_b(j, deltas, distance, wavelength) for j in range(len(average_x)-1)])
                elements = [(x - y)**2 for x, y in zip(distances_1, distances_2)]
                goodness.append(sum(elements))
                radius += 0.01

            min_index = goodness.index(min(goodness))
            radius = 0.01 + min_index * 0.01
            deltas = self.equation_a(average_x, distance, radius)
            distances_1 = [d - distance for d in deltas]
            distances_2 = [deltas[0] - distance]
            distances_2.extend([self.equation_b(j, deltas, distance, wavelength) for j in range(len(average_x)-1)])
            elements = [(x - y)**2 for x, y in zip(distances_1, distances_2)]
            var_goodness = sum(elements)

            # Výsledné parametre
            formatted_goodness = "{:.2e}".format(var_goodness)
            print(f"Goodness: {formatted_goodness} m^2")
            print(f"Radius of curvature = {radius:.2f} m")
        else:
            print("Radius can't be found, insufficient number of peaks.")

        if len(average_x) >= 1:
            x_coordinate = []
            for i in range(len(average_x)):
                x_coordinate.append(i+1)
            if len(deltas) >= 1:
                x_r_coordinates = x_coordinate
                y_r_1_coordinates = distances_1
                y_r_2_coordinates = distances_2
            else:
                print("Figure can't be plotted, insufficient number of peaks.")
            
        else:
            print("Figure can't be plotted, insufficient number of peaks.")

        return x_c, y_c, x_int_coordinates, y_int_coordinates, x_r_coordinates, y_r_1_coordinates, y_r_2_coordinates
    
    def intensity_calc(self):
        x_11, y_11, x_12, y_12, x_22, y_21, y_22 = self.intensity_image()
        self.figure.clear()  # Vymaže predošlý graf

        # Dva grafy pod sebou
        ax1 = self.figure.add_subplot(2, 1, 1)
        ax1.plot(x_11, y_11, label='intensity', color='blue')
        ax1.plot(x_12, y_12, 'x', label='peaks', color='red')
        ax1.set_ylabel("Intensity")
        ax1.legend()
        ax1.grid(True)

        ax2 = self.figure.add_subplot(2, 1, 2)
        ax2.plot(x_22, y_22, label='Equation (B)', color='blue')
        ax2.plot(x_22, y_21, 'x', label='Equation (A)', color='red')
        ax2.set_ylabel("R")
        ax2.legend()
        ax2.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()

    def get_mouse_click(self, event):
        if self.drawing_line and len(self.line_points) < 2:
            self.dragging = False
            x = int(event.pos().x() * self.img.width // self.image_label.width())
            y = int(event.pos().y() * self.img.height // self.image_label.height())
            self.line_points.append((x, y))
            if len(self.line_points) == 2:
                self.scaling = (10e-3)/np.sqrt((self.line_points[1][0] - self.line_points[0][0])**2 + (self.line_points[1][1] - self.line_points[0][1])**2)
                self.drawing_line = False
                self.line_drawn = True
                self.dragging = True
                self.intensity_button.setEnabled(True)
                self.var_distance_input.setEnabled(True)
                self.var_prominence_input.setEnabled(True)
                self.var_threshold_input.setEnabled(True)
                self.var_gap_input.setEnabled(True)
                self.var_maxima_input.setEnabled(True)
                self.var_avg_input.setEnabled(True)
        else:
            self.dragging = True
            x = int(event.pos().x() * self.img.width // self.image_label.width())
            y = int(event.pos().y() * self.img.height // self.image_label.height())
            self.crosshair_pos = [x, y]

    def closeEvent(self, event):
        self.cam.stop_acquisition()
        self.cam.close_device()
        event.accept()

class EmittingStream(QObject):
    # Presmeruje print() texty do okna programu
    text_written = pyqtSignal(str)

    def write(self, text):
        self.text_written.emit(str(text))

    def flush(self):
        pass

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())
