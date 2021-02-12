import cv2
import matplotlib.pyplot as plt
import numpy as np


def makeMagnituneSpectrumOpenCV(original_image, result_path):
    if original_image is None: return
    # конвертация изображения в черно-белое
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # каждое значение массива трансформируем в тип float
    img_float32 = np.float32(image)
    # каждое значение яркости трансформируем в относительное значение (между 0 и 1)
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    # cдвиг компонент нулевой частоты в центр спектра
    dft_shift = np.fft.fftshift(dft)
    # вычисление спектра изображения
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    magnitude_spectrum = 20 * np.log(np.abs(magnitude_spectrum))

    # создание графика ФЧХ изображения
    # создание новой фигуры в рамках области видимости matpotlib
    f = plt.figure(1)
    # очищение ранее созданной фигуры
    f.clear()
    # создание графика ФЧХ
    plt.imshow(magnitude_spectrum, cmap='gray')
    # добавление названия к графику
    plt.title('ФЧХ')
    # cохранение графика ФЧХ в файл
    plt.savefig(result_path)

def getMagnituneSpectrum(file_path, result_path):
    # чтение файла с диска
    image = cv2.imread(file_path, 0)
    # разложение изображения в матрицу комплексных чисел
    f = np.fft.fft2(image)
    # сдвиг компонент нулевой частоты в центр спектра
    fshift = np.fft.fftshift(f)
    # вычисление спектра изображения
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # создание графика ФЧХ изображения
    f = plt.figure(1)
    f.clear()
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('ФЧХ')
    # cохраниение графика ФЧХ в файл
    plt.savefig(result_path)
