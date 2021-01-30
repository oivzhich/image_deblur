import numpy as np
import matplotlib.pyplot as plt
import cv2


def makeHistogram(file_path, result_path):
    # чтение файла с диска
    image = cv2.imread(file_path)
    # вычисение среднего значение из каналов RGB и сглаживание до одномерного массива
    vals = image.mean(axis=2).flatten()
    # вычисление гистограммы
    counts, bins = np.histogram(vals, range(257))
    # создание графика гистограммы
    figure = plt.figure(0)
    figure.clear()
    counts, bins = np.histogram(vals, range(257))
    plt.bar(bins[:-1] - 0.5, counts, width=1, edgecolor='none')
    plt.xlim([-0.5, 255.5])
    plt.savefig(result_path)


def makeHistogramOpenCV(file_path, result_path):
    # чтение файла с диска
    image = cv2.imread(file_path)
    # вычисление гистограммы
    hist_full = cv2.calcHist([image], [0], None, [256], [0, 256])
    # создание графика гистограммы
    figure = plt.figure(0)
    figure.clear()
    plt.plot(hist_full)
    plt.title('АЧХ')
    plt.xlim([0, 256])
    # cохраниение графика гистограммы в файл
    plt.savefig(result_path)
