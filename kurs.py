from tkinter import *
from tkinter import filedialog

import numpy as np
from PIL import Image, ImageTk

from gist import makeHistogramOpenCV
from spectrum import makeMagnituneSpectrumOpenCV
from weiner import deblur


class App:
    def __init__(self):
        self.hist1Path = 'resources/hist1.jpg'
        self.hist2Path = 'resources/hist2.jpg'
        self.sp1Path = 'resources/sp1.jpg'
        self.sp2Path = 'resources/sp2.jpg'
        self.restoredPath = 'resources/restored.jpg'
        self.root = Tk()
        self.root.title("ПРОГРАММА ВОССТАНОВЛЕНИЯ РАСФОКУСИРОВАННОГО ИЗОБРАЖЕНИЯ")
        self.dsize = (500, 400)

        Button(text="Выбрать картинку:", command=self.select_image).grid(row=0, column=0,
                                                                         sticky=W,
                                                                         pady=10, padx=10)

        Label(text="Исходное изображение:").grid(
            row=1, column=0, sticky=W,
            padx=10, pady=10)

        self.loadImage()

        self.psf_var = DoubleVar()
        self.psf_var.set(13)
        self.psf_scale = Scale(self.root, label="Значение радиуса PSF", orient=HORIZONTAL, length=300, from_=0, to=100,
                               tickinterval=10, variable=self.psf_var, resolution=1,
                               command=self.psfScaleHandler).grid(
            row=2, column=0)

        self.noise_var = DoubleVar()
        self.noise_var.set(25)

        self.noise_scale = Scale(self.root, label="Значение SNR", orient=HORIZONTAL, length=300, from_=0, to=100,
                                 tickinterval=10, variable=self.noise_var, resolution=1,
                                 command=self.noiseScaleHandler).grid(
            row=2, column=1)

        Label(text="Восстановленное изображение:").grid(
            row=4, column=0, sticky=W, pady=10, padx=10)

        self.root.mainloop()

    def select_image(self):
        # ask the user for the filename
        file_path = filedialog.askopenfilename(filetypes=(
            ("all files", "*.*"), ("jpeg files", "*.jpg"), ("png files", "*.png"), ("gif files", "*.gif"),
            ("bmp files", "*.bmp")))

        # only show the image if they chose something
        if file_path:
            self.loadImage(file_path)
        self.restoreImage(self.originalImage, self.noise_var.get(), self.psf_var.get())

    def noiseScaleHandler(self, noise_var):
        self.restoreImage(self.originalImage, noise_var, self.psf_var.get())

    def psfScaleHandler(self, psf_var):
        self.restoreImage(self.originalImage, self.noise_var.get(), psf_var)

    def _getPILImage(self, file_path):
        # чтение файла с диска
        pil_image = Image.open(file_path).convert('RGB')
        return pil_image.resize(self.dsize, Image.ANTIALIAS)

    def _getOpenCVImage(self, pil_image):
        # Convert RGB to BGR
        return np.array(pil_image)[:, :, ::-1].copy()

    def _createCanvasWithImage(self, image, row, column):
        canvas = Canvas(height=400, width=500)
        canvas.create_image(0, 0, anchor='nw', image=image)
        canvas.grid(row=row, column=column)
        return image

    def loadImage(self, file_path="resources/dummy.jpg"):
        pil_image = self._getPILImage(file_path)
        self.originalImage = self._getOpenCVImage(pil_image)
        self.photo1 = self._createCanvasWithImage(ImageTk.PhotoImage(pil_image), 1, 1)

        # Гистограмма изображения
        makeHistogramOpenCV(self.originalImage, self.hist1Path)
        self.hist1Image = self._getPILImage(self.hist1Path)
        self.hist1 = self._createCanvasWithImage(ImageTk.PhotoImage(self.hist1Image), 1, 2)

        # ФЧХ
        makeMagnituneSpectrumOpenCV(self.originalImage, self.sp1Path)
        self.spectrum1Image = self._getPILImage(self.sp1Path)
        self.sp1 = self._createCanvasWithImage(ImageTk.PhotoImage(self.spectrum1Image), 1, 3)

    def restoreImage(self, original_image, noise_var, psf_var):
        if original_image is None: return
        deblur(original_image, self.restoredPath, int(noise_var), int(psf_var))

        pil_image = self._getPILImage(self.restoredPath)
        self.restoredImage = self._getOpenCVImage(pil_image)
        self.photo2 = self._createCanvasWithImage(ImageTk.PhotoImage(pil_image), 3, 1)

        # Гистограмма изображения
        makeHistogramOpenCV(self.restoredImage, self.hist2Path)
        self.hist2Image = self._getPILImage(self.hist2Path)
        self.hist2 = self._createCanvasWithImage(ImageTk.PhotoImage(self.hist2Image), 3, 2)

        # ФЧХ
        makeMagnituneSpectrumOpenCV(self.restoredImage, self.sp2Path)
        self.spectrum2Image = self._getPILImage(self.sp2Path)
        self.sp2 = self._createCanvasWithImage(ImageTk.PhotoImage(self.spectrum2Image), 3, 3)


app = App()
