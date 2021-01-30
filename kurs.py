from tkinter import *
from tkinter import filedialog

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

        Button(text="Выбрать картинку:", command=self.select_image).grid(row=0, column=0,
                                                                         sticky=W,
                                                                         pady=10, padx=10)

        Label(text="Исходное изображение:").grid(
            row=1, column=0, sticky=W,
            padx=10, pady=10)

        self.loadImage()

        Label(text="Восстановленное изображение:").grid(
            row=2, column=0, sticky=W,
            padx=10, pady=10)

        self.root.mainloop()

    def select_image(self):
        # ask the user for the filename
        file_path = filedialog.askopenfilename(filetypes=(
            ("all files", "*.*"), ("jpeg files", "*.jpg"), ("png files", "*.png"), ("gif files", "*.gif"),
            ("bmp files", "*.bmp")))

        # only show the image if they chose something
        if file_path:
            self.loadImage(file_path)
        self.restoreImage(file_path)

    def restoreImage(self, file_path="resources/file.jpg"):
        deblur(file_path, self.restoredPath)
        self.restoredImage = Image.open(self.restoredPath)
        self.restoredImage = self.restoredImage.resize((500, 400), Image.ANTIALIAS)
        self.restoredPhoto = ImageTk.PhotoImage(self.restoredImage)

        self.createCanvasWithImage(self.restoredPhoto, 2, 1)

        # makeHistogram(self.restoredPath, self.hist2Path)
        makeHistogramOpenCV(self.restoredPath, self.hist2Path)
        self.hist2Image = Image.open(self.hist2Path)
        self.hist2Image = self.hist2Image.resize((500, 400), Image.ANTIALIAS)
        self.hist2Photo = ImageTk.PhotoImage(self.hist2Image)

        self.createCanvasWithImage(self.hist2Photo, 2, 2)

        # ФЧХ
        makeMagnituneSpectrumOpenCV(self.restoredPath, self.sp2Path)
        self.spectrum2Image = Image.open(self.sp2Path)
        self.spectrum2Image = self.spectrum2Image.resize((500, 400), Image.ANTIALIAS)
        self.spectrum2Photo = ImageTk.PhotoImage(self.spectrum2Image)

        self.createCanvasWithImage(self.spectrum2Photo, 2, 3)

    def loadImage(self, file_path="resources/dummy.jpg"):
        self.originalImage = Image.open(file_path)
        self.originalImage = self.originalImage.resize((500, 400), Image.ANTIALIAS)
        self.photo1 = ImageTk.PhotoImage(self.originalImage)

        self.createCanvasWithImage(self.photo1, 1, 1)

        # Гистограмма изображения
        makeHistogramOpenCV(file_path, self.hist1Path)
        self.hist1Image = Image.open(self.hist1Path)
        self.hist1Image = self.hist1Image.resize((500, 400), Image.ANTIALIAS)
        self.hist1 = ImageTk.PhotoImage(self.hist1Image)

        self.createCanvasWithImage(self.hist1, 1, 2)

        # ФЧХ
        makeMagnituneSpectrumOpenCV(file_path, self.sp1Path)
        self.spectrum1Image = Image.open(self.sp1Path)
        self.spectrum1Image = self.spectrum1Image.resize((500, 400), Image.ANTIALIAS)
        self.spectrum1Photo = ImageTk.PhotoImage(self.spectrum1Image)

        self.createCanvasWithImage(self.spectrum1Photo, 1, 3)

    def createCanvasWithImage(self, image, row, column):
        canvas = Canvas(height=500, width=500)
        canvas.create_image(0, 0, anchor='nw', image=image)
        canvas.grid(row=row, column=column)


app = App()
