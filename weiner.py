import cv2
import numpy as np

def defocus_kernel(d, sz=65):
    kern = np.zeros((sz, sz), np.uint8)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern

def blur_edge(img, d=31):
    #сохраняем размеры изображения
    h, w = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2 * d + 1, 2 * d + 1), -1)[d:-d, d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w - x - 1, y, h - y - 1]).min(-1)
    w = np.minimum(np.float32(dist) / d, 1.0)
    return img * w + img_blur * (1 - w)

def write_image(path, img):
    # img = img*(2**16-1)
    # img = img.astype(np.uint16)
    # img = img.astype(np.uint8)
    img = cv2.convertScaleAbs(img, alpha=(255.0))
    cv2.imwrite(path, img)

def deblur(file_path, restored_path):
    # чтение файла с диска
    # https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
    # исходное изображение трансформируется в черно белое (зачем)?
    img = cv2.imread(file_path, 0)

    # каждое значение массива трансформируем в тип float
    # каждое значение яркости трансформируем в относительное значение (между 0 и 1)
    img = np.float32(img) / 255.0

    # сглаживаем края изображения чтобы восстановление было равномерныхм
    img = blur_edge(img)

    # двумерное фурье преобразование
    IMG = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

    ang = np.deg2rad(135)
    d = 22
    noise = 10**(-0.1*25)

    #вычисление спектра мощности исходного изображения
    psf = defocus_kernel(d)

    # if defocus:
    #     psf = defocus_kernel(d)
    # else:
    #     psf = motion_kernel(ang, d)

    psf /= psf.sum()
    psf_pad = np.zeros_like(img)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf
    # двумерное фурье преобразование
    PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
    PSF2 = (PSF**2).sum(-1)
    # удаление шума из изображения
    iPSF = PSF / (PSF2 + noise)[...,np.newaxis]
    RES = cv2.mulSpectrums(IMG, iPSF, 0)
    # обратное фурье преобразование
    res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
    res = np.roll(res, -kh//2, 0)
    res = np.roll(res, -kw//2, 1)
    write_image(restored_path, res)
