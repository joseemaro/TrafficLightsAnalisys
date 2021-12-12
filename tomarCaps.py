from cv2 import cv2
import numpy as np

np.random.seed(0)
import matplotlib.pyplot as plt
import os
from shutil import rmtree
import gluoncv as gcv
from cv2 import cv2
import mxnet as mx
from gluoncv import model_zoo, data, utils


def obtener_frames(video, aux):
    # toma el video a analizar
    # aca se define el video del cual se van a obtener las capturas
    vidcap = cv2.VideoCapture(video)
    count = 0
    success = True
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))

    # cada 10 frames se captura la pantalla
    while success:
        success, image = vidcap.read()
        if fps != 0:
            if count % (10 * fps) == 0:
                aux += 1
                imgPath = "caps_ta/%s.jpg" % str(aux)
                cv2.imwrite(imgPath, image)
                # cv2.imwrite('tpCaps/frame%d.jpg'%count,image)
            count += 1

    s = 'Se lograron tomar capturas de pantalla cada 10 segundos'

    return aux


if __name__ == '__main__':
    '''aux = 0
    # Crea un archivo para guardar el fotograma del video
    carp = 'caps_ma'
    rmtree(carp)
    os.makedirs(carp)
    print('procesando video 1')
    video = 'VIDEOS FINAL SI/ma1.mp4'
    aux = obtener_frames(video, aux)
    print('procesando video 2')
    video = 'VIDEOS FINAL SI/ma2.mp4'
    aux = obtener_frames(video, aux)
    print('procesando video 3')
    video = 'VIDEOS FINAL SI/ma3.mp4'
    aux = obtener_frames(video, aux)
    print('procesando video 4')
    video = 'VIDEOS FINAL SI/ma4.mp4'
    aux = obtener_frames(video, aux)'''

    '''aux = 0
    # Crea un archivo para guardar el fotograma del video
    carp = 'caps_me'
    rmtree(carp)
    os.makedirs(carp)
    print('procesando video 1')
    video = 'VIDEOS FINAL SI/me1.mp4'
    aux = obtener_frames(video, aux)
    print('procesando video 2')
    video = 'VIDEOS FINAL SI/me2.mp4'
    aux = obtener_frames(video, aux)
    print('procesando video 3')
    video = 'VIDEOS FINAL SI/me3.mp4'
    aux = obtener_frames(video, aux)'''

    aux = 0
    # Crea un archivo para guardar el fotograma del video
    carp = 'caps_ta'
    rmtree(carp)
    os.makedirs(carp)
    print('procesando video 1')
    video = 'VIDEOS FINAL SI/ta1.mp4'
    aux = obtener_frames(video, aux)
    print('procesando video 2')
    video = 'VIDEOS FINAL SI/ta2.mp4'
    aux = obtener_frames(video, aux)
    print('procesando video 3')
    video = 'VIDEOS FINAL SI/ta3.mp4'
    aux = obtener_frames(video, aux)
    print('procesando video 4')
    video = 'VIDEOS FINAL SI/ta4.mp4'
    aux = obtener_frames(video, aux)