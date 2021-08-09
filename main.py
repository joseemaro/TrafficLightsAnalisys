from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox
from numpy.lib.polynomial import poly
import os
from shutil import rmtree


def conteo_autos(path, bool):
    image = cv2.imread(path)
    box, label, count = cv.detect_common_objects(image)
    if bool == 1:
        output = draw_bbox(image, box, label, count)
        plt.imshow(output)
        plt.show()
    return label.count('car')


def obtener_frames():
    # Crea un archivo para guardar el fotograma del video
    rmtree("tpCaps")
    os.makedirs("tpCaps")
    # toma el video a analizar
    # aca se define el video del cual se van a obtener las capturas
    vidcap = cv2.VideoCapture('Ema.mp4')
    count = 0
    success = True
    fps = int(vidcap.get(cv2.CAP_PROP_FPS))
    aux = 0

    # cada 10 frames se captura la pantalla
    while success:
        success, image = vidcap.read()
        if fps != 0:
            if count % (10 * fps) == 0:
                aux += 1
                imgPath = "tpCaps/%s.jpg" % str(aux)
                cv2.imwrite(imgPath, image)
                # cv2.imwrite('tpCaps/frame%d.jpg'%count,image)
            count += 1

    s = 'Se lograron tomar capturas de pantalla cada 10 segundos'

    return s


def recorrer_frames(dire, bool):
    global imagenes
    contenido = os.listdir(dire)
    cant_autos = []
    imagenes = []
    for imagen in contenido:
        if os.path.isfile(os.path.join(dire, imagen)) and imagen.endswith('.jpg'):
            imagenes.append(imagen)
            c = conteo_autos(os.path.join(dire, imagen), bool)
            cant_autos.append(c)
            print("Imagen = " + imagen + " - Cantidad autos= " + str(c))

    return cant_autos


def graficar_histograma(l_cant, l_fotos):
    mean = sum(l_cant)/len(l_cant)
    plt.rcParams['figure.figsize'] = [14, 6]
    x = np.array(range(0, len(l_fotos)))
    y = np.array(l_cant)
    my_xticks = l_fotos
    plt.xticks(x, my_xticks)
    plt.yticks(y, l_cant)
    plt.plot(x, y, color="skyblue")
    plt.xlabel('Imagenes')
    plt.ylabel('Cantidad autos')
    plt.title('Congestion vehicular en semaforos')

    # specifying horizontal line type
    plt.axhline(y=mean, color='r', linestyle='-', label="Promedio de autos")
    plt.legend(("Cantidad de autos", "Promedio de autos"))
    plt.show()


def obtener_nombres(dire):
    contenido = os.listdir(dire)
    imagenes = []
    for imagen in contenido:
        if os.path.isfile(os.path.join(dire, imagen)) and imagen.endswith('.jpg'):
            imagenes.append(imagen)

    return imagenes


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Para obtener los frames de un video definido arriba
    print("Obteniendo capturas del video....")
    #res = obtener_frames()
    # se recorren las imagenes obtenidas
    direc = 'tpCaps'
    see = input('¿Desea ver el analisis de cada imagen? 1-si 2-no')
    see_num = int(see)
    l_cant = recorrer_frames(direc, see_num)
    l_fotos = obtener_nombres(direc)
    graficar_histograma(l_cant, l_fotos)

