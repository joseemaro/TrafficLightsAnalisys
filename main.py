import numpy
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

conf = []
conf_total = []
cont_img = [0]

'''def conteo_autos(path, bool):
    image = cv2.imread(path)
    box, label, count = cv.detect_common_objects(image)
    conf.append(count)
    if bool == 1:
        output = draw_bbox(image, box, label, count)
        plt.imshow(output)
        plt.show()
    return label.count('car')'''


def conteo_autos(path, bool, net):
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    xrgb = mx.nd.array(rgb).astype('uint8')
    rgb_nd, xrgb = gcv.data.transforms.presets.ssd.transform_test(xrgb, short=512)
    ## (3) Interface
    class_IDs, scores, bounding_boxes = net(rgb_nd)
    ## (4) Display
    cont = 0
    confi = 0
    for i in range(len(scores[0])):
        # print(class_IDs.reshape(-1))
        # print(scores.reshape(-1))
        cid = int(class_IDs[0][i].asnumpy())
        cname = net.classes[cid]
        score = float(scores[0][i].asnumpy())
        if score < 0.5:
            break
        # x, y, w, h = bbox = bounding_boxes[0][i].astype(int).asnumpy()
        tag = "{}; {:.4f}".format(cname, score)
        print('Clase= ', cname, '- Score= ', score)
        if cname == 'car':
            cont = cont + 1
            confi = confi + score
            conf_total.append(score)
            # conf.append(score)
        # cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
        # cv2.putText(img, tag, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    '''
    cont_img[0] = cont_img[0]+1
    formato = '_x.jpg'
    path_r = 'C:\\Users\\Emanuel\\Desktop\\PROC. IMAGENES\\final\\results\\' + str(cont_img[0]) + formato
    cv2.imwrite(path_r, img)
    '''
    confi = confi / cont
    conf.append(confi)

    return cont


def save_ax(ax, filename, **kwargs):
    ax.axis("off")
    ax.figure.canvas.draw()
    trans = ax.figure.dpi_scale_trans.inverted()
    bbox = ax.bbox.transformed(trans)
    plt.savefig(filename, dpi="figure", bbox_inches=bbox, **kwargs)
    plt.close()
    ax.axis("on")
    im = plt.imread(filename)
    return im


'''
def conteo_autos(path, bool, net):
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    xrgb = mx.nd.array(rgb).astype('uint8')
    rgb_nd, xrgb = gcv.data.transforms.presets.ssd.transform_test(xrgb, short=512)
    ## (3) Interface
    class_IDs, scores, bounding_boxes = net(rgb_nd)
    ## (4) Display
    cont = 0
    confi = 0
    for i in range(len(scores[0])):
        # print(class_IDs.reshape(-1))
        # print(scores.reshape(-1))
        cid = int(class_IDs[0][i].asnumpy())
        cname = net.classes[cid]
        score = float(scores[0][i].asnumpy())
        if score < 0.5:
            break
        tag = "{}; {:.4f}".format(cname, score)
        print('clase ', cname, 'score', score)
        if cname == 'car':
            cont = cont + 1
            confi = confi + score
            conf_total.append(score)
            # conf.append(score)
    #save img
    cont_img[0] = cont_img[0]+1
    x, img = data.transforms.presets.ssd.load_test(path, short=512)
    class_IDs, scores, bounding_boxes = net(x)
    ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0],
                             class_IDs[0], class_names=net.classes)
    formato = '_x.jpg'
    path_r = 'C:\\Users\\Emanuel\\Desktop\\PROC. IMAGENES\\final\\results\\' + str(cont_img[0]) + formato
    arr = save_ax(ax, path_r)
    confi = confi / cont
    conf.append(confi)
    return cont
'''

'''
def guardar_res(path, net,cont):
    x, img = data.transforms.presets.ssd.load_test(path, short=512)
    class_IDs, scores, bounding_boxes = net(x)
    ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0],
                             class_IDs[0], class_names=net.classes)
    formato = '_x.jpg'
    path_r = 'C:\\Users\\Emanuel\\Desktop\\PROC. IMAGENES\\final\\results\\' + str(cont) + formato
    arr = save_ax(ax, path_r)
    return arr
'''


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
    cont = 0
    #net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
    net = gcv.model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
    print('X------------------------------X')
    print('Aplicando Modelo para deteccion de autos')
    for imagen in contenido:
        if os.path.isfile(os.path.join(dire, imagen)) and imagen.endswith('.jpg'):
            imagenes.append(imagen)
            c = conteo_autos(os.path.join(dire, imagen), bool, net)
            # arr = guardar_res(os.path.join(dire, imagen), net, cont)
            cont = cont + 1
            cant_autos.append(c)
            print("Imagen = " + imagen + " - Cantidad autos= " + str(c))
            print('--------------------------------')

    return cant_autos


def save_res(cont, dire):
    net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)
    contenido = os.listdir(dire)
    print('X-------------------------------X')
    print('Guardando Resultados de la Deteccion en carpeta results...')
    for imagen in contenido:
        cont_img[0] = cont_img[0] + 1
        cont = cont_img[0]
        if os.path.isfile(os.path.join(dire, imagen)) and imagen.endswith('.jpg'):
            path = os.path.join(dire, imagen)
            x, img = data.transforms.presets.ssd.load_test(path, short=512)
            class_IDs, scores, bounding_boxes = net(x)
            ax = utils.viz.plot_bbox(img, bounding_boxes[0], scores[0],
                                     class_IDs[0], class_names=net.classes)
            formato = '_x.jpg'
            path_r = 'results\\' + str(cont) + formato
            arr = save_ax(ax, path_r)

    print('Proceso Finalizado.')


def graficar_histograma(l_cant, l_fotos, di):
    mean = sum(l_cant) / len(l_cant)
    plt.rcParams['figure.figsize'] = [14, 6]
    x = np.array(range(0, len(l_fotos)))
    y = np.array(l_cant)
    my_xticks = l_fotos
    #plt.xticks(x, my_xticks)
    #plt.yticks(y, l_cant)
    plt.plot(x, y, color="skyblue")
    plt.xlabel('Imagenes')
    plt.ylabel('Cantidad autos')
    plt.title('Congestion vehicular en semaforos ' + di)
    # specifying horizontal line type
    plt.axhline(y=mean, color='r', linestyle='-', label="Promedio de autos")
    plt.legend(("Cantidad de autos", "Promedio de autos"))
    plt.show()


def graficar_histograma_confianza(l_cant, di):
    mean = sum(l_cant) / len(l_cant)
    plt.rcParams['figure.figsize'] = [14, 6]
    x = np.array(range(0, len(l_cant)))
    y = np.array(l_cant)
    # my_xticks = l_fotos
    # plt.xticks(x, my_xticks)
    # plt.yticks(y, l_cant)
    plt.plot(x, y, color="skyblue")
    plt.xlabel('Autos')
    plt.ylabel('Confianza')
    plt.title('Confianza de la prediccion de cada auto ' + di)
    # specifying horizontal line type
    plt.axhline(y=mean, color='r', linestyle='-', label="Promedio de confianza")
    plt.legend(("Confianza", "Promedio de confianza"))
    plt.show()


def obtener_nombres(dire):
    contenido = os.listdir(dire)
    imagenes = []
    for imagen in contenido:
        if os.path.isfile(os.path.join(dire, imagen)) and imagen.endswith('.jpg'):
            imagenes.append(imagen)
    return imagenes


'''
def graficar_confianza(l_fotos):
    prom_imagen = []
    for lis in conf:
        #cant de autos y valor de confianza de cada uno
        values = 0
        cant = 0
        prom = 0
        for li in lis:
            values += li
            cant = cant + 1
        #calculo el valor promedio de confianza para esa imagen
        prom = values/cant
        prom_imagen.append(prom)

    #limito los numeros periodicos
    aux = []
    for i in prom_imagen:
        aux.append(round(i, 4))

    prom_imagen = aux

    plt.rcParams['figure.figsize'] = [12, 6]
    x = np.array(range(0, len(l_fotos)))
    y = np.array(prom_imagen)
    my_xticks = l_fotos
    plt.xticks(x, my_xticks)
    plt.yticks(y, prom_imagen)
    plt.plot(x, y, color="skyblue")
    plt.xlabel('Imagenes')
    plt.ylabel('Confianza Promedio', labelpad=20)
    plt.title('Confianza Promedio en cada imagen')

    # specifying horizontal line type
    plt.axhline(y=np.mean(prom_imagen), color='r', linestyle='-', label="Promedio de confianza")
    plt.legend(("Confianza", "Promedio de confianza"))
    plt.show()'''


def graficar_confianza(l_fotos, di):
    prom_imagen = []
    values = 0
    cant = 0
    prom = 0
    for lis in conf:
        values += lis
        cant = cant + 1
        # calculo el valor promedio de confianza para esa imagen
        prom = values / cant
        prom_imagen.append(prom)
    # limito los numeros periodicos
    aux = []
    for i in prom_imagen:
        aux.append(round(i, 4))
    prom_imagen = aux
    plt.rcParams['figure.figsize'] = [12, 6]
    x = np.array(range(0, len(l_fotos)))
    y = np.array(prom_imagen)
    my_xticks = l_fotos
    #plt.xticks(x, my_xticks)
    # plt.yticks(y, prom_imagen)
    plt.plot(x, y, color="skyblue")
    plt.xlabel('Imagenes')
    plt.ylabel('Confianza Promedio', labelpad=20)
    plt.title('Confianza Promedio en cada imagen ' + di)
    # specifying horizontal line type
    plt.axhline(y=np.mean(prom_imagen), color='r', linestyle='-', label="Promedio de confianza")
    plt.legend(("Confianza", "Promedio de confianza"))
    plt.show()


def graficar_barras_conf(direc):
    ## Declaramos valores para el eje x
    eje_x = ['1-0.99', '0.98-0.95', '0.94-0.9', '0.89-08', '0.79-0.7', '0.69-0']
    ## Declaramos valores para el eje y
    eje_y = [0, 0, 0, 0, 0, 0]

    ## Generar los bins
    for i in conf_total:
        if i >= 0.99:
            eje_y[0] = eje_y[0] + 1
        if 0.95 <= i <= 0.98:
            eje_y[1] = eje_y[1] + 1
        if 0.90 <= i <= 0.94:
            eje_y[2] = eje_y[2] + 1
        if 0.89 <= i <= 0.8:
            eje_y[3] = eje_y[3] + 1
        if 0.79 <= i <= 0.70:
            eje_y[4] = eje_y[4] + 1
        if 0.69 <= i <= 0:
            eje_y[5] = eje_y[5] + 1
    ## Creamos Gráfica
    plt.bar(eje_x, eje_y)
    ## Legenda en el eje y
    plt.ylabel('Cantidad de autos')
    ## Legenda en el eje x
    plt.xlabel('Confianza')
    ## Título de Gráfica
    plt.title('Grafico de barras de confianza ' + direc)
    ## Mostramos Gráfica
    plt.show()


def graficar_barras_cant(cant_autos_h, direcciones):
    ## Creamos Gráfica
    plt.bar(direcciones, cant_autos_h)
    ## Legenda en el eje y
    plt.ylabel('Cantidad de autos')
    ## Legenda en el eje x
    plt.xlabel('Horario')
    ## Título de Gráfica
    plt.title('Cantidad de autos en cada horario')
    ## Mostramos Gráfica
    plt.show()


def graficar_barras_conf_h(conf_h_prom, direcciones):
    ## Creamos Gráfica
    plt.bar(direcciones, conf_h_prom)
    ## Legenda en el eje y
    plt.ylabel('Confianza Promedio')
    ## Legenda en el eje x
    plt.xlabel('Horario')
    ## Título de Gráfica
    plt.title('Promedio de confianza en cada horario')
    ## Mostramos Gráfica
    plt.show()


if __name__ == '__main__':
    # Para obtener los frames de un video definido arriba
    print("Obteniendo capturas del video....")
    # res = obtener_frames()
    conf_h_prom = [0, 0, 0]
    cant_autos_h = [0, 0, 0]

    cont_h = 0
    # se recorren las imagenes obtenidas
    direcciones = ['caps_ma', 'caps_me', 'caps_ta']
    for i in direcciones:
        conf = []
        conf_total = []
        direc = i
        # see = input('¿Desea ver el analisis de cada imagen? 1-si 2-no')
        see = 2
        see_num = int(see)
        l_cant = recorrer_frames(direc, see_num)

        suma = 0
        for valor in l_cant:
            suma = suma + valor

        cant_autos_h[cont_h] = cant_autos_h[cont_h] + suma
        conf_h_prom[cont_h] = conf_h_prom[cont_h] + numpy.mean(conf_total)

        # obtiene nombre de cada foto
        l_fotos = obtener_nombres(direc)

        print('X---------------X')
        print('Graficando....')
        graficar_confianza(l_fotos, direcciones[cont_h])
        graficar_histograma_confianza(conf_total, direcciones[cont_h])
        graficar_barras_conf(direcciones[cont_h])
        graficar_histograma(l_cant, l_fotos, direcciones[cont_h])
        print('X---------------X')
        # save_res(cont_img[0], direc)
        cont_h = cont_h + 1

    graficar_barras_cant(cant_autos_h, direcciones)
    graficar_barras_conf_h(conf_h_prom, direcciones)
    print(conf_h_prom)
    print('fin.')
