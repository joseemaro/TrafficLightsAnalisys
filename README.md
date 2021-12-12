# Traffic Lights Analisys

### En este proyecto se busca hacer un analisis de la congestion vehicular en semaforos, para esto se va a  tomar un video sobre autos esperando en un semaforo, del cual se van a obtener capturas del mismo y se van a analizar para saber la cantidad de autos en ese instante en un semaforo y ver si hay congestion.

### En el siguiente link se puede observar el avance del proyecto:

- https://docs.google.com/document/d/15iAf7_jBiXTsRzwt8omJUqgoQBISpvmVW9zeC0Ssspw/edit?usp=sharing

El dataset utilizado para este proyecto es publico y puede encontrarse en:

- https://www.kaggle.com/emasensei/car-detection-in-traffic-lights/code

Una vez que se tiene el video en la carpeta del proyecto, se debe proceder a procesar el mismo, para esto es necesario descomentar la sigueinte linea de la funcion main:

```# res = obtener_frames()```

Otra opcion es hacerlo desde le archivo 'tomarCaps.py'.

Esto es para que se procese este video y las capturas sean almacenadas, una vez realizado esto solo resta ejecutar el codigo con el siguiente comando:

```python main.py```


