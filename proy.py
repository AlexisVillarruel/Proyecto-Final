import osmnx as ox
import networkx as nx
import json
import numpy as np
from scipy.optimize import minimize
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
import matplotlib.pyplot as plt

# Cargar datos de los polígonos desde un archivo JSON
with open('TV16final.json') as f:
    poligonos = json.load(f)['features']

# Inicializar el entorno gráfico de PyQt5
app = QApplication([])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Localización de Almacén')
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()
        
        self.button_centroide = QPushButton('Calcular Centroide')
        self.button_centroide.clicked.connect(self.calcular_centroide)
        layout.addWidget(self.button_centroide)
        
        self.button_p_median = QPushButton('Calcular p-Median')
        self.button_p_median.clicked.connect(self.calcular_p_median)
        layout.addWidget(self.button_p_median)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
    
    def obtener_grafo(self):
        # Definir el área de interés (puedes usar coordenadas de los polígonos)
        area = ox.geocode_to_gdf('Nombre de la Ciudad')
        return ox.graph_from_polygon(area.geometry.unary_union, network_type='drive')
    
    def calcular_centroide(self):
        G = self.obtener_grafo()
        centroides = [np.mean(feature['geometry']['coordinates'][0], axis=0) for feature in poligonos]
        ubicaciones_centroide = [ox.nearest_nodes(G, centroide[0], centroide[1]) for centroide in centroides]
        
        fig, ax = ox.plot_graph(G, show=False, close=False)
        for nodo in ubicaciones_centroide:
            x, y = G.nodes[nodo]['x'], G.nodes[nodo]['y']
            ax.scatter(x, y, c='red', s=100)
        plt.show()
    
    def calcular_p_median(self):
        G = self.obtener_grafo()
        centroides = [np.mean(feature['geometry']['coordinates'][0], axis=0) for feature in poligonos]
        
        def distancia_total(ubicacion, puntos_demandas, G):
            return sum(ox.distance.nearest_edges(G, punto[0], punto[1])[2] for punto in puntos_demandas)
        
        inicio = np.mean([centroide for centroide in centroides], axis=0)
        resultado = minimize(distancia_total, inicio, args=(centroides, G), method='L-BFGS-B')
        ubicacion_mediana = ox.nearest_nodes(G, resultado.x[0], resultado.x[1])
        
        fig, ax = ox.plot_graph(G, show=False, close=False)
        x, y = G.nodes[ubicacion_mediana]['x'], G.nodes[ubicacion_mediana]['y']
        ax.scatter(x, y, c='blue', s=100)
        plt.show()

# Crear la ventana principal y ejecutar la aplicación
window = MainWindow()
window.show()
app.exec_()
