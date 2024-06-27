import osmnx as ox
import networkx as nx
import json
import numpy as np
from scipy.optimize import minimize
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog
import matplotlib.pyplot as plt
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Localización de Almacén')
        self.setGeometry(100, 100, 800, 600)
        layout = QVBoxLayout()
        
        self.button_cargar_json = QPushButton('Cargar JSON')
        self.button_cargar_json.clicked.connect(self.cargar_json)
        layout.addWidget(self.button_cargar_json)
        
        self.button_centroide = QPushButton('Calcular Centroide')
        self.button_centroide.clicked.connect(self.calcular_centroide)
        layout.addWidget(self.button_centroide)
        
        self.button_p_median = QPushButton('Calcular p-Median')
        self.button_p_median.clicked.connect(self.calcular_p_median)
        layout.addWidget(self.button_p_median)
        
        self.button_centroide_ponderado = QPushButton('Calcular Centroide Ponderado')
        self.button_centroide_ponderado.clicked.connect(self.calcular_centroide_ponderado)
        layout.addWidget(self.button_centroide_ponderado)
        
        self.button_kruskal = QPushButton('Calcular Árbol de Expansión Mínima (Kruskal)')
        self.button_kruskal.clicked.connect(self.calcular_kruskal)
        layout.addWidget(self.button_kruskal)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        
        self.poligonos = None
        self.G = None
    
    def cargar_json(self):
        try:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self, "Seleccionar archivo JSON", "", "JSON Files (*.json);;All Files (*)", options=options)
            if file_name:
                with open(file_name, 'r') as f:
                    data = json.load(f)
                    self.poligonos = data['features'][0]['geometry']['coordinates'][0]  # Tomar todas las coordenadas del primer polígono
                print(f"Archivo JSON '{file_name}' cargado correctamente.")
        except Exception as e:
            print(f"Error al cargar el archivo JSON: {str(e)}")
    
    def crear_grafo_poligono(self):
        if self.poligonos is None:
            print("Por favor, carga un archivo JSON primero.")
            return None
        
        # Crear un MultiDiGraph vacío con CRS definido
        G = nx.MultiDiGraph(crs="EPSG:4326")
        
        coords = np.array(self.poligonos)
        
        # Agregar nodos al grafo
        for i, coord in enumerate(coords):
            G.add_node(i, x=coord[0], y=coord[1])
        
        # Agregar aristas al grafo (conectando nodos consecutivos)
        num_coords = len(coords)
        for i in range(num_coords):
            G.add_edge(i, (i + 1) % num_coords)
        
        self.G = G
        return G
    
    def calcular_centroide(self):
        try:
            if self.poligonos is None:
                print("Por favor, carga un archivo JSON primero.")
                return
            
            G = self.crear_grafo_poligono()
            
            # Calcular el centroide de las coordenadas del polígono
            coords = np.array(self.poligonos)
            centroid_x = np.mean(coords[:, 0])
            centroid_y = np.mean(coords[:, 1])
            
            # Encontrar el nodo más cercano en el grafo a las coordenadas del centroide
            centroid_node = ox.distance.nearest_nodes(G, centroid_x, centroid_y)
            
            # Mostrar el grafo con el centroide marcado
            fig, ax = ox.plot_graph(G, show=False, close=False)
            ax.scatter(centroid_x, centroid_y, c='red', s=100)
            plt.show()
            
        except Exception as e:
            print(f"Error al calcular el centroide: {str(e)}")
    
    def calcular_p_median(self):
        try:
            if self.poligonos is None:
                print("Por favor, carga un archivo JSON primero.")
                return
            
            G = self.crear_grafo_poligono()
            centroides = [np.mean(self.poligonos, axis=0)]
            
            def distancia_total(ubicacion, puntos_demandas, G):
                return sum(np.linalg.norm(ubicacion - punto) for punto in puntos_demandas)
            
            inicio = np.mean([centroide for centroide in centroides], axis=0)
            resultado = minimize(distancia_total, inicio, args=(centroides, G), method='L-BFGS-B')
            ubicacion_mediana = ox.distance.nearest_nodes(G, resultado.x[0], resultado.x[1])
            
            # Mostrar el grafo con el punto de p-Median marcado
            fig, ax = ox.plot_graph(G, show=False, close=False)
            x, y = G.nodes[ubicacion_mediana]['x'], G.nodes[ubicacion_mediana]['y']
            ax.scatter(x, y, c='blue', s=100)
            plt.show()
            
        except Exception as e:
            print(f"Error al calcular p-Median: {str(e)}")
    
    def calcular_centroide_ponderado(self):
        try:
            if self.poligonos is None:
                print("Por favor, carga un archivo JSON primero.")
                return
            
            G = self.crear_grafo_poligono()
            
            # Supongamos que cada polígono tiene un campo 'population' en 'properties'
            centroides = [np.mean(self.poligonos, axis=0)]
            poblaciones = [1]  # Suponiendo población 1 para todos los centroides
            
            # Calcular centroide ponderado
            sum_poblacion = sum(poblaciones)
            centroide_ponderado = np.average(centroides, axis=0, weights=poblaciones)
            ubicacion_centroide_ponderado = ox.distance.nearest_nodes(G, centroide_ponderado[0], centroide_ponderado[1])
            
            # Mostrar el grafo con el centroide ponderado marcado
            fig, ax = ox.plot_graph(G, show=False, close=False)
            x, y = G.nodes[ubicacion_centroide_ponderado]['x'], G.nodes[ubicacion_centroide_ponderado]['y']
            ax.scatter(x, y, c='green', s=100)
            plt.show()
            
        except Exception as e:
            print(f"Error al calcular el centroide ponderado: {str(e)}")
    
    def calcular_kruskal(self):
        try:
            if self.poligonos is None:
                print("Por favor, carga un archivo JSON primero.")
                return
            
            G = self.crear_grafo_poligono()
            
            # Calcular el árbol de expansión mínima usando el algoritmo de Kruskal
            mst = nx.minimum_spanning_tree(G)
            
            # Mostrar el grafo con el árbol de expansión mínima
            fig, ax = ox.plot_graph(mst, show=False, close=False)
            plt.show()
            
        except Exception as e:
            print(f"Error al calcular el árbol de expansión mínima: {str(e)}")

if __name__ == "__main__":
    try:
        app = QApplication([])
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Error al ejecutar la aplicación: {str(e)}")
