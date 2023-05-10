import geopandas as gpd
from qgis.core import *
#from qgis.PyQt.QtCore import QVariant
import csv
import os
import sys

# sys.path.extend(['D:/QGIS/apps/qgis/./python', 'C:/Users/Kartik/AppData/Roaming/QGIS/QGIS3\\profiles\\default/python', 'C:/Users/Kartik/AppData/Roaming/QGIS/QGIS3\\profiles\\default/python/plugins', 'D:/QGIS/apps/qgis/./python/plugins', 'D:\\QGIS\\bin\\python39.zip', 'D:\\QGIS\\apps\\Python39\\DLLs', 'D:\\QGIS\\apps\\Python39\\lib', 'D:\\QGIS\\bin', 'C:\\Users\\Kartik\\AppData\\Roaming\\Python\\Python39\\site-packages', 'D:\\QGIS\\apps\\Python39', 'D:\\QGIS\\apps\\Python39\\lib\\site-packages', 'D:\\QGIS\\apps\\Python39\\lib\\site-packages\\win32', 'D:\\QGIS\\apps\\Python39\\lib\\site-packages\\win32\\lib', 'D:\\QGIS\\apps\\Python39\\lib\\site-packages\\Pythonwin', 'C:/Users/Kartik/AppData/Roaming/QGIS/QGIS3\\profiles\\default/python', 'E:/btp', 'C:\\Users\\Kartik\\AppData\\Roaming\\QGIS\\QGIS3\\profiles\\default\\python\\plugins'])

PATH_TO_ROOT = "E:\\btp"
PATH_TO_CSV = os.path.join(PATH_TO_ROOT, "csv")

def load_QgsVectorLayer(path_to_file, layer_name):
    layer = QgsVectorLayer(path_to_file, layer_name, "ogr")
    if not layer.isValid():
        print("Layer failed to load!")
    else:
        print("Layer loaded successfully!")
    return layer

def write_to_csv(candidate_server_locations, grid_size):
    with open(os.path.join(PATH_TO_CSV, f"candidate_server_locations_{grid_size}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        for c in candidate_server_locations:
            writer.writerow([c.x(), c.y()])

def load_from_csv(filename):
    candidate_server_locations = []
    with open(os.path.join(PATH_TO_ROOT, filename), "r") as f:
        reader = csv.reader(f)
        for row in reader:
            candidate_server_locations.append(QgsPointXY(float(row[0]), float(row[1])))
    return candidate_server_locations

# Exporting the candidate server locations to a shapefile
def export_to_shapefile(filename, layer_name, point_data):
    path_to_file = os.path.join(PATH_TO_CSV, filename)
    layer = QgsVectorLayer("Point?crs=epsg:4326", layer_name, "memory")
    layer.dataProvider().addAttributes([QgsField("id", QVariant.Int)])
    layer.updateFields()
    layer.startEditing()
    for i, c in enumerate(point_data):
        f = QgsFeature()
        f.setGeometry(QgsGeometry.fromPointXY(c))
        f.setAttributes([i])
        layer.addFeature(f)
    layer.commitChanges()
    QgsVectorFileWriter.writeAsVectorFormatV3(
        layer,
        path_to_file,
        QgsProject.instance().transformContext(),
        QgsVectorFileWriter.SaveVectorOptions()
    )

# Calculating the K value for each candidate server location
def calculate_K(server_location_filename, threshold_distance):
    # Load the wards geojson file
    path_to_wards = os.path.join(PATH_TO_ROOT, "wards.geojson")
    print(path_to_wards)
    wards = gpd.read_file(path_to_wards)

    # Load the candidate server locations
    path_to_server_locations = os.path.join(PATH_TO_CSV, server_location_filename)
    server_locations = load_QgsVectorLayer(path_to_server_locations, "Server Locations")

    # Getting the coordinates of the centroids
    features = server_locations.getFeatures()
    server_coordinates = []
    for i, feature in enumerate(features):
        g = feature.geometry()
        g.convertToSingleType()
        # print(g.asPoint())
        server_coordinates.append(g.asPoint())
        if i == 2:
            print(g.asPoint())

    # Calculating the K value for each candidate server location
    print(wards)
