import os
import csv
from qgis.core import *

path_to_root = "E:\\btp\\"


# Loading the centroid layer
path_to_centroid_layer = path_to_root + "Centroid_Layer_1650.gpkg"
# print(os.path.exists(path_to_centroid_layer))
centroid_layer = QgsVectorLayer(path_to_centroid_layer, "Centroid Layer 1650", "ogr")

if not centroid_layer.isValid():
    print("Centroid Layer failed to load!")
else:
    print("Centroid Layer loaded successfully!")

# Loading the highway layer
path_to_highway_layer = path_to_root + "chandigarh_highway.shp"
highway_layer = QgsVectorLayer(path_to_highway_layer, "Highway Layer", "ogr")

if not highway_layer.isValid():
    print("Highway Layer failed to load!")
else:
    print("Highway Layer loaded successfully!")

for field in highway_layer.fields():
    print(field.name(), field.typeName())

# Getting the coordinates of the centroids
features = centroid_layer.getFeatures()
centroid_coordinates = []
for i, feature in enumerate(features):
    g = feature.geometry()
    g.convertToSingleType()
    # print(g.asPoint())
    centroid_coordinates.append(g.asPoint())
    if i == 2:
        print(g.asPoint())

print("---------------------------------------------")

# Getting the coordinates of the highways
features = highway_layer.getFeatures()
highway_coordinates = []
for i, feature in enumerate(features):
    g = feature.geometry()
    g.convertToSingleType()
    highway_coordinates.append(g.asPolyline())
    # TODO: Remove this if statement
    if i == 2:
        print(g.asPolyline())

threshold_distance = 550
def find_server_locations(threshold_distance, centroid_coordinates, highway_coordinates):
    for centroid in centroid_coordinates:
        for highway in highway_coordinates:
            flag = False
            for h_coord in highway:
                d = QgsDistanceArea()
                # d.setEllipsoidalMode(True)
                d.setEllipsoid('WGS84')
                dist = d.measureLine(centroid, h_coord)
                # print(dist)
                # print(QgsUnitTypes.toString(d.lengthUnits()))
                # print(d.measureLine(centroid_coordinates[0], centroid_coordinates[1]))
                # return []
                if dist < threshold_distance:
                    candidate_server_locations.append(centroid)
                    flag = True
                    break
            if flag:
                break
    return candidate_server_locations

def write_to_csv(candidate_server_locations):
    with open(path_to_root + "candidate_server_locations_1650.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for c in candidate_server_locations:
            writer.writerow([c.x(), c.y()])

def load_from_csv(filename):
    candidate_server_locations = []
    with open(path_to_root + filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            candidate_server_locations.append(QgsPointXY(float(row[0]), float(row[1])))
    return candidate_server_locations

# candidate_server_locations = find_server_locations(threshold_distance, centroid_coordinates, highway_coordinates)
# write_to_csv(candidate_server_locations)
candidate_server_locations = load_from_csv("candidate_server_locations_1650.csv")
print(candidate_server_locations)

# Exporting the candidate server locations to a shapefile
def export_to_shapefile(filename, layer_name, point_data):
    path_to_file = path_to_root + filename
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


# path_to_candidate_server_locations_file = path_to_root + "candidate_server_locations2"
# candidate_server_locations_layer = QgsVectorLayer("Point?crs=epsg:4326", "Candidate Server Locations", "memory")
# candidate_server_locations_layer.dataProvider().addAttributes([QgsField("id", QVariant.Int)])
# candidate_server_locations_layer.updateFields()
# candidate_server_locations_layer.startEditing()
# for i, c in enumerate(candidate_server_locations):
#     f = QgsFeature()
#     f.setGeometry(QgsGeometry.fromPointXY(c))
#     f.setAttributes([i])
#     candidate_server_locations_layer.addFeature(f)
# candidate_server_locations_layer.commitChanges()
# QgsVectorFileWriter.writeAsVectorFormatV3(
#     candidate_server_locations_layer,
#     path_to_candidate_server_locations_file,
#     QgsProject.instance().transformContext(),
#     QgsVectorFileWriter.SaveVectorOptions()    
# )
# export_to_shapefile("candidate_server_locations_1650", "Candidate Server Locations", candidate_server_locations)

solution = load_from_csv("S_24_1650.csv")
print(solution)
export_to_shapefile("S_24_1650", "Solution for S=24", solution)
