import os
import csv
import numpy as np
# from utils import load_from_csv, write_to_csv, export_to_shapefile, calculate_K, PATH_TO_ROOT, PATH_TO_CSV
from qgis.core import *
import geopandas as gpd
import shapely

PATH_TO_ROOT = "E:\\btp"
PATH_TO_CSV = os.path.join(PATH_TO_ROOT, "csv")
K_MAX = 4
threshold_distance = 550

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
    path_to_file = os.path.join(PATH_TO_ROOT, filename)
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

# Loading the centroid layer
# path_to_centroid_layer = os.path.join(PATH_TO_ROOT, "Centroid_Layer_1650.gpkg")
# print(os.path.exists(path_to_centroid_layer))
# centroid_layer = QgsVectorLayer(path_to_centroid_layer, "Centroid Layer 1650", "ogr")

# if not centroid_layer.isValid():
#     print("Centroid Layer failed to load!")
# else:
#     print("Centroid Layer loaded successfully!")

# Loading the highway layer
# path_to_highway_layer = os.path.join(PATH_TO_ROOT, "chandigarh_highway.shp")
# highway_layer = QgsVectorLayer(path_to_highway_layer, "Highway Layer", "ogr")

# if not highway_layer.isValid():
#     print("Highway Layer failed to load!")
# else:
#     print("Highway Layer loaded successfully!")

# for field in highway_layer.fields():
#     print(field.name(), field.typeName())

# Getting the coordinates of the centroids
# features = centroid_layer.getFeatures()
# centroid_coordinates = []
# for i, feature in enumerate(features):
#     g = feature.geometry()
#     g.convertToSingleType()
#     # print(g.asPoint())
#     centroid_coordinates.append(g.asPoint())
#     if i == 2:
#         print(g.asPoint())

print("----------------------------------------------------------------------")

# Getting the coordinates of the highways
# features = highway_layer.getFeatures()
# highway_coordinates = []
# for i, feature in enumerate(features):
#     g = feature.geometry()
#     g.convertToSingleType()
#     highway_coordinates.append(g.asPolyline())
#     # TODO: Remove this if statement
#     if i == 2:
#         print(g.asPolyline())

def find_server_locations(threshold_distance, centroid_coordinates, highway_coordinates):
    candidate_server_locations = []
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



# candidate_server_locations = find_server_locations(threshold_distance, centroid_coordinates, highway_coordinates)
# write_to_csv(candidate_server_locations, 1650)
# candidate_server_locations = load_from_csv("candidate_server_locations.csv")
# print(candidate_server_locations)




# path_to_candidate_server_locations_file = PATH_TO_ROOT + "candidate_server_locations2"
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

solution = load_from_csv("S_45_1650_K.csv")
print(solution)
export_to_shapefile("S_45_1650_K", "Solution for S=45", solution)

def robust_scaling(data):
    median = np.median(data)
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    scaled_data = (data - median) / iqr
    k_step = (np.max(scaled_data) - np.min(scaled_data)) / (K_MAX - 1)
    min_val = np.min(scaled_data)
    k_values = list(map(lambda x: int((x - min_val) / k_step) + 1, scaled_data))
    return k_values


# Calculating the K value for each candidate server location
def calculate_K(server_location_filename, threshold_distance):
    # Load the wards geojson file
    path_to_wards = os.path.join(PATH_TO_ROOT, "wards.geojson")
    print(path_to_wards)
    try:
        wards = gpd.read_file(path_to_wards)
        wards = wards.to_crs("EPSG:4326")
        wards = wards[wards["Ward_name"].str.isdigit()]
    except FileNotFoundError:
        print("File not found!")

    # Load the candidate server locations
    path_to_server_locations = os.path.join(PATH_TO_CSV, server_location_filename)
    server_locations = load_from_csv(path_to_server_locations)
    
    # define the range of possible k values
    population: dict[str, int] = {}
    for ward_name, p in zip(wards["Ward_name"], wards["Population"]):
        if str(ward_name).isdigit() and str(p).isdigit():
            population[str(ward_name)] = int(p)
    k_values = robust_scaling(list(population.values()))
    print(k_values)

    # add a new column in the wards dataframe for k_values
    wards["K"] = np.NaN
    ward_K = dict(zip(population.keys(), k_values))
    for ward_name in wards["Ward_name"]:
        if str(ward_name) in population.keys():
            wards.loc[wards["Ward_name"] == ward_name, "K"] = ward_K[str(ward_name)]
    print(wards)

    # Calculating the K value for each candidate server location
    for i, server_location in enumerate(server_locations):
        server = shapely.geometry.Point(server_location.x(), server_location.y())
        within = wards["geometry"].apply(lambda x: server.within(shapely.geometry.Polygon(x)))
        if within.any():
            ward = wards[within]
            server_locations[i] = (server_location, min(K_MAX, ward["K"].values[0] + 1))
        else:
            # calculate the distance between the server location and the ward
            distances = wards["geometry"].apply(lambda x: server.distance(shapely.geometry.Polygon(x)))
            min_dist, idx = distances.min(), distances.idxmin()
            if min_dist <= threshold_distance:
                server_locations[i] = (server_location, wards.loc[idx, "K"])
            else:
                server_locations[i] = (server_location, 1)
        # for j, ward in wards.iterrows():
        #     ward_polygon = shapely.geometry.Polygon(ward["geometry"])
        #     if server.within(ward_polygon):
        #         server_locations[i] = (server_location, ward["K"])
        #         break
        #     else:
                # calculate the distance between the server location and the ward
    return server_locations

# server_locations_K = calculate_K("candidate_server_locations_1650.csv", 500)
# print(server_locations_K)
# Export to a csv file
# with open(os.path.join(PATH_TO_CSV, "server_locations_1650_K.csv"), "w", newline="") as f:
#     writer = csv.writer(f)
#     for c, k in server_locations_K:
#         writer.writerow([c.x(), c.y(), k])
#     print("Done!")
# Export to a shape file
# slk_layer = QgsVectorLayer("Point?crs=epsg:4326", "Server Locations K Values", "memory")
# slk_layer.dataProvider().addAttributes([QgsField("K", QVariant.Int)])
# slk_layer.updateFields()
# slk_layer.startEditing()
# for i, (c, k) in enumerate(server_locations_K):
#     print(i,c,k)
#     f = QgsFeature()
#     f.setGeometry(QgsGeometry.fromPointXY(c))
#     f.setAttributes([k])
#     slk_layer.addFeature(f)
# slk_layer.commitChanges()
# print(slk_layer.countSymbolFeatures())
# print(slk_layer.featureCount())
# QgsVectorFileWriter.writeAsVectorFormatV3(
#     slk_layer,
#     os.path.join(PATH_TO_ROOT, "server_locations_1650_K"),
#     QgsProject.instance().transformContext(),
#     QgsVectorFileWriter.SaveVectorOptions()
# )
