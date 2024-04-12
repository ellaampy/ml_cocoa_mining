import fiona
import sys
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, MultiPolygon
from sklearn.cluster import KMeans

#Read input shapefile
input_shape_file = sys.argv[1]
polygons = [shape(record['geometry']) for record in fiona.open(input_shape_file)]

#Extract single polygons for multipolygon instances
cleaned_polygons = []
for p in polygons:
    if isinstance(p, MultiPolygon):
        cleaned_polygons.extend(p.geoms)
    else:
        cleaned_polygons.append(p)

cleaned_polygons = np.array(cleaned_polygons)

# Extract centroid coordinates
centroids = np.array([list(p.centroid.coords)[0] for p in cleaned_polygons])

# Apply KMeans clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=0) #default max_iter = 300
cluster_labels = kmeans.fit_predict(centroids)
#in cluster_labels I have the label for each polygon in cleaned_polygons

# Group polygons by cluster
clustered_polygons = [[] for _ in range(num_clusters)]
for i, label in enumerate(cluster_labels):
    clustered_polygons[label].append(cleaned_polygons[i])

# Write the blocks shapefiles
# Geopandas is best for writing as it automatically creates the .dbf, .shx, .prj files
for i, cp in enumerate(clustered_polygons):
    # Convert the clustered polygons to a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=cp)
    # Define the output shapefile name
    output_shapefile = 'clustered_polygons_' + str(i) + '.shp'
    # Write the GeoDataFrame to a shapefile
    gdf.to_file(output_shapefile)