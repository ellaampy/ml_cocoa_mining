import fiona
import sys
import numpy as np
import geopandas as gpd
from shapely.geometry import shape, MultiPolygon
from sklearn.cluster import KMeans

# Read input shapefile
input_shape_file = sys.argv[1]
with fiona.open(input_shape_file) as input_fiona:
    # Read the attribute data along with geometries
    data = [(shape(record['geometry']), record['properties']) for record in input_fiona]
    original_crs = input_fiona.crs

# Extract single polygons for multipolygon instances
cleaned_polygons = []
attributes = []
for geom, attr in data:
    if isinstance(geom, MultiPolygon):
        for sub_geom in geom.geoms:
            #import pdb; pdb.set_trace()
            cleaned_polygons.append(sub_geom)
            attributes.append(attr)
    else:
        cleaned_polygons.append(geom)
        attributes.append(attr)

cleaned_polygons = np.array(cleaned_polygons)

# Extract centroid coordinates -- here we have to manage better the fact that data is different from 'polygons' of the previous script!
centroids = np.array([list(p.centroid.coords)[0] for p in cleaned_polygons])

# Apply KMeans clustering
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=0) #default max_iter = 300
cluster_labels = kmeans.fit_predict(centroids)

# Group polygons by cluster
clustered_polygons = [[] for _ in range(num_clusters)]
clustered_attributes = [[] for _ in range(num_clusters)]
for i, label in enumerate(cluster_labels):
    clustered_polygons[label].append(cleaned_polygons[i])
    clustered_attributes[label].append(attributes[i])

# Write the blocks shapefiles
# Geopandas is best for writing as it automatically creates the .dbf, .shx, .prj files
for i, (cp, ca) in enumerate(zip(clustered_polygons, clustered_attributes)):
    # Convert the clustered polygons to a GeoDataFrame
    gdf = gpd.GeoDataFrame(geometry=cp)
    # Add attribute data
    gdf = gdf.assign(**{key: [dic[key] for dic in ca] for key in ca[0].keys()})
    # Define the output shapefile name
    output_shapefile = 'clustered_polygons_' + str(i) + '.shp'
    # Write the GeoDataFrame to a shapefile, explicitly specifying CRS
    gdf.to_file(output_shapefile, crs=original_crs)