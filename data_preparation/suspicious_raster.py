import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio import features
import numpy as np

def rasterize_shapefile(shapefile, attribute):
    # Read the shapefile
    gdf = gpd.read_file(shapefile)
    # Create a new raster with the same extent as the shapefile
    xmin, ymin, xmax, ymax = gdf.total_bounds
    res = 1  # Define the resolution of the raster (adjust as needed)
    transform = from_origin(xmin, ymax, res, res)
    width = int((xmax - xmin) / res)
    height = int((ymax - ymin) / res)
    raster = np.zeros((height, width), dtype=np.uint8)
    # Rasterize the shapefile based on the specified attribute
    shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf[attribute]))
    burned = features.rasterize(shapes, out_shape=raster.shape, transform=transform)
    return burned, transform

def main(shapefile1, shapefile2, output_raster):
    # Rasterize both shapefiles for the 'Target' attribute
    raster1, transform1 = rasterize_shapefile(shapefile1, 'Target')
    raster2, transform2 = rasterize_shapefile(shapefile2, 'Target')
    # Ensure the same dimensions for both rasters
    min_height = min(raster1.shape[0], raster2.shape[0])
    min_width = min(raster1.shape[1], raster2.shape[1])
    raster1 = raster1[:min_height, :min_width]
    raster2 = raster2[:min_height, :min_width]
    # Compute suspicious areas
    suspicious_raster = np.zeros_like(raster2)
    suspicious_raster[(raster1 == 1) & (raster2 == 0)] = 1

    # Write the difference raster
    with rasterio.open(
        output_raster,
        'w',
        driver='GTiff',
        height=suspicious_raster.shape[0],
        width=suspicious_raster.shape[1],
        count=1,
        dtype=np.uint8,
        crs=gpd.read_file(shapefile1).crs,
        transform=transform1,
    ) as dst:
        dst.write(suspicious_raster, 1)

if __name__ == "__main__":
    shapefile1 = '/localhome/zapp_an/Desktop/fasteo/binary_maps/2016/binary_map_cl_0_2016.shp'  # Provide the path to the first shapefile
    shapefile2 = '/localhome/zapp_an/Desktop/fasteo/binary_maps/2022/binary_map_cl_0_2022.shp'  # Provide the path to the second shapefile
    output_raster = '/localhome/zapp_an/Desktop/fasteo/binary_maps/diff_2016-2022/cluster_0/suspicious_raster_cl_0.tif'  # Provide the path for the output raster
    main(shapefile1, shapefile2, output_raster)