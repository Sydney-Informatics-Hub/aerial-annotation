# -*- coding: utf-8 -*-
import glob
import logging
import os

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def merge_osm_blocks(
    osm_path: str = "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/",
    save: bool = True,
    ignored_files: list = ["merged.geojson", "merged_filtered.geojson"],
):
    # Read in the buildings from all files in the osm directory
    osm_files = glob.glob(os.path.join(osm_path, "*.geojson"))
    osm_files = [
        file for file in osm_files if os.path.basename(file) not in ignored_files
    ]
    # Merge all the files into one dataframe
    # Initialize a list to hold GeoDataFrames
    gdfs = []
    first_gdf = gpd.read_file(osm_files[0])
    crs = first_gdf.crs

    input(f"crs will be set to {crs}. \nPress Enter to continue...")

    # Read each file into a GeoDataFrame and add it to the list
    for file in tqdm(osm_files):
        try:
            gdf = gpd.read_file(file)
            gdf.geometry = gdf.geometry.to_crs(crs)
            gdfs.append(gdf)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")
            input("Error reading file. Press Enter to continue...")
    # Concatenate all GeoDataFrames in the list into one GeoDataFrame
    osm = pd.concat(gdfs)
    gdf_osm = gpd.GeoDataFrame(osm, crs=crs, geometry=osm.geometry)

    if save:
        gdf_osm.to_file(os.path.join(osm_path, "merged.geojson"), driver="GeoJSON")

    return gdf_osm


def filter_osm_columns(
    osm_path: str = "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/",
    columns: str = "/home/sahand/Data/GIS2COCO/osm_columns.csv",
    save: bool = True,
):

    # Read in the OSM data
    if os.path.isdir(osm_path):
        osm = merge_osm_blocks(osm_path=osm_path, save=False)
    else:
        osm = gpd.read_file(osm_path)

    # Read in the columns to keep
    columns = pd.read_csv(columns).potentially_good.values.tolist()

    # Filter out the columns we don't need
    osm = osm[columns]

    # Save the filtered OSM data
    if save:
        osm.to_file(os.path.join(osm_path, "merged_filtered.geojson"), driver="GeoJSON")

    return osm


def osm_level_cleaner():
    raise NotImplementedError


def level_interpolation():
    raise NotImplementedError


def osm_level_categorise(
    osm_path: str = "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/merged_filtered.geojson",
    column: str = "building:levels",
    save: bool = True,
):
    raise NotImplementedError

    # Read in the OSM data
    if os.path.isdir(osm_path):
        annotations = merge_osm_blocks(osm_path=osm_path, save=False)
    else:
        annotations = gpd.read_file(osm_path)

    # Categorise `column` column and add the vategories based on level category: 1-3 | 4-9 | 10+ to a new column of `level_categories`
    annotations["level_categories"] = annotations[column].apply(
        lambda x: "low" if x <= 3 else ("mid" if x <= 9 else "high")
    )

    # Save the filtered OSM data
    if save:
        annotations.to_file(
            os.path.join(osm_path, "merged_categorised.geojson"), driver="GeoJSON"
        )

    return annotations


def osm_landuse_concat():
    raise NotImplementedError
