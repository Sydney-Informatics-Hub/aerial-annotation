# -*- coding: utf-8 -*-
import glob
import logging
import os

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

tqdm.pandas()


def merge_osm_blocks(
    osm_path: str = "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/",
    save: bool = True,
    ignored_files: list = ["merged.geojson", "merged_filtered.geojson"],
):
    """Merge all the OSM files in the osm_path into one GeoDataFrame.

    Args:
        osm_path (str): Path to the OSM files.
        save (bool, optional): Whether to save the merged GeoDataFrame. Defaults to True.
        ignored_files (list, optional): List of files to ignore. Defaults to ["merged.geojson", "merged_filtered.geojson"].

    Returns:
        gpd.GeoDataFrame: The merged GeoDataFrame.
    """
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
        gdf_osm.to_file(
            os.path.join(os.path.dirname(osm_path), "merged.geojson"), driver="GeoJSON"
        )

    return gdf_osm


def filter_osm_columns(
    osm_path: str = "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/",
    columns: str = "/home/sahand/Data/GIS2COCO/osm_columns.csv",
    save: bool = True,
):
    """Filter out the columns we don't need from the OSM data.

    Args:
        osm_path (str, optional): Path to the OSM file. Defaults to "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/".
        columns (str, optional): Path to the columns csv file. Defaults to "/home/sahand/Data/GIS2COCO/osm_columns.csv".
        save (bool, optional): Whether to save the filtered OSM data. Defaults to True.

    Returns:
        gpd.GeoDataFrame: The filtered OSM data.
    """

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
        osm.to_file(
            os.path.join(os.path.dirname(osm_path), "merged_filtered.geojson"),
            driver="GeoJSON",
        )

    return osm


def to_int(x):
    """Convert a string to int.

    Args:
        x (str): The string to convert.

    Returns:
        int: The converted int.
    """
    try:
        return int(x)
    except Exception as e:
        logger.info(f"Skipped converting {x} to int: {e}")
        return None


def cleaner_function(x):
    """Clean the level column in the OSM data.

    Args:
        x (str): The string to clean.

    Returns:
        int: The cleaned int.
    """
    if x == ">1" or x == "1.5" or x == 1.5:
        return 1
    elif x == 0 or x == "0":
        return 1
    elif str(x).lower() == "kiosk":
        return 1
    else:
        return to_int(x)


def osm_level_cleaner(
    osm_path: str = "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/merged_filtered.geojson",
    column: str = "building:levels",
    save: bool = True,
    clean=cleaner_function,
):
    """Clean the level column in the OSM data.

    Args:
        osm_path (str, optional): Path to the OSM file. Defaults to "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/merged_filtered.geojson".
        column (str, optional): The column to clean. Defaults to "building:levels".
        save (bool, optional): Whether to save the cleaned OSM data. Defaults to True.

    Returns:
        gpd.GeoDataFrame: The cleaned OSM data.
    """

    # Read in the OSM data
    if os.path.isdir(osm_path):
        annotations = merge_osm_blocks(osm_path=osm_path, save=False)
    else:
        annotations = gpd.read_file(osm_path)

    # Clean the level column
    annotations[column] = annotations[column].progress_apply(lambda x: clean(x))

    # Save the cleaned OSM data
    if save:
        out_path = os.path.join(os.path.dirname(osm_path), "merged_cleaned.geojson")
        annotations.to_file(out_path, driver="GeoJSON")

    return annotations


def level_interpolation():
    raise NotImplementedError


def level_bracketing(x):
    """Categorise the OSM data based on the number of levels.

    Args:
        x (int): The number of levels.

    Returns:
        str: The level category.
    """
    if x <= 3:
        return "low"
    elif x <= 9:
        return "mid"
    elif x > 9:
        return "high"
    else:
        return None


def osm_level_categorise(
    osm_path: str = "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/merged_filtered.geojson",
    column: str = "building:levels",
    save: bool = True,
    categorise=level_bracketing,
):
    """Categorise the OSM data based on the number of levels.

    Args:
        osm_path (str, optional): Path to the OSM file. Defaults to "/home/sahand/Data/GIS2COCO/osm_building_annotations_by_10_percent_grid/merged_filtered.geojson".
        column (str, optional): The column to categorise. Defaults to "building:levels".
        save (bool, optional): Whether to save the categorised OSM data. Defaults to True.

    Returns:
        gpd.GeoDataFrame: The categorised OSM data.
    """

    # Read in the OSM data
    if os.path.isdir(osm_path):
        annotations = merge_osm_blocks(osm_path=osm_path, save=False)
    else:
        annotations = gpd.read_file(osm_path)

    # Categorise `column` column and add the vategories based on level category: 1-3 | 4-9 | 10+ to a new column of `level_categories`
    annotations["level_categories"] = annotations[column].apply(lambda x: categorise(x))

    # Save the filtered OSM data
    if save:
        annotations.to_file(
            os.path.join(os.path.dirname(osm_path), "merged_categorised.geojson"),
            driver="GeoJSON",
        )

    return annotations


def osm_landuse_concat():
    raise NotImplementedError


osm_level_cleaner()
