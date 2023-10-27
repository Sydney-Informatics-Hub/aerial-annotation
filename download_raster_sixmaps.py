# -*- coding: utf-8 -*-
import os
import sys

import geopandas as gpd

from sa1 import SA1Image


def download_sa1(sa1_list, nsw_gdf, output_folder: str):
    for sa1 in sa1_list:
        print(sa1)
        sa1_gdf = nsw_gdf[nsw_gdf["SA1_CODE21"] == sa1]
        if not sa1_gdf.empty:
            sa1_image = SA1Image(sa1_gdf, 21)
            sa1_image.save_as_full_geotiff(output_folder=f"{output_folder}_full")
            # sa1_image.save_as_sa1_geotiff(output_folder=f"{output_folder}_sa1_only")


def download_grid(id_list, nsw_gdf, output_folder: str):
    for id in id_list:
        print(id)
        sa1_gdf = nsw_gdf[nsw_gdf["id"] == id]
        if not sa1_gdf.empty:
            sa1_image = SA1Image(sa1_gdf, 21)
            # sa1_image.save_as_full_geotiff(output_folder=f"{output_folder}_full")
            # sa1_image.save_as_sa1_geotiff(output_folder=f"{output_folder}_sa1_only")
            sa1_image.save_as_full_geotiff(
                output_folder=output_folder,
                move_to_folder="~/DATA/sixmaps_raster/greater_sydney_grid",
            )


def main_sa1():
    # shapefile_path = "../data/SA1_2021_AUST_SHP_GDA2020/SA1_2021_AUST_GDA2020.shp"
    # gdf = gpd.read_file(shapefile_path)
    # print(gdf.crs)
    # # getting greater sydney only SA1 polygons
    # nsw_gdf = gdf[gdf["GCC_CODE21"] == "1GSYD"]
    # nsw_gdf.loc[:, "xmin"] = nsw_gdf["geometry"].bounds["minx"]
    # nsw_gdf.loc[:, "ymin"] = nsw_gdf["geometry"].bounds["miny"]
    # nsw_gdf.loc[:, "xmax"] = nsw_gdf["geometry"].bounds["maxx"]
    # nsw_gdf.loc[:, "ymax"] = nsw_gdf["geometry"].bounds["maxy"]
    # nsw_gdf = nsw_gdf[["SA1_CODE21", "xmin", "ymin", "xmax", "ymax", "geometry"]]

    # Don't download rasters that is already downloaded, so find them first and remove from nsw_gdf

    if len(sys.argv) == 1:
        root_directory = "."
    else:
        root_directory = sys.argv[1]

    extracted_sa1 = set()

    for root, dirs, files in os.walk(root_directory):
        for dir in dirs:
            if dir.startswith("coverage"):
                dir_path = os.path.join(root, dir)
                for file in os.listdir(dir_path):
                    extracted_sa1.add(file.split("_")[0])

    extracted_sa1 = list(extracted_sa1)
    # sa1 with building annotations over 10%
    nsw_gdf = gpd.read_file(os.path.join(root_directory, "filtered_nsw_sa1.geojson"))
    nsw_gdf = nsw_gdf[~nsw_gdf["SA1_CODE21"].isin(extracted_sa1)]

    # Define the folder path to sa1 codes and respective percentage of building annotation coverage
    folder_path = "draft"

    for filename in os.listdir(os.path.join(root_directory, folder_path)):
        if filename.startswith("sa1images") and filename.endswith(".txt"):
            file_path = os.path.join(
                os.path.join(root_directory, folder_path), filename
            )

            with open(file_path, "r") as file:
                sa1_list = file.read().split("\n")

            download_sa1(
                sa1_list,
                nsw_gdf=nsw_gdf,
                output_folder=os.path.join(
                    root_directory, f"coverage_{filename.split('.')[0]}"
                ),
            )


def main_id():
    # Don't download rasters that is already downloaded, so find them first and remove from nsw_gdf

    extracted_id = set()

    # for root, dirs, files in os.walk(root_directory):
    #     for files in dirs:
    #         dir_path = os.path.join(root, dir)
    #         for file in os.listdir(dir_path):
    #             extracted_sa1.add(file.split("_")[0])

    # sa1 with building annotations over 10%
    nsw_gdf = gpd.read_file("~/10percent_grid_rectangles.geojson")
    nsw_gdf = nsw_gdf[~nsw_gdf["id"].isin(extracted_id)]
    nsw_gdf.rename(
        columns={"left": "xmin", "right": "xmax", "bottom": "ymin", "top": "ymax"},
        inplace=True,
    )

    id_list = nsw_gdf["id"].tolist()

    download_grid(
        id_list,
        nsw_gdf=nsw_gdf,
        output_folder="./temp/",
    )

    os.system("rm ./temp/")


if __name__ == "__main__":
    main_id()
