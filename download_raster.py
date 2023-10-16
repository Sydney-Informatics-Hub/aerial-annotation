# -*- coding: utf-8 -*-
import os

import geopandas as gpd

from sa1 import SA1Image


def download(sa1_list, nsw_gdf, output_folder: str):
    for sa1 in sa1_list:
        print(sa1)
        sa1_gdf = nsw_gdf[nsw_gdf["SA1_CODE21"] == sa1]
        sa1_image = SA1Image(sa1_gdf, 21)
        sa1_image.save_as_full_geotiff(output_folder=f"{output_folder}_full")
        sa1_image.save_as_sa1_geotiff(output_folder=f"{output_folder}_sa1_only")


def main():
    shapefile_path = "../data/SA1_2021_AUST_SHP_GDA2020/SA1_2021_AUST_GDA2020.shp"
    gdf = gpd.read_file(shapefile_path)
    print(gdf.crs)

    # getting greater sydney only SA1 polygons
    nsw_gdf = gdf[gdf["GCC_CODE21"] == "1GSYD"]

    nsw_gdf.loc[:, "xmin"] = nsw_gdf["geometry"].bounds["minx"]
    nsw_gdf.loc[:, "ymin"] = nsw_gdf["geometry"].bounds["miny"]
    nsw_gdf.loc[:, "xmax"] = nsw_gdf["geometry"].bounds["maxx"]
    nsw_gdf.loc[:, "ymax"] = nsw_gdf["geometry"].bounds["maxy"]

    nsw_gdf = nsw_gdf[["SA1_CODE21", "xmin", "ymin", "xmax", "ymax", "geometry"]]

    # Define the folder path
    folder_path = "draft"

    for filename in os.listdir(folder_path):
        if filename.startswith("sa1images") and filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r") as file:
                sa1_list = file.read().split("\n")

            download(sa1_list, nsw_gdf=nsw_gdf, output_folder=f"coverage_{filename}")


if __name__ == "__main__":
    main()
