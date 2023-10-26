#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import shutil

import geojson
from samgeo import tms_to_geotiff


def is_empty(path):
    """Check if specified path is a valid and empty dir."""
    if os.path.exists(path) and not os.path.isfile(path):
        # Checking if the directory is empty or not
        if not os.listdir(path):
            return True
        else:
            return False
    else:
        return True


def download_areas_batch(in_geojson, zoom=21, out_path="./tif_images", overwrite=True):
    """Get a list of bounding boxes from a GeoJSON and download tif files from
    tile map server."""
    if overwrite is False and not is_empty(out_path):
        done_files = glob.glob(os.path.join(out_path, "*.tif"))
    else:
        shutil.rmtree(out_path, ignore_errors=True)
        os.mkdir(out_path)
        done_files = []
    # Open the geojson
    with open(in_geojson) as f:
        in_gj = geojson.load(f)
    # Boxes are in 'features', 'properties'
    for feature in in_gj["features"]:
        p = feature["properties"]
        filename = os.path.join(out_path, str(p["id"]) + ".tif")
        bbox = [p["left"], p["top"], p["right"], p["bottom"]]
        if filename in done_files:
            print(f"Skipping {filename}")
            continue
        tms_to_geotiff(
            output=filename,
            bbox=bbox,
            zoom=zoom,
            source="Satellite",
            overwrite=overwrite,
        )
