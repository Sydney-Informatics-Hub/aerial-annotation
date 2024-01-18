# -*- coding: utf-8 -*-
import os
import time

import geojson
import requests
from owslib.wms import WebMapService

# Download a list of JPEG tiles from the NSW Six Maps Web Map Server.
# Tile boundaries are define in the input GeoJSON file

SIXMAPS_WMS_URL = (
    "http://maps.six.nsw.gov.au/arcgis/services/public/NSW_Imagery/MapServer/WmsServer"
)
SIXMAPS_WMS_VERSION = "1.3.0"
OUTPUT_DIR = "output_tiles"
INPUT_JSON = "GSU_grid.geojson"
# Pixel scale is 0.0746455m at Zoom=21
# GSU_grid.geojson is 300x300m
SIZE = (4019, 4019)


def request_image_from_server(wms_instance, output_file, attempts=3, **kwargs):
    """Try to download image using defined WMS instance with multiple attempts.

    Raise a `ReadTimeout` if the exception is reaised more than
    `attempts` times. kwargs are passed to wms_instance.getmap
    """
    this_attempt = 1
    while this_attempt <= attempts:
        try:
            image_request = wms_instance.getmap(**kwargs)
            with open(output_file, "wb") as this_image:
                this_image.write(image_request.read())
            break
        except requests.exceptions.ReadTimeout:
            this_attempt += 1
            if this_attempt > attempts:
                raise


with open(INPUT_JSON) as file:
    raw_geojson = geojson.load(file)

geojson_feature_list = raw_geojson["features"]
num_tiles = len(geojson_feature_list)
print(f"{len(geojson_feature_list)} tiles to download.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

wms = WebMapService(SIXMAPS_WMS_URL, version=SIXMAPS_WMS_VERSION, timeout=60)

for i, feature in enumerate(geojson_feature_list):
    this_id = feature["properties"]["id"]
    this_bbox = (
        feature["properties"][edge] for edge in ["left", "bottom", "right", "top"]
    )
    output_file = os.path.join(OUTPUT_DIR, str(this_id) + ".jpg")
    # Don't downliad tiles already downloaded.
    if os.path.exists(output_file):
        print(f"Skipping Tile {i + 1}. File {output_file} exists.")
        continue
    print(f"Get tile {i + 1} of {num_tiles} -> {output_file}")
    st = time.time()
    request_image_from_server(
        wms,
        output_file,
        attempts=3,
        bbox=this_bbox,
        srs="EPSG:3857",
        layers=["0"],
        size=SIZE,
        format="image/jpeg",
    )
    et = time.time()
    # Benchmark
    print(et - st)
