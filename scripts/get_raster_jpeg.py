# -*- coding: utf-8 -*-
import os
import time

import geojson
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

with open(INPUT_JSON) as file:
    raw_geojson = geojson.load(file)

geojson_feature_list = raw_geojson["features"]
num_tiles = len(geojson_feature_list)
print(f"{len(geojson_feature_list)} tiles to download.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

wms = WebMapService(SIXMAPS_WMS_URL, version=SIXMAPS_WMS_VERSION)

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
    # st = time.time()
    image_request = wms.getmap(
        bbox=this_bbox, srs="EPSG:3857", layers=["0"], size=SIZE, format="image/jpeg"
    )
    with open(output_file, "wb") as this_image:
        this_image.write(image_request.read())
    # et = time.time()
    # Benchmark
    # print(et - st)
