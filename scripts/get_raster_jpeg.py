# -*- coding: utf-8 -*-
import os
import time

import geojson
import numpy as np
import requests
from dask import compute, delayed
from owslib.wms import WebMapService

# Download a list of JPEG tiles from the NSW Six Maps Web Map Server.
# Tile boundaries are define in the input GeoJSON file


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


def download_tiles(features, output_dir):
    """Download tiles defined in `features` to `output_dir`"""

    SIXMAPS_WMS_URL = "http://maps.six.nsw.gov.au/arcgis/services/public/NSW_Imagery/MapServer/WmsServer"
    SIXMAPS_WMS_VERSION = "1.3.0"
    # Pixel scale is 0.0746455m at Zoom=21
    # GSU_grid.geojson is 300x300m
    SIZE = (4019, 4019)

    wms = WebMapService(SIXMAPS_WMS_URL, version=SIXMAPS_WMS_VERSION, timeout=60)

    for feature in features:
        this_id = feature["properties"]["id"]
        this_bbox = (
            feature["properties"][edge] for edge in ["left", "bottom", "right", "top"]
        )
        output_file = os.path.join(output_dir, str(this_id) + ".jpg")
        # Don't downliad tiles already downloaded.
        if os.path.exists(output_file):
            print(f"File {output_file} exists. Skipping it....")
            continue
        print(f"Download tile to {output_file}")
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


def get_chunk_slices(list_length, num_chunks):
    """Return a list of `num_chunks` slices which roughly split `list_length`
    equally."""
    chunk_indices = np.array_split(np.arange(list_length), num_chunks)
    avg_size = int(np.average([len(chunk) for chunk in chunk_indices]))
    print(
        f"Split {list_length} objects into {num_chunks} x {avg_size} parallel chunks."
    )
    return [
        slice(a[0], a[-1] + 1)
        for a in np.array_split(np.arange(list_length), num_chunks)
    ]


def main():
    INPUT_JSON = "GSU_grid.geojson"
    OUTPUT_DIR = "output_tiles"
    N_THREADS = 5

    with open(INPUT_JSON) as file:
        raw_geojson = geojson.load(file)

    geojson_feature_list = raw_geojson["features"]
    num_tiles = len(geojson_feature_list)
    print(f"{num_tiles} tiles to process....")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Split up geojson_feature_list into N_THREADS roughly equal chunks
    chunk_slices = get_chunk_slices(num_tiles, N_THREADS)
    # Create N_THREADS function calls - one for each chunk slice in the tile list.
    chunked_download = [
        delayed(download_tiles)(geojson_feature_list[part], OUTPUT_DIR)
        for part in chunk_slices
    ]
    # Go get em.
    compute(*chunked_download)


if __name__ == "__main__":
    main()
