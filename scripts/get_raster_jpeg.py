# -*- coding: utf-8 -*-
import argparse
import json
import os
import time

import geopandas
import numpy as np
from dask import compute, delayed
from owslib.wms import WebMapService
from shapely.geometry import Polygon

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
        except Exception:
            this_attempt += 1
            if this_attempt > attempts:
                raise


def download_tiles(features, output_dir, tile_size):
    """Download tiles defined in `features` to `output_dir`"""

    SIXMAPS_WMS_URL = "http://maps.six.nsw.gov.au/arcgis/services/public/NSW_Imagery/MapServer/WmsServer"
    SIXMAPS_WMS_VERSION = "1.3.0"

    wms = WebMapService(SIXMAPS_WMS_URL, version=SIXMAPS_WMS_VERSION, timeout=60)
    srs = features.crs.srs

    for _, feature in features.iterrows():
        this_id = feature.id
        this_bbox = [
            getattr(feature, edge) for edge in ["left", "bottom", "right", "top"]
        ]
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
            srs=srs,
            layers=["0"],
            size=tile_size,
            format="image/jpeg",
        )
        et = time.time()
        # Benchmark
        print(et - st)


def get_chunk_slices(list_length, num_chunks):
    """Return a list of `num_chunks` slices which roughly split `list_length`
    equally."""
    # num_chunks must be <= list_length
    num_chunks = min(num_chunks, list_length)
    chunk_indices = np.array_split(np.arange(list_length), num_chunks)
    avg_size = int(np.average([len(chunk) for chunk in chunk_indices]))
    print(
        f"Split {list_length} objects into {num_chunks} x {avg_size} parallel chunks."
    )
    return [
        slice(a[0], a[-1] + 1)
        for a in np.array_split(np.arange(list_length), num_chunks)
    ]


def increase_bbox_extent(bbox, percentage):
    width = bbox["right"] - bbox["left"]
    height = bbox["bottom"] - bbox["top"]
    increase_width = width * (percentage / 100)
    increase_height = height * (percentage / 100)

    bbox["left"] -= increase_width / 2
    bbox["right"] += increase_width / 2
    bbox["top"] -= increase_height / 2
    bbox["bottom"] += increase_height / 2

    return bbox


def create_polygon_from_bbox(bbox):
    return Polygon(
        [
            (bbox["left"], bbox["top"]),
            (bbox["right"], bbox["top"]),
            (bbox["right"], bbox["bottom"]),
            (bbox["left"], bbox["bottom"]),
            (bbox["left"], bbox["top"]),
        ]
    )


def process_geojson(input_geojson, output_geojson, percentage_increase):
    with open(input_geojson, "r") as infile:
        data = json.load(infile)

    features = data["features"]

    for feature in features:
        properties = feature["properties"]
        geometry = feature["geometry"]

        # Extract the bounding box from the properties
        bbox = {
            "left": properties["left"],
            "top": properties["top"],
            "right": properties["right"],
            "bottom": properties["bottom"],
        }

        # Increase the bounding box extent
        new_bbox = increase_bbox_extent(bbox, percentage_increase)

        # Create a new polygon based on the updated bounding box
        new_polygon = create_polygon_from_bbox(new_bbox)

        # Update the geometry coordinates in the feature
        geometry["coordinates"] = [list(new_polygon.exterior.coords)]

    # Save the modified GeoJSON to a new file
    with open(output_geojson, "w") as outfile:
        json.dump(data, outfile)

    return output_geojson


def main(args=None):
    def parse_arguments():
        parser = argparse.ArgumentParser(
            description="Download raster files in JPEG format from sixmaps Web Map Server"
        )
        parser.add_argument(
            "input_json",
            type=str,
            help="Input GeoJSON filename. The script will use the (`left`, `top`, `right`, `bottom`)"
            "fields in `properties` to derive the bounds of the tile in map units.",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="./output_tiles",
            help="Name of output directory for downloaded JPEG tiles."
            "The filenames will be derived from the id field in the input GeoJSON.",
        )
        parser.add_argument(
            "--tile_size",
            type=lambda t: [s.strip() for s in t.split(",")],
            default="4019,4019",
            help="Number of pixels in x,y in the resulting JPEG."
            "NOTE: The default (4019,4019) will download tiles at zoom 21"
            "(resolution 0.07464 m) for tiles of 300x300 m."
            "Decrease this size for smaller tiles or for coarser zoom levels.",
        )
        parser.add_argument(
            "--overlap",
            type=int,
            default=0,
            help="Overlap percentage to add to each tile. Default is 0.",
        )
        parser.add_argument(
            "--nthreads",
            type=int,
            default=8,
            help="Number of simultaneous downloads from the server to perform."
            "The default of 8 was found to work without error or reducing download speed."
            "Higher numbers may cause requests to raise exceptions.",
        )
        return parser.parse_args()

    args = parse_arguments()

    if args.overlap > 0:
        output_geojson = os.path.join(
            os.path.dirname(args.input_json),
            f"enlarged_{os.path.basename(args.input_json)}",
        )
        args.input_json = process_geojson(args.input_json, output_geojson, args.overlap)

    geojson = geopandas.read_file(args.input_json)

    # geojson_feature_list = raw_geojson["features"]
    num_tiles = len(geojson)
    print(f"GeoJSON CRS is {geojson.crs.srs}")
    print(f"{num_tiles} tiles to process....")

    os.makedirs(args.output_dir, exist_ok=True)

    # Split up geojson_feature_list into N_THREADS roughly equal chunks
    chunk_slices = get_chunk_slices(num_tiles, args.nthreads)
    # Create N_THREADS function calls - one for each chunk slice in the tile list.
    chunked_download = [
        delayed(download_tiles)(geojson[part], args.output_dir, args.tile_size)
        for part in chunk_slices
    ]
    # Go get em.
    compute(*chunked_download)


if __name__ == "__main__":
    main()
