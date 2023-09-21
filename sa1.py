# -*- coding: utf-8 -*-
import io
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import mercantile
import numpy as np
import rasterio
import requests
from osmtogeojson import osmtogeojson
from PIL import Image
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from shapely.geometry import shape


class SA1Image:
    base_url = "https://maps.six.nsw.gov.au/arcgis/rest/services/sixmaps/LPI_Imagery_Best/MapServer"

    class Tile:
        def __init__(self, zoom_level: int, row: int, column: int):
            """Initialize a TileImage object.

            Parameters:
            - zoom_level (int): Zoom level of the tile.
            - row (int): Row position of the tile.
            - column (int): Column position of the tile.
            """
            self.zoom_level = zoom_level
            self.row = row
            self.column = column
            self.url = (
                f"{SA1Image.base_url}/tile/{self.zoom_level}/{self.row}/{self.column}"
            )
            self.image_bytes = None
            self.grid_x = None
            self.grid_y = None
            self.size = None

        def download_image(self):
            params = {
                "blank_tile": "false",
            }
            response = requests.get(self.url, params=params)
            if response.status_code == 200:
                image_bytes = response.content
                self.image_bytes = image_bytes
                self.size = Image.open(io.BytesIO(image_bytes)).size
            else:
                print(
                    f"Downloading tile request failed with status code {response.status_code} for {self.url}"
                )

    def __init__(self, sa1_gdf: gpd.GeoDataFrame, zoom_level: int):
        """Initialize an SA1Image object.

        Args:
            sa1_gdf (GeoDataFrame): GeoDataFrame containing SA1 data.
            zoom_level (int): Zoom level for map tiles.
        """
        self.sa1_code = sa1_gdf.iloc[0]["SA1_CODE21"]
        self.polygon = sa1_gdf
        self.sa1_bbox = [
            sa1_gdf.iloc[0]["xmin"],
            sa1_gdf.iloc[0]["ymin"],
            sa1_gdf.iloc[0]["xmax"],
            sa1_gdf.iloc[0]["ymax"],
        ]  # this bbox touches the SA1 polygon edges
        self.zoom_level = zoom_level

        self.tiles = self.get_tiles()
        self.grid_row_count = (
            max(self.tiles, key=lambda x: x.grid_y).grid_y + 1
        )  # calculate the grid coordinates of stitched images
        self.grid_col_count = max(self.tiles, key=lambda x: x.grid_x).grid_x + 1
        self.tiles_bbox = (
            self.calculate_bounding_box()
        )  # this bbox is the bbox for the final image, which should be larger than self.sa1_bbox

        self.buildings = self.get_osm_building_annotations()

    def get_tiles(self) -> list[Tile]:
        tiles = list(mercantile.tiles(*self.sa1_bbox, self.zoom_level))
        tiles_list = []

        for tile in tiles:
            # url = f'{SA1Image.base_url}/tile/{self.zoom_level}/{tile.y}/{tile.x}'
            tile_image = self.Tile(
                zoom_level=self.zoom_level, row=tile.y, column=tile.x
            )
            tiles_list.append(tile_image)

        # calculate where each tile should be placed in the final grid
        smallest_row = min(tiles_list, key=lambda x: x.row).row
        smallest_column = min(tiles_list, key=lambda x: x.column).column

        for tile_image in tiles_list:
            tile_image.grid_x = tile_image.column - smallest_column
            tile_image.grid_y = tile_image.row - smallest_row

        return tiles_list

    def download_tile_images(self) -> None:
        for tile in self.tiles:
            tile.download_image()

    def calculate_bounding_box(self) -> list:  # extent coordinates for matplotlib
        top_left_tile = next(
            (tile for tile in self.tiles if tile.grid_x == 0 and tile.grid_y == 0), None
        )
        bottom_right_tile = max(
            self.tiles, key=lambda tile: (tile.grid_x, tile.grid_y), default=None
        )

        ul_coordinates = mercantile.ul(
            top_left_tile.column, top_left_tile.row, top_left_tile.zoom_level
        )
        top_left_lat, top_left_lng = ul_coordinates.lat, ul_coordinates.lng

        bottom_right_tile = mercantile.bounds(
            bottom_right_tile.column, bottom_right_tile.row, top_left_tile.zoom_level
        )
        bottom_right_lng, bottom_right_lat = (
            bottom_right_tile.east,
            bottom_right_tile.south,
        )

        return [top_left_lng, bottom_right_lng, bottom_right_lat, top_left_lat]

    def stitch_images(self) -> Image.Image:
        if not self.tiles[0].image_bytes:
            self.download_tile_images()

        # Find dimensions of the final image based on the maximum x and y coordinates in the grid
        max_x = max([tile.grid_x for tile in self.tiles])
        max_y = max([tile.grid_y for tile in self.tiles])

        tile_width, tile_height = self.tiles[0].size

        # Calculate the actual dimensions of the final image based on the grid.
        final_width = (max_x + 1) * tile_width
        final_height = (max_y + 1) * tile_height

        # Create an empty image with the calculated dimensions.
        final_image = Image.new("RGB", (final_width, final_height))

        # Paste each tile image onto the final image based on its grid coordinates.
        for tile in self.tiles:
            if tile.image_bytes:
                tile_image = Image.open(io.BytesIO(tile.image_bytes))
                paste_x = tile.grid_x * tile_width
                paste_y = tile.grid_y * tile_height
                final_image.paste(tile_image, (paste_x, paste_y))

        return final_image

    def plot(self):
        self.image = self.stitch_images()

        fig, ax = plt.subplots()

        ax.imshow(self.image, extent=self.tiles_bbox)

        self.polygon.plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2)
        plt.show()

    def get_osm_building_annotations(self):
        """Get building annotations from OSM for the current SA1 area."""
        sa1_bbox = self.sa1_bbox
        # api = overpass.API()
        # res = api.get(query)

        # -33.80627058022732,151.17650985717773,-33.79785455471434,151.18669152259827
        query = f"""
        [out:json];
        (
        way["building"]({sa1_bbox[1]},{sa1_bbox[0]},{sa1_bbox[3]},{sa1_bbox[2]});
        relation["building"]["type"="multipolygon"]({sa1_bbox[1]},{sa1_bbox[0]},{sa1_bbox[3]},{sa1_bbox[2]});
        );
        out;
        >;
        out qt;
        """

        url = "http://overpass-api.de/api/interpreter"
        r = requests.get(url, params={"data": query})

        result = osmtogeojson.process_osm_json(r.json())

        features = []

        for feature in result["features"]:
            geometry = shape(feature["geometry"])
            properties = feature["properties"]
            features.append({"geometry": geometry, **properties})

        if result["features"] == []:
            osm_gdf = gpd.GeoDataFrame(features, geometry=[None])
        else:
            osm_gdf = gpd.GeoDataFrame(features)

        osm_gdf.crs = self.polygon.crs

        # Perform a spatial join to find polygons in 'osm_gdf' that intersect with the SA1 polygon
        buildings_within_sa1 = gpd.sjoin(
            osm_gdf, self.polygon, how="inner", predicate="intersects"
        )
        buildings_within_sa1 = buildings_within_sa1.drop(columns=["index_right"])

        return buildings_within_sa1

    def calculate_annotated_ratio(self):
        self.buildings = self.buildings.to_crs("EPSG:3857")
        self.polygon = self.polygon.to_crs("EPSG:3857")

        buildings_area = self.buildings["geometry"].area.sum()
        sa1_area = self.polygon["geometry"].area.sum()

        return round(buildings_area / sa1_area, 2) * 100  # returning percentage

    def save_osm_buildings_geojson(self) -> None:
        self.buildings.to_file(
            f"buildings_within_sa1_{self.sa1_code}.geojson", driver="GeoJSON"
        )
        print(f"Saved osm buildings to buildings_within_sa1_{self.sa1_code}.geojson")

    def save_as_full_geotiff(
        self, output_folder: str = "test", file_name: str = ""
    ) -> None:
        self.image = self.stitch_images()
        width, height = self.image.size

        # Replace lon1, lon2, lat1, lat2 with the left side, right side, bottom side, top side bbox coordinates of the area
        # top_left_lng, bottom_right_lng, bottom_right_lat, top_left_lat
        lon1, lon2, lat1, lat2 = self.tiles_bbox

        # Calculate the pixel width and pixel height
        pixel_width = (lon2 - lon1) / width
        pixel_height = (lat1 - lat2) / height

        print("pixel_width", pixel_width, "pixel_height", pixel_height)

        crs = rasterio.crs.CRS.from_epsg(7844)
        transform = from_origin(lon1, lat2, pixel_width, -pixel_height)

        image_array = np.array(self.image)

        height, width, _ = image_array.shape

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        if not file_name:
            file_name = f"{self.sa1_code}_full.tif"

        output_filepath = os.path.join(output_folder, file_name)

        with rasterio.open(
            output_filepath,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            crs=crs,
            transform=transform,
            dtype=image_array.dtype,
            count=image_array.shape[2],
        ) as dst:
            for i in range(image_array.shape[2]):
                dst.write(image_array[:, :, i], i + 1)

    def save_as_sa1_geotiff(
        self, output_folder: str = "test", file_name: str = ""
    ) -> None:
        self.image = self.stitch_images()
        width, height = self.image.size

        # Replace lon1, lon2, lat1, lat2 with the left side, right side, bottom side, top side bbox coordinates of the area
        # top_left_lng, bottom_right_lng, bottom_right_lat, top_left_lat
        lon1, lon2, lat1, lat2 = self.tiles_bbox

        # Calculate the pixel width and pixel height
        pixel_width = (lon2 - lon1) / width
        pixel_height = (lat1 - lat2) / height

        crs = rasterio.crs.CRS.from_epsg(7844)
        transform = from_origin(lon1, lat2, pixel_width, -pixel_height)

        image_array = np.array(self.image)

        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        if not file_name:
            file_name = f"{self.sa1_code}_sa1.tif"

        output_filepath = os.path.join(output_folder, file_name)

        # Create a mask for pixels outside the SA1 polygon
        polygon_geom = self.polygon["geometry"].iloc[0]
        geom = transform_geom(
            self.polygon.crs, crs, polygon_geom.__geo_interface__
        )  # Transform polygon geometry to the GeoTIFF CRS
        mask = geometry_mask(
            [geom], out_shape=(height, width), transform=transform, invert=True
        )

        # Apply the mask to make pixels outside the polygon transparent
        image_array[
            ~mask
        ] = 0  # Set pixel values outside the polygon to 0 (fully transparent)

        with rasterio.open(
            output_filepath,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            crs=crs,
            transform=transform,
            dtype=image_array.dtype,
            count=image_array.shape[2],
        ) as dst:
            for i in range(image_array.shape[2]):
                dst.write(image_array[:, :, i], i + 1)
