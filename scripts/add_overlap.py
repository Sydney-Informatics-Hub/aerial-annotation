# -*- coding: utf-8 -*-
# import json

# from shapely.geometry import Polygon
# from shapely.ops import unary_union

# # Read GeoJSON file
# with open(
#     "/Users/xluo3503/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)/work/PIPE-3956-aerial-segmentation/aerial-annotation/GSU_grid.geojson",
#     "r",
# ) as file:
#     geojson_data = json.load(file)


# # "left": 16755913.7421, "top": -3988494.662, "right": 16756213.7421, "bottom": -3988794.662, "row_index": 54, "col_index": 0

# # Function to increase the edge of a square tile
# def increase_edge(square_tile, edge_increase_percentage=2.5):
#     # Calculate the buffer distance for each edge
#     buffer_distance = (edge_increase_percentage / 100) * square_tile.length

#     # Buffer the square tile on all sides
#     buffered_tile = square_tile.buffer(buffer_distance, cap_style=2)

#     return buffered_tile


# # Iterate through features and increase edge
# for feature in geojson_data["features"]:
#     coordinates = feature["geometry"]["coordinates"][0]
#     square_tile = Polygon(coordinates)

#     if square_tile.is_empty or not square_tile.is_valid:
#         continue

#     edge_increased_tile = increase_edge(square_tile)

#     # Update the coordinates in the GeoJSON
#     feature["geometry"]["coordinates"] = [list(edge_increased_tile.exterior.coords)]


# # Create a new GeoJSON file with the modified data
# output_filename = "GSU_grid_overlap.geojson"
# with open(output_filename, "w") as output_file:
#     json.dump(geojson_data, output_file, indent=2)

# print(f"Overlap GeoJSON saved to {output_filename}")


# import json

# from shapely.geometry import Polygon


# def increase_bbox_extent(bbox, percentage):
#     width = bbox["right"] - bbox["left"]
#     height = bbox["bottom"] - bbox["top"]
#     increase_width = width * (percentage / 100)
#     increase_height = height * (percentage / 100)

#     bbox["left"] -= increase_width / 2
#     bbox["right"] += increase_width / 2
#     bbox["top"] -= increase_height / 2
#     bbox["bottom"] += increase_height / 2

#     return bbox


# def create_polygon_from_bbox(bbox):
#     return Polygon(
#         [
#             (bbox["left"], bbox["top"]),
#             (bbox["right"], bbox["top"]),
#             (bbox["right"], bbox["bottom"]),
#             (bbox["left"], bbox["bottom"]),
#             (bbox["left"], bbox["top"]),
#         ]
#     )


# def process_geojson(input_geojson, output_geojson, percentage_increase):
#     with open(input_geojson, "r") as infile:
#         data = json.load(infile)

#     features = data["features"]

#     for feature in features:
#         properties = feature["properties"]
#         geometry = feature["geometry"]

#         # Extract the bounding box from the properties
#         bbox = {
#             "left": properties["left"],
#             "top": properties["top"],
#             "right": properties["right"],
#             "bottom": properties["bottom"],
#         }

#         # Increase the bounding box extent
#         new_bbox = increase_bbox_extent(bbox, percentage_increase)

#         # Create a new polygon based on the updated bounding box
#         new_polygon = create_polygon_from_bbox(new_bbox)

#         # Update the geometry coordinates in the feature
#         geometry["coordinates"] = [list(new_polygon.exterior.coords)]

#     # Save the modified GeoJSON to a new file
#     with open(output_geojson, "w") as outfile:
#         json.dump(data, outfile)


# # Example usage
# input_geojson = "GSU_grid.geojson"
# output_geojson = "GSU_grid_enlarged.geojson"
# percentage_increase = 10  # You can change this to the desired percentage increase

# process_geojson(input_geojson, output_geojson, percentage_increase)
