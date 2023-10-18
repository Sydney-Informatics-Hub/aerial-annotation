#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import shutil
from functools import partialmethod

import cv2
import geojson
import numpy as np
import rasterio
import rioxarray
import torch
from matplotlib import pylab as plt
from PIL import Image
from samgeo import split_raster, tms_to_geotiff
from samgeo.common import raster_to_geojson
from samgeo.text_sam import LangSAM, array_to_image


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


def show_mask(
    image, mask, alpha=0.1, cmap="viridis", edges=True, edge_colour="green", output=None
):
    """Plot a mask overlaid onto an image, with highlighted edges if required.

    Inputs
    ======
    image: an input image array - ideally in colour
    mask: an input mask array - ideally with two values (0=masked, 255=unmasked)

    alpha: transparency of mask when overlaid onto image
    color: color of mask image
    edges: determine the edges of the mask and draw a solid line from these
    edge_colour: colour of the edge highlight
    output: filename to output figure to
    """
    fig = plt.figure(figsize=(20, 20))
    plt.imshow(image)
    plt.axis("off")
    mask_arr = mask[:, :, 0]
    mask_arr = np.ma.masked_where(mask_arr == 0, mask_arr)
    plt.imshow(mask_arr, alpha=alpha, cmap=cmap, vmin=0, vmax=255)
    if edges:
        plt.contour(mask[:, :, 0], [254], colors=[edge_colour])
    if output:
        plt.savefig(output)
    else:
        plt.show()
    plt.clf()
    plt.close(fig)


def my_predict(
    self,
    image,
    text_prompt,
    box_threshold,
    text_threshold,
    output=None,
    mask_multiplier=255,
    dtype=np.uint8,
    save_args={},
    return_results=False,
    return_coords=False,
    box_reject=1.1,
    **kwargs,
):
    """Stolen from predict bu thtis one romoves boxes larger and 0.6 times the
    image area before the sam predict. Run both GroundingDINO and SAM model
    prediction.

    Parameters:
        image (Image): Input PIL Image.
        text_prompt (str): Text prompt for the model.
        box_threshold (float): Box threshold for the prediction.
        text_threshold (float): Text threshold for the prediction.
        output (str, optional): Output path for the prediction. Defaults to None.
        mask_multiplier (int, optional): Mask multiplier for the prediction. Defaults to 255.
        dtype (np.dtype, optional): Data type for the prediction. Defaults to np.uint8.
        save_args (dict, optional): Save arguments for the prediction. Defaults to {}.
        return_results (bool, optional): Whether to return the results. Defaults to False.

    Returns:
        tuple: Tuple containing masks, boxes, phrases, and logits.
    """

    if isinstance(image, str):
        if image.startswith("http"):
            image = download_file(image)

        if not os.path.exists(image):
            raise ValueError(f"Input path {image} does not exist.")

        self.source = image

        # Load the georeferenced image
        with rasterio.open(image) as src:
            image_np = src.read().transpose(
                (1, 2, 0)
            )  # Convert rasterio image to numpy array
            self.transform = src.transform  # Save georeferencing information
            self.crs = src.crs  # Save the Coordinate Reference System
            image_pil = Image.fromarray(
                image_np[:, :, :3]
            )  # Convert numpy array to PIL image, excluding the alpha channel
    else:
        image_pil = image
        image_np = np.array(image_pil)

    self.image = image_pil

    boxes, logits, phrases = self.predict_dino(
        image_pil, text_prompt, box_threshold, text_threshold
    )
    # Maximum area for a box in box_reject
    keep_b, keep_l, keep_p = ([], [], [])
    im_w, im_h = image_pil.size
    max_area = box_reject * im_w * im_h
    for this_b, this_l, this_p in zip(boxes, logits, phrases):
        this_area = (this_b[2] - this_b[0]) * (this_b[3] - this_b[1])
        if this_area < max_area:
            keep_b.append(this_b)
            keep_l.append(this_l)
            keep_p.append(this_p)
        else:
            print(f"rejected box {this_b}, size: {this_area}, max_size: {max_area}")
    if len(keep_b) > 0:
        boxes, logits, phrases = (torch.stack(keep_b), torch.stack(keep_l), keep_p)
    masks = torch.tensor([])
    if len(boxes) > 0:
        masks = self.predict_sam(image_pil, boxes)
        masks = masks.squeeze(1)

    if boxes.nelement() == 0:  # No "object" instances found
        print("No objects found in the image.")
        return
    else:
        # Create an empty image to store the mask overlays
        mask_overlay = np.zeros_like(
            image_np[..., 0], dtype=dtype
        )  # Adjusted for single channel

        for i, (box, mask) in enumerate(zip(boxes, masks)):
            # Convert tensor to numpy array if necessary and ensure it contains integers
            if isinstance(mask, torch.Tensor):
                mask = (
                    mask.cpu().numpy().astype(dtype)
                )  # If mask is on GPU, use .cpu() before .numpy()
            mask_overlay += ((mask > 0) * (i + 1)).astype(
                dtype
            )  # Assign a unique value for each mask

        # Normalize mask_overlay to be in [0, 255]
        mask_overlay = (mask_overlay > 0) * mask_multiplier  # Binary mask in [0, 255]

    if output is not None:
        array_to_image(mask_overlay, output, self.source, dtype=dtype, **save_args)

    self.masks = masks
    self.boxes = boxes
    self.phrases = phrases
    self.logits = logits
    self.prediction = mask_overlay
    # png_anns = os.path.splitext(output)[0]
    # print(f'Saving boxes to: {png_anns}')
    # self.show_anns(output=png_anns)

    if return_results:
        return masks, boxes, phrases, logits

    if return_coords:
        boxlist = []
        for box in self.boxes:
            box = box.cpu().numpy()
            boxlist.append((box[0], box[1]))
        return boxlist


def run_model(
    input_images,
    bt=0.23,
    tt=0.24,
    output_dir="masks",
    text_prompt="tree",
    box_reject=0.99,
):
    # Generate segment anything model and use it to make masks
    LangSAM.predict = partialmethod(my_predict, box_reject=box_reject)
    sam = LangSAM()
    sam.predict_batch(
        images=input_images,
        out_dir=output_dir,
        text_prompt=text_prompt,
        box_threshold=bt,
        text_threshold=tt,
        mask_multiplier=255,
        dtype="uint8",
        merge=True,
        verbose=True,
    )


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
        filename = os.path.join(out_path, p["SA1_CODE21"] + ".tif")
        bbox = [p["xmin"], p["ymin"], p["xmax"], p["ymax"]]
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


def annotate_trees_batch(input_images, output_dir, **kwargs):
    """Run annotate_trees on a list of input .tiff image and collate the output
    into output dir."""

    for image in input_images:
        print(f"Annotating trees in {image}")
        image_filename = os.path.basename(image)
        output_basename = os.path.splitext(image_filename)[0]
        annotate_trees(
            image, output_root=os.path.join(output_dir, output_basename), **kwargs
        )


def annotate_trees(
    input_image,
    b_thresh=0.23,
    output_root=None,
    tile_size=1500,
    text_prompt="tree",
    overwrite=True,
    tile_dir="tiles",
    class_dir="masks",
    cleanup=True,
    box_reject=0.99,
):
    """Split up an image, Run segment anything on it and plot the result."""

    # For debug, can hard code a satellite image bounding box for download from TMS.
    sat_bbox = None
    sat_zoom = 21

    tt = 0.24
    if output_root is None:
        output_root = os.path.splitext(input_image)[0]
    text_prompt = "tree"
    overwrite = True  # Overwrite tiles and masks
    output_png = output_root + ".png"
    output_geojson = output_root + ".geojson"
    output_mask = output_root + "_mask.tif"
    # SAM seems to only work with EPSG:3857 so reproject the input to that before running it.
    reproject = 3857
    # Plot the derived annotation mask on top of the input image
    do_show_mask = True

    # Just for hard-coded debugging for now (will overwrite input image if used.)
    if sat_bbox is not None:
        tms_to_geotiff(
            output=input_image,
            bbox=sat_bbox,
            zoom=sat_zoom,
            source="Satellite",
            overwrite=True,
        )

    # Ensure the input is reprojected to desired projection
    in_name, in_ext = os.path.splitext(input_image)
    rep_image = in_name + "_rep" + in_ext
    in_im = rioxarray.open_rasterio(input_image)
    rep_im = in_im.rio.reproject(reproject)
    rep_im.rio.to_raster(rep_image, compress="DEFLATE", tiled=True)
    image_array = cv2.imread(rep_image)

    print(
        f"Input image {input_image} has dimensions {image_array.shape[0]}x{image_array.shape[1]}"
    )
    # Check if tile_dir exists and get out if it does
    if not overwrite:
        if not is_empty(tile_dir):
            raise IOError(
                f"The spcified tile_dir '{tile_dir}' exists and is not empty."
            )
        if not is_empty(class_dir):
            raise IOError(
                f"The spcified class_dir '{class_dir}' exists and is not empty."
            )
    else:
        shutil.rmtree(tile_dir, ignore_errors=True)
        shutil.rmtree(class_dir, ignore_errors=True)

    # Retile the input image (always square tiles for now)
    split_raster(rep_image, out_dir=tile_dir, tile_size=tile_size, overlap=50)
    tile_dir_list = glob.glob(os.path.join(tile_dir, "*"))
    tile_size = cv2.imread(tile_dir_list[0]).shape
    print(
        f"Split image into {len(tile_dir_list)} tiles of size {tile_size[0]}x{tile_size[1]}"
    )

    # Now we have our tiles, generate the segment anything classification
    run_model(
        tile_dir,
        bt=b_thresh,
        tt=tt,
        output_dir=class_dir,
        text_prompt=text_prompt,
        box_reject=box_reject,
    )

    # Output mask is in merged.tif raster - convert to geojson.
    merged_mask = os.path.join(class_dir, "merged.tif")
    if os.path.exists(merged_mask):
        os.rename(merged_mask, output_mask)
        raster_to_geojson(output_mask, output_geojson)

    # Now a merge file will be in class_dir - lets plot it.
    if do_show_mask:
        if os.path.exists(output_mask):
            merge_mask_array = cv2.imread(output_mask)
        else:
            merge_mask_array = np.zeros(tile_size, dtype=np.uint8)

        show_mask(
            image_array[:, :, ::-1], merge_mask_array, output=output_png, alpha=0.3
        )

    if cleanup:
        # Delete the reprojected image
        os.remove(rep_image)
        # Delete the tiles and class dirs
        shutil.rmtree(tile_dir, ignore_errors=True)
        shutil.rmtree(class_dir, ignore_errors=True)


def main(args=None):
    def create_parser():
        parser = argparse.ArgumentParser(
            description="Detect trees in GeoTiff images using text prompt ant LangSAM model"
        )
        parser.add_argument("image", type=str, help="Path to input image.")
        parser.add_argument(
            "--output_root",
            type=str,
            help="Root filename of output PNG overlays, and raster/vector masks.",
        )
        parser.add_argument(
            "--box_threshold",
            type=float,
            default=0.23,
            help="Box threshold for GorundingDINO predictions from LangSAM model",
        )
        parser.add_argument(
            "--tile_size",
            type=int,
            default=1500,
            help="Size of tiles in pixels to split images into before annotating",
        )
        return parser

    parser = create_parser()
    args = parser.parse_args(args)

    annotate_trees(
        args.image,
        b_thresh=args.box_threshold,
        output_root=args.output_root,
        tile_size=args.tile_size,
    )


if __name__ == "__main__":
    main()
