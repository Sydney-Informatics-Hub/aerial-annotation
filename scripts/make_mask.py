#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import shutil
from functools import partialmethod

import cv2
import numpy as np
import rasterio
import rioxarray
import torch
from matplotlib import pylab as plt
from PIL import Image
from samgeo import split_raster
from samgeo.common import download_file, raster_to_geojson
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
    image (np.ndarray): An input image array - ideally in colour
    mask (np.ndarray): An input mask array - two values (0=masked, 255=unmasked)

    alpha (float, optional): Transparency of mask when overlaid onto image
    cmap (str, optional): Colourmap to use for mask image
    edges (bool, optional): determine the edges of the mask and draw a solid line from these
    edge_colour (str, optional): colour of the edge highlight
    output (str, optional): filename to output figure to (if None, plot on the screen)
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


def predict_with_box_reject(
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
    """Run both GroundingDINO and SAM model prediction on a single image.

    NOTE: Stolen from LangSAM.predict() but this adds an option to reject boxes larger than
    a given fraction of the image area before the SAM predict step. This is inteded to be
    monkey-patched into the LangSAM model via the following signature:

            from functools import partialmethod
            from samgeo.text_sam import LangSAM

            LangSAM.predict = partialmethod(predict_with_box_reject, box_reject=...)
            sam = LangSAM()
            ...
            result = sam.predict_batch(...)

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
        box_reject (float, optional): Fraction of image area to reject box predictions.

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
    box_threshold=0.23,
    text_threshold=0.24,
    output_dir="masks",
    text_prompt="tree",
    box_reject=0.99,
):
    """Generate a LangSAM segment anything geospatial model and predict on the
    input images.

    Parameters:
    input_images: (str) Path to a directory of GeoTIFF tile images.
    box_threshold: (float) Box threshold for the prediction.
    text_threshold: (float) Text threshold for the prediction.
    output_dir: (str) Output directory for the prediction mask files.
    text_prompt: (str) Text prompt for the model.
    box_reject: (float) Fraction of image area to reject box predictions.
    """

    LangSAM.predict = partialmethod(predict_with_box_reject, box_reject=box_reject)
    sam = LangSAM()
    sam.predict_batch(
        images=input_images,
        out_dir=output_dir,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        mask_multiplier=255,
        dtype="uint8",
        merge=True,
        verbose=True,
    )


def annotate_trees_batch(input_images, output_dir, delete_mask_raster=False, **kwargs):
    """Run annotate_trees on a list of input GeoTIFF images.

    Parameters:
    input_images: (list) List of input GeoTIFF image filenames.
    output_dir: (str) Directory to place output files.
    delete_mask_raster: (bool) Remove the merged raster mask (in favour of the vector GeoJSON)
    **kwargs: (dict) Keyword arguments passed to annotate_trees.
    """

    for image in input_images:
        print(f"Annotating {image}")
        image_filename = os.path.basename(image)
        output_basename = os.path.splitext(image_filename)[0]
        annotate_trees(
            image, output_root=os.path.join(output_dir, output_basename), **kwargs
        )
        if delete_mask_raster:
            os.remove(os.path.join(output_basename, "_mask.tif"))


def annotate_trees(
    input_image,
    box_threshold=0.23,
    output_root=None,
    tile_size=1500,
    tile_overlap=0,
    text_prompt="tree",
    overwrite=True,
    tile_dir="tiles",
    class_dir="masks",
    cleanup=True,
    box_reject=0.95,
    reproject=3857,
    plot_result=False,
):
    """Tile up an image, Run segment anything geospatial on it and plot the
    result.

    Parameters:
    input_image: (str) Filename of input GeoTIFF
    box_threshold: (float) Box threshold for the prediction.
    output_root: (str) Root filename for outputs (None = use CWD and input filename root).
    tile_size: (int) Size of the tiles to subdivide the input into.
    tile_overlap: (int) Number of pixels to overlap the tiles.
    text_prompt: (str) Text input for GroundingDINO detections.
    overwrite: (bool) Allow overwriting of output files if they already exist.
    tile_dir: (str) Location of directory to store tiled images.
    class_dir: (str) Location of directory to store annotation masks for each tile.
    cleanup: (bool) Remove tile and class directories after processing.
    box_reject: (float) Fraction of image area to reject box predictions.
    reproject: (int) EPSG code to reproject input before processing.
    plot_result: (bool) Plot the derived annotations on the input TIFF as a PNG.
    """

    text_threshold = (
        0.24  # Hard coded for jnow - since it makes no difference for 'tree' class.
    )
    if output_root is None:
        output_root = os.path.splitext(input_image)[0]
    output_png = output_root + ".png"
    output_geojson = output_root + ".geojson"
    output_mask = output_root + "_mask.tif"

    # Ensure the input is reprojected to desired projection
    # This is because the model failes on some projections.
    if reproject is not None:
        in_name, in_ext = os.path.splitext(input_image)
        rep_image = in_name + "_rep" + in_ext
        in_im = rioxarray.open_rasterio(input_image)
        rep_im = in_im.rio.reproject(reproject)
        rep_im.rio.to_raster(rep_image, compress="DEFLATE", tiled=True)
    else:
        rep_image = input_image
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
    split_raster(rep_image, out_dir=tile_dir, tile_size=tile_size, overlap=tile_overlap)
    tile_dir_list = glob.glob(os.path.join(tile_dir, "*"))
    tile_size = cv2.imread(tile_dir_list[0]).shape
    print(
        f"Split image into {len(tile_dir_list)} tiles of size {tile_size[0]}x{tile_size[1]}"
    )

    # Now we have our tiles, generate the segment anything classification
    run_model(
        tile_dir,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
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
    if plot_result:
        if os.path.exists(output_mask):
            merge_mask_array = cv2.imread(output_mask)
        else:
            merge_mask_array = np.zeros(tile_size, dtype=np.uint8)

        show_mask(
            image_array[:, :, ::-1], merge_mask_array, output=output_png, alpha=0.3
        )

    if cleanup:
        # Delete the tiles and class dirs
        shutil.rmtree(tile_dir, ignore_errors=True)
        shutil.rmtree(class_dir, ignore_errors=True)

    if reproject is not None:
        # Delete the reprojected image
        os.remove(rep_image)


def main(args=None):
    def create_parser():
        parser = argparse.ArgumentParser(
            description="Detect trees in GeoTIFF images using text prompt and LangSAM model"
        )
        parser.add_argument(
            "image",
            type=str,
            help="Path to single input TIFF image or directory of TIFF images (if running in batch mode).",
        )
        parser.add_argument(
            "--output_root",
            "-o",
            type=str,
            help="Root filename (with path if in batch mode) of output PNG overlays, and raster/vector masks.",
        )
        parser.add_argument(
            "--box_threshold",
            type=float,
            default=0.23,
            help="Box threshold for GroundingDINO predictions from LangSAM model",
        )
        parser.add_argument(
            "--tile_size",
            type=int,
            default=1500,
            help="Size of tiles in pixels to split images into before annotating",
        )
        parser.add_argument(
            "--tile_overlap",
            type=int,
            default=0,
            help="Number of pixels to overlap the tiles.",
        )
        parser.add_argument(
            "--box-reject",
            type=float,
            default=0.95,
            help="Reject predicted boxes with this fraction of the input tile area.",
        )
        return parser

    parser = create_parser()
    args = parser.parse_args(args)
    if os.path.isdir(args.image):
        # Run in batch mode if a directory.
        annotate_trees_batch(
            args.image,
            args.output_root,
            box_threshold=args.box_threshold,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            box_reject=args.box_reject,
        )
    else:
        annotate_trees(
            args.image,
            box_threshold=args.box_threshold,
            output_root=args.output_root,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            box_reject=args.box_reject,
        )


if __name__ == "__main__":
    main()
