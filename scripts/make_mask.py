#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
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
    return os.path.isdir(path) and not os.listdir(path)


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
        plt.contour(mask[:, :, 0], [0], colors=[edge_colour])
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
    mask_multiplier=1,
    dtype=np.uint8,
    save_args={},
    return_results=False,
    return_coords=False,
    box_reject=0.85,
    high_thresh=0.36,
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
        elif this_l > high_thresh:
            keep_b.append(this_b)
            keep_l.append(this_l)
            keep_p.append(this_p)
        else:
            print(
                f"rejected box {this_b}, size: {this_area}, max_size: {max_area}, logit: {this_l}"
            )
    if len(keep_b) > 0:
        boxes, logits, phrases = (torch.stack(keep_b), torch.stack(keep_l), keep_p)
    masks = torch.tensor([])
    if len(boxes) > 0:
        masks = self.predict_sam(image_pil, boxes)
        masks = masks.squeeze(1)

    # Create an empty image to store the mask overlays
    mask_overlay = np.zeros_like(
        image_np[..., 0], dtype=dtype
    )  # Adjusted for single channel

    if boxes.nelement() == 0:  # No "object" instances found
        print("No objects found in the image.")
    else:
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


def my_show_anns(
    self,
    figsize=(12, 10),
    axis="off",
    cmap="viridis",
    alpha=0.4,
    add_boxes=True,
    box_color="r",
    box_linewidth=1,
    title=None,
    output=None,
    blend=True,
    **kwargs,
):
    """Show the annotations (objects with random color) on the input image.

    Args:
        figsize (tuple, optional): The figure size. Defaults to (12, 10).
        axis (str, optional): Whether to show the axis. Defaults to "off".
        cmap (str, optional): The colormap for the annotations. Defaults to "viridis".
        alpha (float, optional): The alpha value for the annotations. Defaults to 0.4.
        add_boxes (bool, optional): Whether to show the bounding boxes. Defaults to True.
        box_color (str, optional): The color for the bounding boxes. Defaults to "r".
        box_linewidth (int, optional): The line width for the bounding boxes. Defaults to 1.
        title (str, optional): The title for the image. Defaults to None.
        output (str, optional): The path to the output image. Defaults to None.
        blend (bool, optional): Whether to show the input image. Defaults to True.
        kwargs (dict, optional): Additional arguments for matplotlib.pyplot.savefig().
    """

    import warnings

    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore")

    anns = self.prediction

    if anns is None:
        print("Please run predict() first.")
        return
    elif len(anns) == 0:
        print("No objects found in the image.")
        return

    plt.figure(figsize=figsize)
    plt.imshow(self.image)

    if add_boxes:
        for box, phrases, logit in zip(self.boxes, self.phrases, self.logits):
            # Draw bounding box
            box = box.cpu().numpy()  # Convert the tensor to a numpy array
            rect = patches.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=box_linewidth,
                edgecolor=box_color,
                facecolor="none",
            )
            plt.gca().add_patch(rect)
            if phrases == "":
                phrases = "None"
            lab = f"{phrases}: {logit:.2f}"
            plt.gca().text(box[0], box[3], lab, color="red")

    if "dpi" not in kwargs:
        kwargs["dpi"] = 100

    if "bbox_inches" not in kwargs:
        kwargs["bbox_inches"] = "tight"

    plt.imshow(anns, cmap=cmap, alpha=alpha)

    if title is not None:
        plt.title(title)
    plt.axis(axis)

    if output is not None:
        if blend:
            plt.savefig(output, **kwargs)
        else:
            array_to_image(self.prediction, output, self.source)


def run_model(
    input_images,
    box_threshold=0.23,
    text_threshold=0.24,
    output_dir="masks",
    text_prompt="tree",
    box_reject=0.99,
    high_box_threshold=0.0,
):
    """Generate a LangSAM segment anything geospatial model and predict on the
    input images.

    The parameters `box_threshold` and `test_threshold` control the GroundingDINO
    detection thresholds for the input image.
    See: https://github.com/IDEA-Research/GroundingDINO

    `box_threshold` specifies the sensitivity of object detection in the image.
    Higher values make the model more selective, identifying only the most
    confident object instances. The default value of 0.23 was chosen via visual
    verification of 'tree' detections in OMS aerial imagery at zoom=21.

    `text_threshold` associates the detected objects with the text prompt, higher
    values require stronger probability that a detected object is associated.
    For text_prompt='tree', the value of text_threshold makes no difference to
    the resulting detections so we use the model default value of 0.24.

    Parameters:
    input_images: (str) Path to a directory of GeoTIFF tile images.
    box_threshold: (float) Box threshold for the prediction.
    text_threshold: (float) Text threshold for the prediction.
    output_dir: (str) Output directory for the prediction mask files.
    text_prompt: (str) Text prompt for the model.
    box_reject: (float) Fraction of image area to reject box predictions.
    high_box_threshold: (float) Box threshold for boxes larger than box-reject.
    """

    LangSAM.predict = partialmethod(
        predict_with_box_reject, box_reject=box_reject, high_thresh=high_box_threshold
    )
    LangSAM.show_anns = my_show_anns
    sam = LangSAM()
    sam.predict_batch(
        images=input_images,
        out_dir=output_dir,
        text_prompt=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        mask_multiplier=1,
        dtype="uint8",
        merge=False,
        verbose=True,
    )
    del sam
    gc.collect()


def annotate_trees_batch(
    input_images, output_dir, delete_mask_raster=False, restart=True, **kwargs
):
    """Run annotate_trees on a list of input GeoTIFF images.

    Parameters:
    input_images: (list) List of input GeoTIFF image filenames.
    output_dir: (str) Directory to place output files.
    delete_mask_raster: (bool) Remove the merged raster mask (in favour of the vector GeoJSON)
    restart: (bool) Dont do annotations on files where its already been done
    **kwargs: (dict) Keyword arguments passed to annotate_trees.
    """

    for image in input_images:
        print(f"Annotating {image}")
        image_filename = os.path.basename(image)
        output_basename = os.path.splitext(image_filename)[0]
        if restart and os.path.exists(
            os.path.join(output_dir, output_basename) + ".geojson"
        ):
            continue
        annotate_trees(
            image, output_root=os.path.join(output_dir, output_basename), **kwargs
        )
        if delete_mask_raster:
            os.remove(os.path.join(output_basename, "_mask.tif"))


def merge_mask(tile_files, template, output):
    """Merge the tiles tile_files onto the grid defined by template."""

    # Get metadata and shape of template
    temp_rio = rasterio.open(template)
    temp_meta = temp_rio.meta
    output_shape = (
        temp_meta["height"],
        temp_meta["width"],
    )
    output_mask = np.zeros(output_shape, dtype=np.uint8)
    scratch_mask = np.zeros(output_shape, dtype=np.uint8)
    weight_mask = np.zeros(output_shape, dtype=np.uint8)

    for mask_file in tile_files:
        tile_rio = rasterio.open(mask_file)
        tile_data = tile_rio.read()[0]
        tile_meta = tile_rio.meta
        rep_kwargs = {
            "src_transform": tile_meta["transform"],
            "dst_transform": temp_meta["transform"],
            "src_crs": tile_meta["crs"],
            "dst_crs": temp_meta["crs"],
            "dst_nodata": 0,
            "destination": scratch_mask,
        }
        # Regrid tile onto accum_mask
        rasterio.warp.reproject(tile_data, **rep_kwargs)
        output_mask += scratch_mask
        # sum onto weight mask
        tile_weight = np.ones_like(tile_data)
        rasterio.warp.reproject(tile_weight, **rep_kwargs)
        weight_mask += scratch_mask

    out_mask = np.where(output_mask / weight_mask > 0.5, 255, 0)
    # plt.imshow(weight_mask)
    # plt.show()
    # plt.imshow(output_mask)
    # plt.show()
    # plt.imshow(out_mask)
    # plt.show()
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        width=output_shape[1],
        height=output_shape[0],
        count=1,
        dtype=np.uint8,
        nodata=0,
        transform=temp_meta["transform"],
        crs=temp_meta["crs"],
    ) as dst:
        dst.write(out_mask, indexes=1)


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
    cleanup=False,
    box_reject=0.85,
    reproject=None,
    plot_result=False,
    high_box_threshold=0.0,
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
    high_box_threshold: (float) Box threshold for boxes larger than box-reject.
    """

    # Hard code `text_threshold` for now
    # since it makes no difference for 'tree' class.
    text_threshold = 0.24
    if output_root is None:
        output_root = os.path.splitext(input_image)[0]
    output_png = output_root + ".png"
    output_geojson = output_root + ".geojson"
    output_mask = output_root + "_mask.tif"

    # Ensure the input is reprojected to desired projection
    # This is because the model failes on some projections.
    if reproject is not None:
        _, in_ext = os.path.splitext(input_image)
        rep_image = output_root + "_rep" + in_ext
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
        high_box_threshold=high_box_threshold,
    )

    # Merge tiles into a single output
    mask_dir_list = glob.glob(os.path.join(class_dir, "*.tif"))
    merge_mask(mask_dir_list, input_image, output_mask)

    # Output mask is in merged.tif raster - convert to geojson.
    # merged_mask = os.path.join(class_dir, "merged.tif")
    if os.path.exists(output_mask):
        #        os.rename(merged_mask, output_mask)
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
            nargs="+",
            help="Path to single input TIFF image or directory of TIFF images (if running in batch mode).",
        )
        parser.add_argument(
            "--output-root",
            "-o",
            type=str,
            help="Root filename (or path if in batch mode) of output PNG overlays, and raster/vector masks.",
        )
        parser.add_argument(
            "--box-threshold",
            type=float,
            default=0.23,
            help="Box threshold for GroundingDINO predictions from LangSAM model",
        )
        parser.add_argument(
            "--tile-size",
            type=int,
            default=1500,
            help="Size of tiles in pixels to split images into before annotating",
        )
        parser.add_argument(
            "--tile-overlap",
            type=int,
            default=0,
            help="Number of pixels to overlap the tiles.",
        )
        parser.add_argument(
            "--box-reject",
            type=float,
            default=0.8,
            help="Reject predicted boxes with this fraction of the input tile area.",
        )
        parser.add_argument(
            "--high-box-threshold",
            type=float,
            default=0.0,
            help="Box threshold for boxes larger than box-reject.",
        )
        parser.add_argument(
            "--plot-overlay",
            action="store_true",
            help="Plot the annotation mask onto the input image as a PNG.",
        )
        return parser

    parser = create_parser()
    args = parser.parse_args(args)
    if len(args.image) > 1:
        # Run in batch mode
        annotate_trees_batch(
            args.image,
            args.output_root,
            box_threshold=args.box_threshold,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            box_reject=args.box_reject,
            plot_result=args.plot_overlay,
            high_box_threshold=args.high_box_threshold,
        )
    else:
        annotate_trees(
            args.image[0],
            box_threshold=args.box_threshold,
            output_root=args.output_root,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
            box_reject=args.box_reject,
            plot_result=args.plot_overlay,
            high_box_threshold=args.high_box_threshold,
        )


if __name__ == "__main__":
    main()
