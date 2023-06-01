import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Union

import boto3
# import neuroglancer
import pandas as pd
from botocore.exceptions import ClientError

from ng_link import NgState

# from ng_link.ng_layer import generate_precomputed_cells

# IO types
PathLike = Union[str, Path]


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s : %(message)s",
    datefmt="%Y-%m-%d %H:%M",
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler(
        #     f"update_instruments_logs_{CURR_DATE_TIME}.log", "a"
        # ),
    ],
    force=True,
)
logging.disable("DEBUG")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def list_top_level_folders_s3(
    s3_bucket: object, prefix: Optional[str] = ""
) -> List:
    """
    Lists top level folders in a S3 bucket

    Parameters
    -----------
    s3_bucket: object
        botocore object with the connection
        to the s3 bucket
    prefix: Optional[str]
        Prefix path. Default ''

    Returns
    -----------
    List
        List with the top level folders
    """
    top_level_folders = s3_bucket.meta.client.list_objects(
        Bucket=s3_bucket.name, Prefix=prefix, Delimiter="/"
    )
    folders = []
    try:
        for o in top_level_folders.get("CommonPrefixes"):
            folders.append(o.get("Prefix"))
    except:
        logger.error(
            "Error while trying to retrieve folders from provided path"
        )

    return folders


def list_top_level_files_s3(
    s3_bucket: object,
    file_ext: str,
    prefix: Optional[str] = "",
) -> List:
    """
    Lists top level files in a S3 bucket

    Parameters
    -----------
    s3_bucket: object
        botocore object with the connection
        to the s3 bucket
    file_ext: str
        File extension to look for in the folder
    prefix: Optional[str]
        Prefix path. Default ''
    Returns
    -----------
    List
        List with the top level folders
    """
    files = s3_bucket.meta.client.list_objects(
        Bucket=s3_bucket.name, Prefix=prefix, Delimiter="/"
    )

    found_files = []

    try:
        for o in files.get("Contents"):
            if o.get("Key").endswith(file_ext):
                found_files.append(o.get("Key"))
    except:
        logger.error(
            f'Error while trying to retrieve files from provided path {files["Prefix"]}'
        )

    return found_files


def read_json_s3(
    boto_resource: object, json_path: PathLike, bucket_name: str
) -> dict:
    """
    Reads a json content hosted in S3

    Parameters
    ---------------
    boto_resource: object
    Boto3 resource pointing to s3.

    json_path: PathLike
    Path where the json is stored in S3

    bucket_name: str
    Bucket name

    Returns
    ---------------
    dict
        Dictionary with the json content
    """
    obj = boto_resource.Object(bucket_name, json_path)

    try:
        json_content = obj.get()["Body"].read().decode("utf-8")
        json_dict = json.loads(json_content)
    except ClientError as ex:
        if ex.response["Error"]["Code"] == "NoSuchKey":
            json_dict = {}
            logging.error(
                f"An error occurred when trying to read json file from {json_path}"
            )
        else:
            raise

    return json_dict


def generate_regex_expression_stitched():
    """
    Generates the regular expression for
    the stitched smartspim datasets

    Returns
    ---------
    str
        Regular expression that identifies
        stitched smartspim dataset
    """
    date_structure = (
        "(20\d{2}-(\d\d{1})-(\d\d{1}))(_|-)((\d{2})-(\d{2})-(\d{2}))"
    )

    smartspim_id = "SmartSPIM_(\d{7}|\d{6})"
    smartspim_id_regex = "({})".format(smartspim_id)

    smartspim_str = f"{smartspim_id}_{date_structure}_(stitched|processed)_{date_structure}/$"
    smartspim_processed_regex = "({})".format(smartspim_str)

    return smartspim_processed_regex


def get_registered_brains(channel_name: str):
    args = {"bucket_name": "aind-open-data"}

    bucket_name = args["bucket_name"]
    capture_date_regex = r"(20[0-9]{2}-([0-9][0-9]{1})-([0-9][0-9]{1}))"
    capture_time_regex = r"(_(\d{2})-(\d{2})-(\d{2}))"
    smartspim_raw_regex = generate_regex_expression_stitched()

    s3 = boto3.resource("s3")
    s3_bucket = s3.Bucket(bucket_name)

    folders = list_top_level_folders_s3(s3_bucket, prefix="")

    smartspim_folders = [
        folder for folder in folders if re.match(smartspim_raw_regex, folder)
    ]

    total_brains = len(smartspim_folders)
    logger.info(
        f"Smartspim folders in s3: {smartspim_folders} - Total brains: {total_brains}"
    )

    prefix_path = f"processed/CCF_Atlas_Registration/{channel_name}/OMEZarr/"
    look_for_file = "image.zarr"

    datasets_with_ccf_image = {}

    # Iterating over smartspim datasets to place the instrument json
    for smart_folder in smartspim_folders:
        # logger.info(f"Looking for {look_for_file} in {smart_folder}"
        smart_folder = smart_folder.replace("/", "")
        prefix_path_dataset = f"{smart_folder}/{prefix_path}"
        logger.info(f"{smart_folder} -> {prefix_path}")

        folders = list_top_level_folders_s3(
            s3_bucket=s3_bucket, prefix=prefix_path_dataset
        )

        # Look for file
        if len(folders):
            for folder in folders:
                if look_for_file in folder:
                    # The file was found, ending loop
                    datasets_with_ccf_image[smart_folder] = folder.replace(
                        ".zarr/", ".zarr"
                    )

                    break

    logger.info(f"Selected folders are: {len(datasets_with_ccf_image)}\n\n")

    return datasets_with_ccf_image


def main():
    # Creating neuroglancer link
    channel_name = "Ex_488_Em_525"
    brains = get_registered_brains(channel_name)
    # brains_488 = get_registered_brains("Ex_488_Em_525")

    for dataset_name, cff_image_path in brains.items():

        micron_res = 25
        zarr_path = f"s3://aind-open-data/{cff_image_path}"

        ccf_data = {
            "dimensions": {
                # check the order
                "z": {"voxel_size": micron_res, "unit": "microns"},
                "y": {"voxel_size": micron_res, "unit": "microns"},
                "x": {"voxel_size": micron_res, "unit": "microns"},
                "t": {"voxel_size": 0.001, "unit": "seconds"},
            },
            "layers": [
                {
                    "source": zarr_path,
                    "type": "image",
                    "channel": 0,
                    "shaderControls": {
                        "normalized": {"range": [0, 500]}
                    },  # Optional
                    "name": "image_25_um",
                },
                {
                    "type": "segmentation",
                    "source": "precomputed://SmartSPIM_660850_2023-04-03_10-21-14_stitched_2023-04-10_18-28-49/processed/Quantification/Ex_488_Em_525/visualization/ccf_cell_precomputed",
                    "tab": "source",
                    "name": "cell_counting_in_CCF",
                },
            ],
        }

        neuroglancer_link = NgState(
            input_config=ccf_data,
            mount_service="s3",
            bucket_path="aind-open-data",
            output_json="/Users/camilo.laiton/repositories/aind-ng-link/src/ng_link/scripts/ccf_datasets",
            json_name=f"process_output_{dataset_name}_{channel_name}.json",
        )

        neuroglancer_link.save_state_as_json()


if __name__ == "__main__":
    main()
