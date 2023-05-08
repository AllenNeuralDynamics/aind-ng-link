"""
Utilities for parsing BDV XML
"""
from collections import OrderedDict

import xmltodict


def extract_tile_paths(xml_path: str) -> dict[int, str]:
    """
    Parses BDV xml and outputs map of setup_id -> tile path.

    Parameters
    ------------------------
    xml_path: str
        Path of xml outputted from BigStitcher.

    Returns
    ------------------------
    dict[int, str]:
        Dictionary of tile ids to tile paths.

    """

    view_paths: dict[int, str] = {}
    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())

    for id, zgroup in enumerate(
        data["SpimData"]["SequenceDescription"]["ImageLoader"]["zgroups"][
            "zgroup"
        ]
    ):
        view_paths[int(id)] = zgroup["path"]

    return view_paths


def extract_tile_vox_size(xml_path: str) -> tuple[float, float, float]:
    """
    Parses BDV xml and output 3-ple of voxel sizes: (x, y, z)

    Parameters
    ------------------------
    xml_path: str
        Path of xml outputted by BigStitcher.

    Returns
    ------------------------
    tuple[float, float, float]:
        Tuple containing voxel sizes.

    """

    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())

    first_tile_metadata = data["SpimData"]["SequenceDescription"][
        "ViewSetups"
    ]["ViewSetup"][0]
    vox_sizes: str = first_tile_metadata["voxelSize"]["size"]
    return tuple(float(val) for val in vox_sizes.split(" "))


def extract_tile_transforms(xml_path: str) -> dict[int, list[dict]]:
    """
    Parses BDV xml and outputs map of setup_id -> list of transformations
    Output dictionary maps view number to list of {'@type', 'Name', 'affine'}
    where 'affine' contains the transform as string of 12 floats.

    Matrices are listed in the order of forward execution.

    Parameters
    ------------------------
    xml_path: str
        Path of xml outputted by BigStitcher.

    Returns
    ------------------------
    dict[int, list[dict]]
        Dictionary of tile ids to transform list. List entries described above.

    """

    view_transforms: dict[int, list[dict]] = {}
    with open(xml_path, "r") as file:
        data: OrderedDict = xmltodict.parse(file.read())

    for view_reg in data["SpimData"]["ViewRegistrations"]["ViewRegistration"]:
        tfm_stack = view_reg["ViewTransform"]
        if type(tfm_stack) is not list:
            tfm_stack = [tfm_stack]
        view_transforms[int(view_reg["@setup"])] = tfm_stack

    view_transforms = {
        view: tfs[::-1] for view, tfs in view_transforms.items()
    }

    return view_transforms
