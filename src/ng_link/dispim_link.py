"""
Dispim link generation. 
Distinct to dispim link: 
    - Deskewing
"""
import numpy as np

from ng_link import NgState, link_utils, xml_parsing

def apply_deskewing(matrix_3x4: np.ndarray, theta: float = 45) -> np.ndarray:
    """
    Compounds deskewing transform to input 3x4 matrix:
        Deskewing @ Input_3x4_Matrix.

    Parameters
    ------------------------
    matrix_3x4: np.ndarray
        3x4 numpy array representing transformation matrix applied to tile.

    theta: float
        Angle of lens during acquisition.

    Returns
    ------------------------
    np.ndarray:
        3x4 numpy array composite transform.

    """

    # Deskewing
    # X vector => XZ direction
    deskew_factor = np.tan(np.deg2rad(theta))
    deskew = np.array([[1, 0, 0], [0, 1, 0], [deskew_factor, 0, 1]])
    matrix_3x4 = deskew @ matrix_3x4

    return matrix_3x4


def generate_dispim_link(
    xml_path: str,
    s3_path: str,
    channels: list[int] = [405, 488, 561, 638],
    max_dr: int = 800,
    opacity: float = 0.5,
    blend: str = "additive",
    deskew_angle: int = 45,
    output_json_path: str = ".",
) -> None:
    """
    Creates an neuroglancer link to visualize
    registration transforms on dispim dataset pre-fusion.

    Parameters
    ------------------------
    xml_path: str
        Path of mounted xml output by BigStitcher.
    s3_path: str
        Path of s3 bucket where dipim dataset is located.
    output_json_path: str
        Local path to write process_output.json file that nueroglancer reads.

    Returns
    ------------------------
    None
    """

    # Gather base channel xml info
    vox_sizes: tuple[float, float, float] = xml_parsing.extract_tile_vox_size(
        xml_path
    )
    tile_paths: dict[int, str] = xml_parsing.extract_tile_paths(xml_path)
    tile_transforms: dict[
        int, list[dict]
    ] = xml_parsing.extract_tile_transforms(xml_path)

    net_transforms: dict[int, np.ndarray] = link_utils.calculate_net_transforms(tile_transforms)

    # Generate input config
    layers = []  # Represent Neuroglancer Tabs
    input_config = {
        "dimensions": {
            "x": {"voxel_size": vox_sizes[0], "unit": "microns"},
            "y": {"voxel_size": vox_sizes[1], "unit": "microns"},
            "z": {"voxel_size": vox_sizes[2], "unit": "microns"},
            "c'": {"voxel_size": 1, "unit": ""},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "layers": layers,
        "showScaleBar": False,
        "showAxisLines": False,
    }

    for channel in channels:
        # Determine color of this layer
        hex_val: int = link_utils.wavelength_to_hex(channel)
        hex_str = f"#{str(hex(hex_val))[2:]}"

        # Init new list of sources for each channel
        sources = []  # Represent Tiles w/in Tabs
        layers.append(
            {
                "type": "image",  # Optional
                "source": sources,
                "channel": 0,  # Optional
                "shaderControls": {
                    "normalized": {"range": [0, max_dr]}
                },  # Optional
                "shader": {
                    "color": hex_str,
                    "emitter": "RGB",
                    "vec": "vec3",
                },
                "visible": True,  # Optional
                "opacity": opacity,
                "name": f"CH_{channel}",
                "blend": blend,
            }
        )

        for tile_id, _ in enumerate(net_transforms):
            net_tf = net_transforms[tile_id]
            t_path = tile_paths[tile_id]

            url = f"{s3_path}/{t_path}"
            final_transform = link_utils.convert_matrix_3x4_to_5x6(net_tf)

            sources.append(
                {"url": url, "transform_matrix": final_transform.tolist()}
            )

    # Generate the link
    neuroglancer_link = NgState(
        input_config=input_config,
        mount_service="s3",
        bucket_path="aind-open-data",
        output_json=output_json_path,
    )
    neuroglancer_link.save_state_as_json()
    print(neuroglancer_link.get_url_link())