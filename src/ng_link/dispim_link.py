"""
Library for generating dispim link.
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
    base_channel_xml_path: str,
    cross_channel_xml_path: str,
    s3_path: str,
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
    base_channel_xml_path: str
        Path to xml file acquired from tile-to-tile
        registration within the base channel.
        These registrations are reused for
        registering tiles in all other channels.

    cross_channel_xml_path: str
        Path to xml file acquired from channel-to-channel registration.
        These registrations are prepended to each tile registration.

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
        base_channel_xml_path
    )
    tile_paths: dict[int, str] = xml_parsing.extract_tile_paths(
        base_channel_xml_path
    )
    tile_transforms: dict[
        int, list[dict]
    ] = xml_parsing.extract_tile_transforms(base_channel_xml_path)
    intertile_transforms: dict[
        int, np.ndarray
    ] = link_utils.calculate_net_transforms(tile_transforms)
    base_channel: int = link_utils.extract_channel_from_tile_path(
        tile_paths[0]
    )

    # Gather cross channel xml info, only care about transforms
    # Massaging transforms into same format as 'intertile_transforms'.
    channel_paths: dict[int, str] = xml_parsing.extract_tile_paths(
        cross_channel_xml_path
    )
    channels: list[int] = [
        link_utils.extract_channel_from_tile_path(cp)
        for cp in channel_paths.values()
    ]

    channel_transforms = xml_parsing.extract_tile_transforms(
        cross_channel_xml_path
    )
    channel_transforms = {
        anchor_id: tfs[-1] for anchor_id, tfs in channel_transforms.items()
    }
    tmp = {}
    for channel_id, tfm in channel_transforms.items():
        nums = [float(val) for val in tfm["affine"].split(" ")]
        tmp[channel_id] = np.hstack(
            (
                np.array([nums[0::4], nums[1::4], nums[2::4]]),
                np.array(nums[3::4]).reshape(3, 1),
            )
        )
    channel_transforms = tmp
    channel_transforms = {
        ch: ch_tf for ch, ch_tf in zip(channels, channel_transforms.values())
    }

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

    for channel, channel_tf in channel_transforms.items():
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

        for tile_id in range(len(intertile_transforms)):
            # Get base tile path, modify path across channels
            base_t_path = tile_paths[tile_id]
            t_path = base_t_path.replace(f"{base_channel}", f"{channel}")

            # Get net transform
            intertile_tf = intertile_transforms[tile_id]
            i_matrix_3x3 = intertile_tf[:, 0:3]
            i_translation = intertile_tf[:, 3]

            c_matrix_3x3 = channel_tf[:, 0:3]
            c_translation = channel_tf[:, 3]

            net_matrix_3x3 = (
                i_matrix_3x3 @ c_matrix_3x3
            )  # NOTE: Right-multiply
            net_translation = (
                np.linalg.inv(c_matrix_3x3) @ i_translation
            ) + c_translation
            net_tf = np.hstack((net_matrix_3x3, net_translation.reshape(3, 1)))

            net_tf = apply_deskewing(net_tf, deskew_angle)

            # Add (path, transform) source entry
            if s3_path.endswith("/"):
                url = f"{s3_path}{t_path}"
            else:
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
