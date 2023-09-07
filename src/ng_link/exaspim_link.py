"""
Exaspim link generation. 
Distinct to exaspim link: 
    - Remove zattrs position
"""
import json
import numpy as np
from pathlib import Path

from ng_link import NgState, link_utils, xml_parsing

def get_zattrs_positions(dataset_path: str) -> dict[str, np.ndarray]:
    """
    Parameters: 
        dataset_path: Path to mounted dataset
    Returns: 
        tile_positions: Map of tilename -> zattrs translations
    """

    def read_json(json_path: str) -> dict: 
        with open(json_path) as f:
            return json.load(f)

    tile_positions = {}
    for tile_path in Path(dataset_path).iterdir():
        if tile_path.name == '.zgroup':
            continue

        zattrs_file = tile_path / '.zattrs'
        zattrs_json = read_json(zattrs_file)
        
        scale = zattrs_json['multiscales'][0]["datasets"][0]["coordinateTransformations"][0]['scale']
        translation = zattrs_json['multiscales'][0]["datasets"][0]["coordinateTransformations"][1]['translation']
 
        scale = np.array(scale[2:][::-1])
        translation = np.array(translation[2:][::-1])
        translation /= scale
        translation = np.round(translation, 4)

        tile_positions[tile_path.name] = translation

    return tile_positions


def generate_exaspim_link(
    xml_path: str,
    dataset_path: str, 
    s3_path: str,
    max_dr: int = 200,
    opacity: float = 1.0,
    blend: str = "default",
    output_json_path: str = ".",
) -> None:
    """Creates an neuroglancer link to visualize
    registration transforms on exaspim dataset pre-fusion.

    Parameters
    ------------------------
    xml_path: str
        Path of mounted xml output by BigStitcher.
    dataset_path: str
        Path of mounted dataset. 
    s3_path: str
        Path of s3 bucket where exaspim dataset is located.
        Function reads data from mount, but need s3 information within NG configuration. 
    output_json_path: str
        Local path to write process_output.json file that nueroglancer reads.

    Returns
    ------------------------
    None
    """

    # Gather xml info
    vox_sizes: tuple[float, float, float] = xml_parsing.extract_tile_vox_size(
        xml_path
    )
    tile_paths: dict[int, str] = xml_parsing.extract_tile_paths(xml_path)
    tile_transforms: dict[
        int, list[dict]
    ] = xml_parsing.extract_tile_transforms(xml_path)
    
    # Zattrs info
    zattrs_positions = get_zattrs_positions(dataset_path)

    # Update first translation in each tile's tile_transforms list
    for tile_id, tf in tile_transforms.items():

        t_path = tile_paths[tile_id]
        zattrs_offset = zattrs_positions[t_path]

        nums = [float(val) for val in tf[0]["affine"].split(" ")]
        nums[3] = nums[3] - zattrs_offset[0]
        nums[7] = nums[7] - zattrs_offset[1]
        nums[11] = nums[11] - zattrs_offset[2]

        tf[0]['affine'] = "".join(f'{n} ' for n in nums)
        tf[0]['affine'] = tf[0]['affine'].strip()

    net_transforms: dict[int, np.ndarray] = link_utils.calculate_net_transforms(tile_transforms)

    # Determine color
    channel: int = link_utils.extract_channel_from_tile_path(tile_paths[0])
    hex_val: int = link_utils.wavelength_to_hex(channel)
    hex_str = f"#{str(hex(hex_val))[2:]}"

    # Generate input config
    layers = []  # Nueroglancer Tabs
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

    sources = []  # Tiles within tabs
    layers.append(
        {
            "type": "image",  # Optional
            "source": sources,
            "channel": 0,  # Optional
            "shaderControls": {
                "normalized": {"range": [0, max_dr]}
            },  # Optional  # Exaspim has low HDR
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