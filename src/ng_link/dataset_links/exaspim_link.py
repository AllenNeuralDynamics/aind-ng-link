import numpy as np

from ng_link import NgState

import xml_parsing
import link_utils

def omit_initial_offsets(view_transforms: dict[int, list[dict]]) -> None:
    """
    For OME-Zarr datasets, inital offsets are already encoded in the metadata and
    extracted my neuroglancer. This function removes the duplicate transform.

    Parameters
    ------------------------
    view_transforms: dict[int, list[dict]]
        Dictionary of tile ids to list of transforms.

    Returns
    ------------------------
    None
    """

    for view, tfs in view_transforms.items(): 
        tfs.pop(0)

def generate_exaspim_link(xml_path: str, 
                          s3_path: str, 
                          max_dr: int = 200, 
                          opacity: float = 1.0, 
                          blend: str = "default",
                          output_json_path: str = ".") -> None:
    """
    Creates an neuroglancer link to visualize registration transforms on exaspim dataset pre-fusion.

    Parameters
    ------------------------
    xml_path: str
        Path of xml outputted by BigStitcher.    
    s3_path: str
        Path of s3 bucket where exaspim dataset is located.
    output_json_path: str
        Local path to write process_output.json file that nueroglancer reads.
        
    Returns
    ------------------------
    None
    """

    # Gather xml info
    vox_sizes: tuple[float, float, float] = xml_parsing.extract_tile_vox_size(xml_path)
    tile_paths: dict[int, str] = xml_parsing.extract_tile_paths(xml_path)
    tile_transforms: dict[int, list[dict]] = xml_parsing.extract_tile_transforms(xml_path)
    omit_initial_offsets(tile_transforms)
    net_transforms: dict[int, np.ndarray] = link_utils.calculate_net_transforms(tile_transforms)

    # Determine color
    channel: int = link_utils.extract_channel_from_tile_path(tile_paths[0])
    hex_val: int = link_utils.wavelength_to_hex(channel)
    hex_str = f'#{str(hex(hex_val))[2:]}'

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
            "shaderControls": {  # Optional
                "normalized": {"range": [0, max_dr]}   # Exaspim has low HDR
            },
            "shader": {
                "color": hex_str,
                "emitter": "RGB",
                "vec": "vec3",
            },
            "visible": True,  # Optional
            "opacity": opacity,
            "name": f"CH_{channel}",
            "blend": blend
        }
    )

    for tile_id, _ in enumerate(net_transforms): 
        net_tf = net_transforms[tile_id]
        t_path = tile_paths[tile_id]

        url = f"{s3_path}/{t_path}"
        final_transform = link_utils.convert_matrix_3x4_to_5x6(net_tf)

        sources.append(
            {
                "url": url,
                "transform_matrix": final_transform.tolist()
            }
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