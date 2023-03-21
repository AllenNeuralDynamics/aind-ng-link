from collections import OrderedDict, defaultdict
import xmltodict
import numpy as np

from ng_state import NgState

def extract_xml_transforms(xml_path: str) -> dict[int, list[dict]]: 
   """
   Parses BDV xml and organizes transformations into view stacks. 
   Output dictionary maps view number to list of {'@type', 'Name', 'affine'}
   where 'affine' contains the transform as string of 12 floats.
   """
   
   view_transforms: dict[int, list[dict]] = {}
   with open(xml_path, 'r') as file: 
      data: OrderedDict = xmltodict.parse(file.read())

   for view_reg in data['SpimData']['ViewRegistrations']['ViewRegistration']:
      view_transforms[view_reg['@setup']] = view_reg['ViewTransform']

   return view_transforms

def calculate_net_transforms(view_transforms: dict[int, list[dict]]) -> dict[int, np.ndarray]:
    """
    Accumulate net transform and net translation for each matrix stack. 
    Net translation = Sum of translation vectors converted into original nominal basis
    Net transform = Product of 3x3 matrices
    """

    identity_transform = np.array([[1., 0., 0., 0.], 
                                  [0., 1., 0., 0.], 
                                  [0., 0., 1., 0.]])
    net_transforms: dict[int, np.ndarray] = defaultdict(lambda: np.copy(identity_transform))

    for view, tfs in view_transforms.items():
        net_translation = np.zeros(3)
        net_matrix_3x3 = np.eye(3)
        curr_inverse = np.eye(3)

        for tf in tfs:   # Tfs is a list of transforms
            nums = [float(val) for val in tf['affine'].split(' ')]
            matrix_3x3 = np.array([nums[0::4], nums[1::4], nums[2::4]])
            translation = np.array(nums[3::4])

            net_translation = net_translation + (curr_inverse @ translation)
            net_matrix_3x3 = matrix_3x3 @ net_matrix_3x3
            curr_inverse = np.linalg.inv(net_matrix_3x3)  # Update curr_inverse
    
        net_transforms[view] = np.hstack((net_matrix_3x3, net_translation.reshape(3, 1)))

    return net_transforms

def apply_visual_transform(matrix_3x4: np.ndarray, camera_index: int, theta: float = 45) -> np.ndarray:
    """
    Applies two visualization transforms on 3x4 matrix: deskewing and camera alignment.
    """

    # Deskewing x vector into xz direction
    matrix_3x4[0, 2] = np.tan(np.deg2rad(theta))

    # Camera Alignment
    if camera_index == 1:
        matrix_3x4[2, 2] = -1
    
    return matrix_3x4

def convert_matrix_3x4_to_5x6(matrix_3x4: np.ndarray) -> np.ndarray:
    """
    Converts classic 3x4 homogeneous coordinates: (x y z T)
    to nueroglancer 5x6 coordinates (t c z y x T)
    """

    # Initalize
    matrix_5x6 = np.zeros((5, 6), np.float16)
    np.fill_diagonal(matrix_5x6, 1)

    # Swap Rows 0 and 2; Swap Colums 0 and 2
    patch = np.copy(matrix_3x4)
    patch[[0, 2], :] = patch[[2, 0], :]
    patch[:, [0, 2]] = patch[:, [2, 0]]
    
    # Place patch in bottom-right corner
    matrix_5x6[2:6, 2:7] = patch

    return matrix_5x6

# Main 
def create_ng_link(xml_path: str, s3_path: str, output_json_path: str = "."):
    """
    Creates NgState object from xml transforms and s3 metadata. 
    Outputs neuroglancer json to output json path. 
    """

    # Transforms from xml
    xml_transforms: dict = extract_xml_transforms(xml_path)
    net_transforms: dict = calculate_net_transforms(xml_transforms)

    # NOTE: HARDCODED
    # Metadata from S3
    camera_indices = [0, 1]
    channel_names = ["0405", "0488", "0561"]
    colors = ["#3f2efe", "#58fea1", "#f15211"]

    # Generate input config
    layers = []
    input_config = {
        "dimensions": {
            # check the order
            "x": {"voxel_size": 0.298, "unit": "microns"},
            "y": {"voxel_size": 0.298, "unit": "microns"},
            "z": {"voxel_size": 0.176, "unit": "microns"},
            "c'": {"voxel_size": 1, "unit": ""},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "layers": layers,
        "showScaleBar": False,
        "showAxisLines": False,
    }

    # Append to layers 
    for camera_index in camera_indices:
        for channel_name, color in zip(channel_names, colors):
            sources = []
            layers.append(
                {
                    "type": "image",  # Optional
                    "source": sources,
                    "channel": 0,  # Optional
                    "shaderControls": {  # Optional
                        "normalized": {"range": [0, 800]}
                    },
                    "shader": {
                        "color": color,
                        "emitter": "RGB",
                        "vec": "vec3",
                    },
                    "visible": True,  # Optional
                    "opacity": 0.50,
                    "name": f"CH_{channel_name}_CAM{camera_index}",
                }
            )

            # Append to sources
            for tile_num, transform in net_transforms.items(): 
                if tile_num < 10: 
                    tile_num = "0" + str(tile_num)
                url = f"{s3_path}/tile_X_00{tile_num}_Y_0000_Z_0000_CH_{channel_name}_cam{camera_index}.zarr"
                final_transform = convert_matrix_3x4_to_5x6(apply_visual_transform(transform))

                sources.append(
                    {
                        "url": url,
                        "transform_matrix": final_transform.tolist(),
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


# TODO: Test in Code Ocean
if __name__ == '__main__':
    xml_path = '...'
    s3_path = "s3://aind-open-data/diSPIM_647459_2022-12-07_00-00-00/diSPIM.zarr"

    create_ng_link(xml_path, s3_path)