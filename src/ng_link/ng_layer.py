from typing import Any, Optional, Union, List, Dict, get_args
from pathlib import Path
import sys
from .utils import utils
import numpy as np

# IO types
PathLike = Union[str, Path]
SourceLike = Union[PathLike, List[Dict]]

def helper_create_ng_translation_matrix(
    delta_x:Optional[float]=0, 
    delta_y:Optional[float]=0,
    delta_z:Optional[float]=0,
    n_cols:Optional[int]=6,
    n_rows:Optional[int]=5
) -> List:
    translation_matrix = np.zeros((n_rows, n_cols), np.float16)
    np.fill_diagonal(translation_matrix, 1)
    
    deltas = [delta_x, delta_y, delta_z]
    start_point = n_rows - 1
    
    if start_point < len(deltas):
        raise ValueError("N size of transformation matrix is not enough for deltas")
    
    # Setting translations for axis
    for delta in deltas:
        translation_matrix[start_point][-1] = delta
        start_point -= 1

    return translation_matrix.tolist()

def helper_reverse_dictionary(dictionary:dict) -> dict:

    keys = list(dictionary.keys())
    values = list(dictionary.values())
    new_dict = {}

    for idx in range(len(keys)-1, -1, -1):
        new_dict[keys[idx]] = values[idx]

    return new_dict

class NgLayer():
    
    def __init__(
        self,
        image_config:dict, 
        mount_service:str,
        bucket_path:str,
        image_type:Optional[str]='image',
        output_dimensions:Optional[dict]=None
    ) -> None:
        """
        Class constructor
        
        Parameters
        ------------------------
        image_config: dict
            Dictionary with the image configuration based on neuroglancer documentation.
        mount_service: Optional[str]
            This parameter could be 'gs' referring to a bucket in Google Cloud or 's3'in Amazon.
        bucket_path: str
            Path in cloud service where the dataset will be saved
        image_type: Optional[str]
            Image type based on neuroglancer documentation.
        
        """
        
        self.__layer_state = {}
        self.image_config = image_config
        self.mount_service = mount_service
        self.bucket_path = bucket_path
        self.image_type = image_type

        # Optional parameter that must be used when we have multiple images per layer
        # Dictionary needs to be reversed for correct visualization
        self.output_dimensions = helper_reverse_dictionary(output_dimensions)

        # Fix image source
        self.image_source = self.__fix_image_source(image_config['source'])
        image_config['source'] = self.image_source

        self.update_state(image_config)
    
    def __fix_image_source(self, source_path:SourceLike) -> str:
        """
        Fixes the image source path to include the type of image neuroglancer accepts.
        
        Parameters
        ------------------------
        source_path: SourceLike
            Path or list of paths where the images are located with their transformation matrix.
        
        Returns
        ------------------------
        SourceLike
            Fixed path(s) for neuroglancer json configuration.
        """
        new_source_path = None

        def set_s3_path(orig_source_path:PathLike) -> str:
            
            s3_path = None
            if not orig_source_path.startswith(f"{self.mount_service}://"):
                orig_source_path = Path(orig_source_path)
                s3_path = f"{self.mount_service}://{self.bucket_path}/{orig_source_path}"
            
            else:
                s3_path = orig_source_path
        
            if s3_path.endswith('.zarr'):
                s3_path = "zarr://" + s3_path
                
            else:
                raise NotImplementedError("This format has not been implemented yet for visualization")

            return s3_path

        if isinstance(source_path, list):
            # multiple sources in single image
            new_source_path = []

            for source in source_path:
                new_dict = {}

                for key in source.keys():
                    if key == 'transform_matrix':
                        new_dict['transform'] = {
                            'matrix': helper_create_ng_translation_matrix(
                                delta_x=source['transform_matrix']['delta_x'],
                                delta_y=source['transform_matrix']['delta_y'],
                                delta_z=source['transform_matrix']['delta_z']
                            ),
                            'outputDimensions' : self.output_dimensions,
                        }
                    
                    elif key == 'url':
                        new_dict['url'] = set_s3_path(source['url'])

                    else:
                        new_dict[key] = source[key]

                new_source_path.append(new_dict)
                # new_source_path.append(
                #     {
                #         'url': set_s3_path(source['url']),
                #         'transform': {
                #             'matrix': helper_create_ng_translation_matrix(
                #                 delta_x=source['transform_matrix']['delta_x'],
                #                 delta_y=source['transform_matrix']['delta_y'],
                #                 delta_z=source['transform_matrix']['delta_z']
                #             ),
                #             'outputDimensions' : self.output_dimensions,
                #         },
                #         'subsources': {
                #             'default': True
                #         },
                #         'enableDefaultSubsources': False,
                #     }
                # )
        
        elif isinstance(source_path, get_args(PathLike)):
            # Single source image
            new_source_path = set_s3_path(source_path)
        
        return new_source_path
    
    def set_default_values(self, image_config:dict={}, overwrite:bool=False) -> None:
        """
        Set default values for the image.
        
        Parameters
        ------------------------
        image_config: dict
            Dictionary with the image configuration. Similar to self.image_config
        
        overwrite: bool
            If the parameters already have values, with this flag they can be overwritten.
        
        """        
        
        if overwrite:
            self.image_channel = 0
            self.shader_control = {
                "normalized": {
                    "range": [0, 200]
                }
            }
            self.visible = True
            self.__layer_state['name'] = str(Path(self.image_source).stem)
            self.__layer_state['type'] = str(self.image_type)
        
        elif len(image_config):
            # Setting default image_config in json image layer
            if 'channel' not in image_config:
                # Setting channel to 0 for image
                self.image_channel = 0
                
            if 'shaderControls' not in image_config:
                self.shader_control = {
                    "normalized": {
                        "range": [0, 200]
                    }
                }
                
            if 'visible' not in image_config:
                self.visible = True
                
            if 'name' not in image_config:
                try:
                    channel = self.__layer_state['localDimensions']["c'"][0]
                
                except KeyError:
                    channel = ''
                
                if isinstance(self.image_source, get_args(PathLike)):
                    self.__layer_state['name'] =  f"{Path(self.image_source).stem}_{channel}"
                
                else:
                    self.__layer_state['name'] =  f"{Path(self.image_source[0]['url']).stem}_{channel}"

            if 'type' not in image_config:
                self.__layer_state['type'] = str(self.image_type)
    
    def update_state(self, image_config:dict) -> None:
        """
        Set default values for the image.
        
        Parameters
        ------------------------
        image_config: dict
            Dictionary with the image configuration. Similar to self.image_config
            e.g.: image_config = {
                'type': 'image', # Optional
                'source': 'image_path',
                'channel': 0, # Optional
                'name': 'image_name', # Optional
                'shader': {
                    'color': 'green',
                    'emitter': 'RGB',
                    'vec': 'vec3'
                },
                'shaderControls': { # Optional
                    "normalized": {
                        "range": [0, 200]
                    }
                }
            }
        """
        
        for param, value in image_config.items():
            if param in ['type', 'name', 'blend']:
                self.__layer_state[param] = str(value)
                
            if param in ['visible']:
                self.visible = value
                
            if param == 'shader':
                self.shader = self.__create_shader(value)
                
            if param == 'channel':
                self.image_channel = value
                
            if param == 'shaderControls':
                self.shader_control = value
            
            if param == 'opacity':
                self.opacity = value

            if param == 'source':
                if isinstance(value, get_args(PathLike)):
                    self.__layer_state[param] = str(value)
                
                elif isinstance(value, list):
                    # Setting list of dictionaries with image configuration
                    self.__layer_state[param] = value
        
        self.set_default_values(image_config)
    
    def __create_shader(self, shader_config:dict) -> str:
        """
        Creates a configuration for the neuroglancer shader.
        
        Parameters
        ------------------------
        shader_config: dict
            Configuration of neuroglancer's shader.
        
        Returns
        ------------------------
        str
            String with the shader configuration for neuroglancer.
        """
        
        color = shader_config['color']
        emitter = shader_config['emitter']
        vec = shader_config['vec']
        
        # Add all necessary ui controls here
        ui_controls = [
            f"#uicontrol {vec} color color(default=\"{color}\")",
            "#uicontrol invlerp normalized",
        ]
        
        # color emitter
        emit_color = "void main() {\n" + f"emit{emitter}(color * normalized());" + "\n}"
        shader_string = ""
        
        for ui_control in ui_controls:
            shader_string += ui_control + '\n'
        
        shader_string += emit_color
        
        return shader_string
    
    @property
    def opacity(self) -> str:
        return self.__layer_state['opacity']

    @opacity.setter
    def opacity(self, opacity:float) -> None:
        """
        Sets the opacity parameter in neuroglancer link.
        
        Parameters
        ------------------------
        opacity: float
            Float number between [0-1] that indicates the opacity.
        
        Raises
        ------------------------
        ValueError:
            If the parameter is not an boolean.
        """
        self.__layer_state['opacity'] = float(opacity)

    @property
    def shader(self) -> str:
        return self.__layer_state['shader']
    
    @shader.setter
    def shader(self, shader_config:str) -> None:
        """
        Sets a configuration for the neuroglancer shader.
        
        Parameters
        ------------------------
        shader_config: str
            Shader configuration for neuroglancer in string format.
            e.g. #uicontrol vec3 color color(default=\"green\")\n#uicontrol invlerp normalized\nvoid main() {\n  emitRGB(color * normalized());\n}
        
        Raises
        ------------------------
        ValueError:
            If the provided shader_config is not a string.
        
        """
        self.__layer_state['shader'] = str(shader_config)
    
    @property
    def shader_control(self) -> dict:
        return self.__layer_state['shaderControls']
    
    @shader_control.setter 
    def shader_control(self, shader_control_config:dict) -> None:
        """
        Sets a configuration for the neuroglancer shader control.
        
        Parameters
        ------------------------
        shader_control_config: dict
            Shader control configuration for neuroglancer.
        
        Raises
        ------------------------
        ValueError:
            If the provided shader_control_config is not a dictionary.
        
        """
        self.__layer_state['shaderControls'] = dict(shader_control_config) 
    
    @property
    def image_channel(self) -> None:
        return self.__layer_state['localDimensions']['c']
    
    @image_channel.setter
    def image_channel(self, channel:int) -> None:
        """
        Sets the image channel in case the file contains multiple channels.
        
        Parameters
        ------------------------
        channel: int
            Channel position. It will be incremented in 1 since neuroglancer channels starts in 1.
        
        Raises
        ------------------------
        ValueError:
            If the provided channel is not an integer.
        
        """
        self.__layer_state['localDimensions'] = {
            "c'": [
                int(channel) + 1,
                ""
            ]
        }
    
    @property
    def visible(self) -> bool:
        return self.__layer_state['visible']
    
    @visible.setter
    def visible(self, visible:bool) -> None:
        """
        Sets the visible parameter in neuroglancer link.
        
        Parameters
        ------------------------
        visible: bool
            Boolean that dictates if the image is visible or not.
        
        Raises
        ------------------------
        ValueError:
            If the parameter is not an boolean.
        """
        self.__layer_state['visible'] = bool(visible)
    
    @property
    def layer_state(self) -> dict:
        return self.__layer_state

    @layer_state.setter
    def layer_state(self, new_layer_state:dict) -> None:
        self.__layer_state = dict(new_layer_state)
        
if __name__ == '__main__':
    
    example_data = {
        'type': 'image', # Optional
        'source': 'relative/folder/to_bucket/image_path.zarr',
        'channel': 0, # Optional
        # 'name': 'image_name', # Optional
        'shader': {
            'color': 'red',
            'emitter': 'RGB',
            'vec': 'vec3'
        },
        'shaderControls': { # Optional
            "normalized": {
                "range": [0, 500]
            }
        },
        'visible': False, # Optional
        'opacity': 0.50
    }

    image_config = example_data
    mount_service = 's3'
    bucket_path = 'aind-open-data'

    dict_data = NgLayer(image_config, mount_service, bucket_path).layer_state
    print(dict_data)

    example_data = {
        'type': 'image', # Optional
        'source': [
            {
                'url': 's3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0000_y_0000_z_0000_ch_488.zarr',
                'transform_matrix': {
                    'delta_x' : -14192,
                    'delta_y': -10640,
                    'delta_z': 0
                }
            },
            {
                'url': 's3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0000_y_0001_z_0000_ch_488.zarr',
                'transform_matrix': {
                    'delta_x' : -14192,
                    'delta_y': -19684.000456947142,
                    'delta_z': 0
                }
            },
            {
                'url': 's3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0000_y_0002_z_0000_ch_488.zarr',
                'transform_matrix': {
                    'delta_x' : -14192,
                    'delta_y': -28727.998694435275,
                    'delta_z': 0
                }
            },
            {
                'url': 's3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0001_y_0000_z_0000_ch_488.zarr',
                'transform_matrix': {
                    'delta_x' : -26255.200652782467,
                    'delta_y': -10640,
                    'delta_z': 0
                }
            },
            {
                'url': 's3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0001_y_0001_z_0000_ch_488.zarr',
                'transform_matrix': {
                    'delta_x' : -26255.200652782467,
                    'delta_y': -19684.000456947142,
                    'delta_z': 0
                }
            },
            {
                'url': 's3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0001_y_0002_z_0000_ch_488.zarr',
                'transform_matrix': {
                    'delta_x' : -26255.200652782467,
                    'delta_y': -28727.998694435275,
                    'delta_z': 0
                }
            },
            {
                'url': 's3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0002_y_0000_z_0000_ch_488.zarr',
                'transform_matrix': {
                    'delta_x' : -38318.39686664473,
                    'delta_y': -10640,
                    'delta_z': 0
                }
            },
            {
                'url': 's3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0002_y_0001_z_0000_ch_488.zarr',
                'transform_matrix': {
                    'delta_x' : -38318.39686664473,
                    'delta_y': -19684.000456947142,
                    'delta_z': 0
                }
            },
            {
                'url': 's3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0002_y_0002_z_0000_ch_488.zarr',
                'transform_matrix': {
                    'delta_x' : -38318.39686664473,
                    'delta_y': -28727.998694435275,
                    'delta_z': 0
                }
            },
            {
                'url': 's3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0003_y_0000_z_0000_ch_488.zarr',
                'transform_matrix': {
                    'delta_x' : -50381.5952999671,
                    'delta_y': -10640,
                    'delta_z': 0
                }
            },
            {
                'url': 's3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0003_y_0001_z_0000_ch_488.zarr',
                'transform_matrix': {
                    'delta_x' : -50381.5952999671,
                    'delta_y': -19684.000456947142,
                    'delta_z': 0
                }
            },
            {
                'url': 's3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0003_y_0002_z_0000_ch_488.zarr',
                'transform_matrix': {
                    'delta_x' : -50381.5952999671,
                    'delta_y': -28727.998694435275,
                    'delta_z': 0
                }
            }
        ],
        'channel': 0, # Optional
        'shaderControls': { # Optional
            "normalized": {
                "range": [30, 70]
            }
        },
        'visible': True, # Optional
        'opacity': 0.50
    }

    image_config = example_data
    mount_service = 's3'
    bucket_path = 'aind-open-data'
    output_dimensions = {
        "t": [
        0.001,
        "s"
        ],
        "c'": [
        1,
        ""
        ],
        "z": [
        0.000001,
        "m"
        ],
        "y": [
        7.480000201921053e-7,
        "m"
        ],
        "x": [
        7.480000148631623e-7,
        "m"
        ]
    }

    dict_data = NgLayer(
        image_config=image_config, 
        mount_service=mount_service, 
        bucket_path=bucket_path,
        output_dimensions=output_dimensions
    ).layer_state
    print(dict_data)
    utils.save_dict_as_json('test.json', dict_data)
