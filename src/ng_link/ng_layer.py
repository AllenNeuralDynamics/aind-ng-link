from typing import Any, Optional, Union
from pathlib import Path
import sys
from utils import utils

# IO types
PathLike = Union[str, Path]

class NgLayer():
    
    def __init__(
        self, 
        image_config:dict, 
        image_type:Optional[str]='image',
        mount_service:Optional[str]="s3",
    ) -> None:
        """
        Class constructor
        
        Parameters
        ------------------------
        image_config: dict
            Dictionary with the image configuration based on neuroglancer documentation.
        image_type: Optional[str]
            Image type based on neuroglancer documentation.
        mount_service: Optional[str]
            This parameter could be 'gs' referring to a bucket in Google Cloud or 's3'in Amazon.
        
        """
        
        self.__layer_state = {}
        self.image_config = image_config
        self.mount_service = mount_service
        self.image_type = image_type
        
        # Fix image source
        self.image_source = self.__fix_image_source(image_config['source'])
        
        
        self.update_state(image_config)
    
    def __fix_image_source(self, source_path:PathLike) -> str:
        """
        Fixes the image source path to include the type of image neuroglancer accepts.
        
        Parameters
        ------------------------
        source_path: PathLike
            Path where the image is located.
        
        Returns
        ------------------------
        str
            Fixed path for neuroglancer json configuration.
        """
        
        source_path = str(source_path)
        
        # replacing jupyter path or cloud run job path
        source_path = source_path.replace(
            '/home/jupyter/', ''
        ).replace(
            "////", "//"
        )
        
        source_path = f"{self.mount_service}://{source_path}" 
        
        if source_path.endswith('.zarr'):
            source_path = "zarr://" + source_path
            
        else:
            raise NotImplementedError("This format has not been implemented yet for visualization")
        
        return source_path
    
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
                    "range": [0, 600]
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
                        "range": [0, 600]
                    }
                }
                
            if 'visible' not in image_config:
                self.visible = True
                
            if 'name' not in image_config:
                try:
                    channel = self.__layer_state['localDimensions']["c'"][0]
                
                except KeyError:
                    channel = ''
                self.__layer_state['name'] =  f"{Path(self.image_source).stem}_{channel}"
                
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
                        "range": [0, 600]
                    }
                }
            }
        """
        
        for param, value in image_config.items():
            if param in ['type', 'source', 'name']:
                self.__layer_state[param] = str(value)
                
            if param in ['visible']:
                self.visible = value
                
            if param == 'shader':
                self.shader = self.__create_shader(value)
                
            if param == 'channel':
                self.image_channel = value
                
            if param == 'shaderControls':
                self.shader_control = value
        
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
        'source': 'image_path.zarr',
        'channel': 1, # Optional
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
        'visible': False # Optional
    }
    
    dict_data = NgLayer(image_config=example_data).layer_state
    print(dict_data)