from typing import List, Any, Optional, Union
from .ng_layer import NgLayer
from pint import UnitRegistry
from pathlib import Path
import sys
import re
from utils import utils

# IO types
PathLike = Union[str, Path]

class NgState():
    def __init__(
        self, 
        input_config:dict, 
        output_json:Optional[PathLike], 
        verbose:Optional[bool]=False,
        mount_service:Optional[str]="s3",
        base_url:Optional[str]='https://neuroglancer-demo.appspot.com/',
        json_name:Optional[str]='process_output.json'
    ) -> None:
        """
        Class constructor
        
        Parameters
        ------------------------
        image_config: dict
            Dictionary with the image configuration based on neuroglancer documentation.
        output_json: PathLike
            Path where the json will be written.
        
        verbose: Optional[bool]
            If true, additional information will be shown. Default False.
        """
        
        self.input_config = input_config
        self.output_json = Path(self.__fix_output_json_path(output_json))
        self.verbose = verbose
        self.mount_service = mount_service
        self.base_url = base_url
        self.json_name = json_name
        
        # State and layers attributes
        self.__state = {}
        self.__dimensions = {}
        self.__layers = []
        
        # Initialize principal attributes
        self.initialize_attributes(self.input_config)
    
    def __fix_output_json_path(self, output_json:PathLike) -> str:
        
        """
        Fixes the json output path in order to have a similar structure for all links.
        
        Parameters
        ------------------------
        output_json: PathLike
            Path of the json output path.
        
        Returns
        ------------------------
        str
            String with the fixed outputh path.
        """
        output_json = Path(
            str(output_json).replace(
                '/home/jupyter/', ''
            ).replace(
                "////", "//"
            )
        )

        return output_json
    
    def __unpack_axis(self, axis_values:dict, dest_metric:Optional[str]='meters') -> List:
        """
        Unpack axis voxel sizes converting them to meters which neuroglancer uses by default.
        
        Parameters
        ------------------------
        axis_values: dict
            Dictionary with the axis values with the following structure for an axis: 
            e.g. for Z dimension {
                "voxel_size": 2.0,
                "unit": 'microns'
            }
        
        dest_metric: Optional[str]
            Destination metric to be used in neuroglancer. Default 'meters'.
        
        Returns
        ------------------------
        List
            List with two values, the converted quantity and it's metric in neuroglancer format.
        """
        
        if dest_metric not in ['meters', 'seconds']:
            raise NotImplementedError(f"{dest_metric} has not been implemented")
        
        # Converting to desired metric
        unit_register = UnitRegistry()
        quantity = axis_values['voxel_size'] * unit_register[axis_values['unit']]    
        dest_quantity = quantity.to(dest_metric)
        
        # Neuroglancer metric
        neuroglancer_metric = None
        if dest_metric == 'meters':
            neuroglancer_metric = 'm'
        
        elif dest_metric == 'seconds':
            neuroglancer_metric = 's'
        
        return [dest_quantity.m, neuroglancer_metric]
    
    @property
    def dimensions(self) -> dict:
        """
        Property getter of dimensions.
        
        Returns
        ------------------------
        dict
            Dictionary with neuroglancer dimensions' configuration.
        """
        return self.__dimensions
    
    @dimensions.setter
    def dimensions(self, new_dimensions:dict) -> None:
        
        """
        Set dimensions with voxel sizes for the image.
        
        Parameters
        ------------------------
        dimensions: dict
            Dictionary with the axis values with the following structure for an axis: 
            e.g. for Z dimension {
                "voxel_size": 2.0,
                "unit": 'microns'
            }
            
        """
        
        if not isinstance(new_dimensions, dict):
            raise ValueError(f"Dimensions accepts only dict. Received value: {new_dimensions}")

        regex_axis = r'([x-zX-Z])$'
        
        for axis, axis_values in new_dimensions.items():
            
            if re.search(regex_axis, axis):
                self.__dimensions[axis] = self.__unpack_axis(axis_values)
            else:
                self.__dimensions[axis] = self.__unpack_axis(axis_values, 'seconds')
    
    @property
    def layers(self) -> List[dict]:
        """
        Property getter of layers.
        
        Returns
        ------------------------
        List[dict]
            List with neuroglancer layers' configuration.
        """
        return self.__layers
    
    @layers.setter
    def layers(self, layers:List[dict]) -> None:
        """
        Property setter of layers.
        
        Parameters
        ------------------------
        layers: List[dict]
            List that contains a configuration for each image layer.
            
        """
        
        if not isinstance(layers, list):
            raise ValueError(f"layers accepts only list. Received value: {layers}")

        for layer in layers:
            self.__layers.append(
                NgLayer(
                    image_config=layer,
                    mount_service=self.mount_service
                ).layer_state
            )

    @property
    def state(self, new_state:dict) -> None:
        """
        Property setter of state.
        
        Parameters
        ------------------------
        input_config: dict
            Dictionary with the configuration for the neuroglancer state
        
        """
        self.__state = dict(new_state)
    
    @state.getter
    def state(self) -> dict:
        """
        Property getter of state.
        
        Returns
        ------------------------
        dict
            Dictionary with the actual layer state.
        """
        
        actual_state = {}
        actual_state['ng_link'] = self.get_url_link()
        actual_state['dimensions'] = {}

        # Getting actual state for all attributes
        for axis, value_list in self.__dimensions.items():
            actual_state['dimensions'][axis] = value_list
            
        actual_state['layers'] = self.__layers
        
        return actual_state
    
    def initialize_attributes(self, input_config:dict) -> None:
        """
        Initializes the following attributes for a given image layer: dimensions, layers.
        
        Parameters
        ------------------------
        input_config: dict
            Dictionary with the configuration for each image layer
    
        """
        
        # Initializing dimension
        self.dimensions = input_config['dimensions']
        
        # Initializing layers
        self.layers = input_config['layers']
        
        # Initializing state
        self.__state = self.state
    
    def save_state_as_json(
        self,
        update_state:Optional[bool]=False
    ) -> None:
        """
        Saves a neuroglancer state as json.
        
        Parameters
        ------------------------
        output_json: Optional[PathLike]
            Path where the neuroglancer state will be written as json
        
        update_state: Optional[bool]
            Updates the neuroglancer state with dimensions and layers in case they were changed 
            using class methods.
        """
        
        if update_state:
            self.__state = self.state

        final_path = Path(self.output_json).joinpath(self.json_name)
        utils.save_dict_as_json(final_path, self.__state, verbose=self.verbose)
    
    def get_url_link(
        self
    ) -> str:
        """
        Creates the neuroglancer link based on where the json will be written.
        
        Parameters
        ------------------------
        base_url: Optional[str]
            Base url where neuroglancer app was deployed. Default: https://neuroglancer-demo.appspot.com/
        
        save_txt: Optional[bool]
            Saves the url link to visualize data as a .txt file in a specific path given by 
            output_txt parameter.
        
        Returns
        ------------------------
        str
            Neuroglancer url to visualize data.
        """
        
        json_path = str(self.output_json.joinpath(self.json_name))
        json_path = f"{self.mount_service}://{json_path}"
        
        link = f"{self.base_url}#!{json_path}"
        
        return link
    
if __name__ == '__main__':
    
    example_data = {
        'dimensions': {
            # check the order
            "z": {
                "voxel_size": 2.0,
                "unit": 'microns'
            },
            "y": {
                "voxel_size": 1.8,
                "unit": 'microns'
            },
            "x": {
                "voxel_size": 1.8,
                "unit": 'microns'
            },
            "t": {
                "voxel_size": 0.001,
                "unit": 'seconds'
            },
        },
        'layers': [
            {
                'source': 'image_path.zarr',
                'channel': 0,
                # 'name': 'image_name_0',
                'shader': {
                    'color': 'green',
                    'emitter': 'RGB',
                    'vec': 'vec3'
                },
                'shaderControls': { # Optional
                    "normalized": {
                        "range": [0, 500]
                    }
                }
            },
            {
                'source': 'image_path.zarr',
                'channel': 1,
                # 'name': 'image_name_1',
                'shader': {
                    'color': 'red',
                    'emitter': 'RGB',
                    'vec': 'vec3'
                },
                'shaderControls': { # Optional
                    "normalized": {
                        "range": [0, 500]
                    }
                }
            }
        ]
    }
    
    neuroglancer_link = NgState(example_data, "C:/Users/camilo.laiton/Documents/Presentations")
    data = neuroglancer_link.state
    print(data)
    # neuroglancer_link.save_state_as_json('test.json')
    neuroglancer_link.save_state_as_json()
    print(neuroglancer_link.get_url_link())
    