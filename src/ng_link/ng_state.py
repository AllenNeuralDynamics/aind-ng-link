import re
from pathlib import Path
from typing import List, Optional, Union

from pint import UnitRegistry

from .ng_layer import NgLayer
from .utils import utils

# IO types
PathLike = Union[str, Path]


class NgState:
    def __init__(
        self,
        input_config: dict,
        mount_service: str,
        bucket_path: str,
        output_json: PathLike,
        verbose: Optional[bool] = False,
        base_url: Optional[str] = "https://neuroglancer-demo.appspot.com/",
        json_name: Optional[str] = "process_output.json",
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
        output_json: PathLike
            Path where the json will be written.
        verbose: Optional[bool]
            If true, additional information will be shown. Default False.
        base_url: Optional[str]
            Neuroglancer service url
        json_name: Optional[str]
            Name of json file with neuroglancer configuration

        """

        self.input_config = input_config
        self.output_json = Path(self.__fix_output_json_path(output_json))
        self.verbose = verbose
        self.mount_service = mount_service
        self.bucket_path = bucket_path
        self.base_url = base_url
        self.json_name = json_name

        # State and layers attributes
        self.__state = {}
        self.__dimensions = {}
        self.__layers = []

        # Initialize principal attributes
        self.initialize_attributes(self.input_config)

    def __fix_output_json_path(self, output_json: PathLike) -> str:

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
            str(output_json)
            .replace("/home/jupyter/", "")
            .replace("////", "//")
        )

        return output_json

    def __unpack_axis(
        self, axis_values: dict, dest_metric: Optional[str] = "meters"
    ) -> List:
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

        if dest_metric not in ["meters", "seconds"]:
            raise NotImplementedError(
                f"{dest_metric} has not been implemented"
            )

        # Converting to desired metric
        unit_register = UnitRegistry()
        quantity = (
            axis_values["voxel_size"] * unit_register[axis_values["unit"]]
        )
        dest_quantity = quantity.to(dest_metric)

        # Neuroglancer metric
        neuroglancer_metric = None
        if dest_metric == "meters":
            neuroglancer_metric = "m"

        elif dest_metric == "seconds":
            neuroglancer_metric = "s"

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
    def dimensions(self, new_dimensions: dict) -> None:

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
            raise ValueError(
                f"Dimensions accepts only dict. Received value: {new_dimensions}"
            )

        regex_axis = r"([x-zX-Z])$"

        for axis, axis_values in new_dimensions.items():

            if re.search(regex_axis, axis):
                self.__dimensions[axis] = self.__unpack_axis(axis_values)
            elif axis == "t":
                self.__dimensions[axis] = self.__unpack_axis(
                    axis_values, "seconds"
                )
            elif axis == "c'":
                self.__dimensions[axis] = [
                    axis_values["voxel_size"],
                    axis_values["unit"],
                ]

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
    def layers(self, layers: List[dict]) -> None:
        """
        Property setter of layers.

        Parameters
        ------------------------
        layers: List[dict]
            List that contains a configuration for each image layer.

        """

        if not isinstance(layers, list):
            raise ValueError(
                f"layers accepts only list. Received value: {layers}"
            )

        for layer in layers:
            self.__layers.append(
                NgLayer(
                    image_config=layer,
                    mount_service=self.mount_service,
                    bucket_path=self.bucket_path,
                    output_dimensions=self.dimensions,
                ).layer_state
            )

    @property
    def state(self, new_state: dict) -> None:
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
        actual_state["ng_link"] = self.get_url_link()
        actual_state["dimensions"] = {}

        # Getting actual state for all attributes
        for axis, value_list in self.__dimensions.items():
            actual_state["dimensions"][axis] = value_list

        actual_state["layers"] = self.__layers

        actual_state["showAxisLines"] = True
        actual_state["showScaleBar"] = True

        return actual_state

    def initialize_attributes(self, input_config: dict) -> None:
        """
        Initializes the following attributes for a given image layer: dimensions, layers.

        Parameters
        ------------------------
        input_config: dict
            Dictionary with the configuration for each image layer

        """

        # Initializing dimension
        self.dimensions = input_config["dimensions"]

        # Initializing layers
        self.layers = input_config["layers"]

        # Initializing state
        self.__state = self.state

        for key, val in input_config.items():
            if key == "showAxisLines":
                self.show_axis_lines = val

            elif key == "showScaleBar":
                self.show_scale_bar = val

    @property
    def show_axis_lines(self) -> bool:
        return self.__state["showAxisLines"]

    @show_axis_lines.setter
    def show_axis_lines(self, new_show_axis_lines: bool) -> None:
        """
        Sets the visible parameter in neuroglancer link.

        Parameters
        ------------------------
        new_show_axis_lines: bool
            Boolean that dictates if the image axis are visible or not.

        Raises
        ------------------------
        ValueError:
            If the parameter is not an boolean.
        """
        self.__state["showAxisLines"] = bool(new_show_axis_lines)

    @property
    def show_scale_bar(self) -> bool:
        return self.__state["showScaleBar"]

    @show_scale_bar.setter
    def show_scale_bar(self, new_show_scale_bar: bool) -> None:
        """
        Sets the visible parameter in neuroglancer link.

        Parameters
        ------------------------
        new_show_scale_bar: bool
            Boolean that dictates if the image scale bar are visible or not.

        Raises
        ------------------------
        ValueError:
            If the parameter is not an boolean.
        """
        self.__state["showScaleBar"] = bool(new_show_scale_bar)

    def save_state_as_json(self, update_state: Optional[bool] = False) -> None:
        """
        Saves a neuroglancer state as json.

        Parameters
        ------------------------
        update_state: Optional[bool]
            Updates the neuroglancer state with dimensions and layers in case they were changed
            using class methods. Default False
        """

        if update_state:
            self.__state = self.state

        final_path = Path(self.output_json).joinpath(self.json_name)
        utils.save_dict_as_json(final_path, self.__state, verbose=self.verbose)

    def get_url_link(self) -> str:
        """
        Creates the neuroglancer link based on where the json will be written.

        Returns
        ------------------------
        str
            Neuroglancer url to visualize data.
        """

        dataset_name = Path(self.output_json.stem)

        json_path = str(dataset_name.joinpath(self.json_name))
        json_path = f"{self.mount_service}://{self.bucket_path}/{json_path}"

        link = f"{self.base_url}#!{json_path}"

        return link


if __name__ == "__main__":

    example_data = {
        "dimensions": {
            # check the order
            "z": {"voxel_size": 2.0, "unit": "microns"},
            "y": {"voxel_size": 1.8, "unit": "microns"},
            "x": {"voxel_size": 1.8, "unit": "microns"},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "layers": [
            {
                "source": "/Users/camilo.laiton/repositories/aind-ng-link/src/ng_link/image_path.zarr",
                "channel": 0,
                # 'name': 'image_name_0',
                "shader": {"color": "green", "emitter": "RGB", "vec": "vec3"},
                "shaderControls": {  # Optional
                    "normalized": {"range": [0, 500]}
                },
            },
            {
                "source": "/Users/camilo.laiton/repositories/aind-ng-link/src/ng_link/image_path.zarr",
                "channel": 1,
                # 'name': 'image_name_1',
                "shader": {"color": "red", "emitter": "RGB", "vec": "vec3"},
                "shaderControls": {  # Optional
                    "normalized": {"range": [0, 500]}
                },
            },
        ],
    }

    neuroglancer_link = NgState(
        input_config=example_data,
        mount_service="s3",
        bucket_path="aind-msma-data",
        output_json="/Users/camilo.laiton/repositories/aind-ng-link/src",
    )

    data = neuroglancer_link.state
    print(data)
    # neuroglancer_link.save_state_as_json('test.json')
    neuroglancer_link.save_state_as_json()
    print(neuroglancer_link.get_url_link())

    example_data = {
        "dimensions": {
            # check the order
            "x": {"voxel_size": 0.74800002019210531934, "unit": "microns"},
            "y": {"voxel_size": 0.74800002019210531934, "unit": "microns"},
            "z": {"voxel_size": 1, "unit": "microns"},
            "c'": {"voxel_size": 1, "unit": ""},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "layers": [
            {
                "type": "image",  # Optional
                "source": [
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0000_y_0000_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -14192,
                            "delta_y": -10640,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0000_y_0001_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -14192,
                            "delta_y": -19684.000456947142,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0000_y_0002_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -14192,
                            "delta_y": -28727.998694435275,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0001_y_0000_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -26255.200652782467,
                            "delta_y": -10640,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0001_y_0001_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -26255.200652782467,
                            "delta_y": -19684.000456947142,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0001_y_0002_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -26255.200652782467,
                            "delta_y": -28727.998694435275,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0002_y_0000_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -38318.39686664473,
                            "delta_y": -10640,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0002_y_0001_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -38318.39686664473,
                            "delta_y": -19684.000456947142,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0002_y_0002_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -38318.39686664473,
                            "delta_y": -28727.998694435275,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0003_y_0000_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -50381.5952999671,
                            "delta_y": -10640,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0003_y_0001_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -50381.5952999671,
                            "delta_y": -19684.000456947142,
                            "delta_z": 0,
                        },
                    },
                    {
                        "url": "s3://aind-open-data/exaSPIM_609107_2022-09-21_14-48-48/exaSPIM/tile_x_0003_y_0002_z_0000_ch_488.zarr",
                        "transform_matrix": {
                            "delta_x": -50381.5952999671,
                            "delta_y": -28727.998694435275,
                            "delta_z": 0,
                        },
                    },
                ],
                "channel": 0,  # Optional
                "shaderControls": {  # Optional
                    "normalized": {"range": [30, 70]}
                },
                "visible": True,  # Optional
                "opacity": 0.50,
            }
        ],
        "showScaleBar": False,
        "showAxisLines": False,
    }

    neuroglancer_link = NgState(
        input_config=example_data,
        mount_service="s3",
        bucket_path="aind-msma-data",
        output_json="/Users/camilo.laiton/repositories/aind-ng-link/src",
    )

    data = neuroglancer_link.state
    # print(data)
    neuroglancer_link.save_state_as_json()
    print(neuroglancer_link.get_url_link())
