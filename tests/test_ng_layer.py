"""Tests ng layer class methods."""
import pytest
import unittest

from ng_link.ng_layer import ImageLayer


class NgLayerTest(unittest.TestCase):
    """Tests ng layer class methods."""

    def test_image_layer_monochrome_shader(self):
        """
        Test that ImageLayer can generate a monochromatic shader
        """
        image_config = {
            "shader": {
                "color": "green",
                "emitter": "RGB",
                "vec": "vec3"},
            "source": "bogus.zarr"
        }

        layer = ImageLayer(
            image_config=image_config,
            mount_service="s3",
            bucket_path="silly/bucket",
            output_dimensions=dict())

        expected = '#uicontrol vec3 color color(default="green")\n'
        expected += '#uicontrol invlerp normalized\n'
        expected += 'void main() {\n'
        expected += 'emitRGB(color * normalized());\n'
        expected += '}'

        assert layer.shader == expected

    def test_image_layer_shader_failure(self):
        """
        Test that ImageLayer gives the expected failure
        when given nonsense image config paramters
        """
        image_config = {
            "shader": {
                "foo": "bar"},
            "source": "bogus.zarr"
        }

        msg = "Do not know how to create shader code"
        with pytest.raises(RuntimeError, match=msg):
            ImageLayer(
                image_config=image_config,
                mount_service="s3",
                bucket_path="silly/bucket",
                output_dimensions=dict())


if __name__ == "__main__":
    unittest.main()
