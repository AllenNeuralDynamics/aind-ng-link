"""Tests ng layer class methods."""
import unittest

from ng_link.ng_layer import ImageLayer


class NgLayerTest(unittest.TestCase):
    """Tests ng layer class methods."""

    def test_image_layer_monochrome_shader(self):
        """
        Test that ImageLayer can generate a monochromatic shader
        """
        image_config = {
            "shader": {"color": "green", "emitter": "RGB", "vec": "vec3"},
            "source": "bogus.zarr",
        }

        layer = ImageLayer(
            image_config=image_config,
            mount_service="s3",
            bucket_path="silly/bucket",
            output_dimensions=dict(),
        )

        expected = '#uicontrol vec3 color color(default="green")\n'
        expected += "#uicontrol invlerp normalized\n"
        expected += "void main() {\n"
        expected += "emitRGB(color * normalized());\n"
        expected += "}"

        self.assertEqual(layer.shader, expected)

    def test_image_layer_rgb_shader(self):
        """
        Test that ImageLayer can generate a RGB shader
        """
        image_config = {
            "shader": {
                "r_range": (1, 9),
                "g_range": (18, 37),
                "b_range": (0, 100),
            },
            "source": "bogus.zarr",
        }

        layer = ImageLayer(
            image_config=image_config,
            mount_service="s3",
            bucket_path="silly/bucket",
            output_dimensions=dict(),
        )

        expected = "#uicontrol invlerp "
        expected += "normalized_r(range=[1, 9])\n"
        expected += "#uicontrol invlerp normalized_g(range=[18, 37])\n"
        expected += "#uicontrol invlerp normalized_b(range=[0, 100])\n"
        expected += "void main(){\n"
        expected += "float r = normalized_r(getDataValue(0));\n"
        expected += "float g = normalized_g(getDataValue(1));\n"
        expected += "float b = normalized_b(getDataValue(2));\n"
        expected += "emitRGB(vec3(r, g, b));\n}\n"

        self.assertEqual(layer.shader, expected)

    def test_image_layer_shader_failure(self):
        """
        Test that ImageLayer gives the expected failure
        when given nonsense image config paramters
        """
        image_config = {"shader": {"foo": "bar"}, "source": "bogus.zarr"}

        self.assertRaisesRegex(
            RuntimeError,
            "Do not know how to create shader code",
            ImageLayer,
            image_config=image_config,
            mount_service="s3",
            bucket_path="silly/bucket",
            output_dimensions=dict(),
        )


if __name__ == "__main__":
    unittest.main()
