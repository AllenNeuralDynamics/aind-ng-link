"""
Tests of utility functions for generating neuroglancer shader code
"""
import unittest

from ng_link.utils.shader_utils import (create_monochrome_shader,
                                        create_rgb_shader)


class ShaderUtilsTest(unittest.TestCase):
    """Class to contain tests of shader utils"""

    def test_monochrome_shader(self):
        """
        Just test that create_monochrome_shader runs.
        """
        result = create_monochrome_shader(
            color="green", emitter="RGB", vec="vec3"
        )

        expected = '#uicontrol vec3 color color(default="green")\n'
        expected += "#uicontrol invlerp normalized\n"
        expected += "void main() {\n"
        expected += "emitRGB(color * normalized());\n"
        expected += "}"

        self.assertEqual(result, expected)

    def test_rgb_shader(self):
        """
        Test that create_rgb_shader runs
        """
        result = create_rgb_shader(
            r_range=(1, 9), g_range=(18, 37), b_range=(0, 100)
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

        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
