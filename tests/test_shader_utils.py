from ng_link.utils.shader_utils import (
    create_monochrome_shader)


def test_monochrome_shader():
    """
    Just test that create_monochrome_shader runs.
    """
    result = create_monochrome_shader(
        color='green',
        emitter='RGB',
        vec='vec3')

    expected = '#uicontrol vec3 color color(default="green")\n'
    expected += '#uicontrol invlerp normalized\n'
    expected += 'void main() {\n'
    expected += 'emitRGB(color * normalized());\n'
    expected += '}'

    assert result == expected
