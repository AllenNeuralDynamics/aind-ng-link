def create_monochrome_shader(
        color: str,
        emitter: str,
        vec: str) -> str:
    """
    Creates a configuration for the neuroglancer shader.
    This shader generates a monochromatic image.

    Parameters
    ------------------------
    color:  str
        The color of the monochromatic image
    emitter: str
        suffix of WebGL emit* method being called (e.g. 'RGB')
    vec: str
        class of vector we want color returned as in WebGL
        (e.g. 'vec3')

    Returns
    ------------------------
    str
        String with the shader configuration for neuroglancer.
    """

    # Add all necessary ui controls here
    ui_controls = [
        f'#uicontrol {vec} color color(default="{color}")',
        "#uicontrol invlerp normalized",
    ]

    # color emitter
    emit_color = (
        "void main() {\n" + f"emit{emitter}(color * normalized());" + "\n}"
    )
    shader_string = ""

    for ui_control in ui_controls:
        shader_string += ui_control + "\n"

    shader_string += emit_color

    return shader_string
