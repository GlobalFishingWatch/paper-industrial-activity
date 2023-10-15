"""Color utilities to make cmaps.

See:
  https://www.learnui.design/tools/data-color-picker.html

"""
import numpy as np
import matplotlib.colors as mpcolors


def hex_to_rgb(value):
    """Converts hex to rgb colours.

    Args:
        value: string of 6 characters representing a hex colour.

    Returns:
        list length 3 of RGB values
    """
    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(
        int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3)
    )


def rgb_to_dec(value):
    """Converts rgb to decimal colours (divides each value by 256).

    Args:
        value: list (length 3) of RGB values

    Returns:
        list (length 3) of decimal values
    """
    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None, n=256):
    """Create a color map that can be used in heat map.

    If float_list is not provided, colour map graduates
      linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is
      mapped to the respective location in float_list.

    Args:
      hex_list: list of hex code strings
      float_list: list of floats between 0 and 1, same length
        as hex_list. Must start with 0 and end with 1.

    Returns:
      colour map
    """
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = mpcolors.LinearSegmentedColormap("my_cmp", segmentdata=cdict, N=n)
    return cmp


def piecewise_constant_cmap(colors, name="pccm"):
    """Make a discrete color map.

    Args:
        colors is list[tuple(float, float, float)]

    Returns:
        colour map
    """
    breaks = np.linspace(0, 1.0, len(colors) + 1, endpoint=True)
    arg = {"red": [], "green": [], "blue": []}
    last_clr = colors[0]
    colors = colors + [colors[-1]]
    for i, clr in enumerate(colors):
        arg["red"].append((breaks[i], last_clr[0], clr[0]))
        arg["green"].append((breaks[i], last_clr[1], clr[1]))
        arg["blue"].append((breaks[i], last_clr[2], clr[2]))
        last_clr = clr
    return mpcolors.LinearSegmentedColormap(name, arg)


palette1 = [
    "#003f5c",
    "#58508d",
    "#bc5090",
    "#ff6361",
    "#ffa600",
]

palette2 = [
    # '#003f5c',
    "#444e86",
    "#955196",
    "#dd5182",
    "#ff6e54",
    "#ffa600",
]

palette3 = [
    "#004c6d",
    "#346888",
    "#5886a5",
    "#7aa6c2",
    "#9dc6e0",
    # '#c1e7ff',
]

palette4 = [
    "#90e28d",
    "#2cbd9b",
    "#00949b",
    "#006a87",
    "#1f4260",
]

palette5 = [
    "#374c80",
    "#7a5195",
    "#bc5090",
    "#ef5675",
    "#ff764a",
]

mycmap1 = get_continuous_cmap(palette1)
mycmap2 = get_continuous_cmap(palette2)
mycmap3 = get_continuous_cmap(palette3)
mycmap4 = get_continuous_cmap(palette4)
mycmap5 = get_continuous_cmap(palette5)
