#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File name: plot_settings.py
Author: Yannick Heiser
Created: 2025-11-27
Version: 1.0
Description:
    Settings for plotting visualizations.

Contact: yahei@dtu.dk
Dependencies:
    - matplotlib
"""

# plot_settings.py
import matplotlib.pyplot as plt

def apply_plot_settings():
    """
    Apply default plot settings for visualizations.
    """
    plt.style.use('seaborn-v0_8-colorblind')
    x_length = 10
    golden_ratio = (1 + 5 ** 0.5) / 2
    plt.rcParams['figure.figsize'] = (x_length, x_length / golden_ratio)
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

# Color palettes
color_palette_1 = {
    'black': (0, 0, 0),
    'orange': (230, 159, 0),
    'sky_blue': (86, 180, 233),
    'bluish_green': (0, 158, 115),
    'yellow': (240, 228, 66),
    'blue': (0, 114, 178),
    'vermillion': (213, 94, 0),
    'reddish_purple': (204, 121, 167)
}
color_palette_1 = {name: (r/255, g/255, b/255) for name, (r, g, b) in color_palette_1.items()}

color_palette_2 = {
    'blue': (68, 119, 170),
    'cyan': (102, 204, 238),
    'green': (34, 136, 51),
    'yellow': (204, 187, 68),
    'red': (238, 102, 119),
    'purple': (170, 51, 119),
    'grey': (187, 187, 187)
}
color_palette_2 = {name: (r/255, g/255, b/255) for name, (r, g, b) in color_palette_2.items()}
