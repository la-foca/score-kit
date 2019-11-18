# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def color_digits(x, threshold_red, threshold_yellow=None):
    '''
    TECH

    Defines digit color based on value (correlation coefficient or VIF) for detailed output and excel export

    Parameters
    -----------
    x: input value
    threshold_red: the lowest value to be colored red
    threshold_yellow: the lowest value to be colored yellow (if None, then no yellow color used)

    Returns
    --------
    color description for style.applymap()
    '''
    if abs(x) > threshold_red:
        color = 'red'
    elif threshold_yellow is not None and abs(x) > threshold_yellow:
        color = 'orange'
    else:
        color = 'green'
    return 'color: %s' % color



def color_background(x, mn, mx, cntr=None, cmap=None, low=0, high=0):
    '''
    TECH

    Defines cell color based on value for excel export
    Values normalization uses colors.Normalize(mn - rng*low, mx + rng*high)

    Parameters
    -----------
    x: input value
    mn: minimal scale value for values normalization
    cntr: middle scale value for values normalization
    mx: maximal scale value for values normalization
    cmap: color map (if None, then custom colormap for mn=green, cntr=yellow, mx=red will be used)
    low: correcting coefficient for lower edge of gradient scale
    high: correcting coefficient for upper edge of gradient scale

    Returns
    --------
    color description for style.apply()
    '''
    if cmap is None:
        cmap=matplotlib.colors.LinearSegmentedColormap('PSImap', {'red': ((0.0, 0, 0),
                                                                            (cntr/mx, 1, 1),
                                                                            (1.0, 0.77, 0.77)),
                                                                  'green': ((0.0, 0.5, 0.5),
                                                                            (cntr/mx, 1, 1),
                                                                            (1.0, 0.0, 0.0)),
                                                                   'blue': ((0.0, 0.25, 0.25),
                                                                            (cntr/mx, 0.6, 0.6),
                                                                            (1.0, 0.07, 0.07))
                                                                  })
    rng = mx - mn
    norm = colors.Normalize(mn - rng*low, mx + rng*high)
    normed = norm(x.values)
    c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
    colors_list=['background-color: %s' % color for color in c]
    #костыль для корректной работы с пустым списком на выходе
    if len(colors_list)==0:
        colors_list=pd.Series(colors_list)
    return colors_list
