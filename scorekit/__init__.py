# -*- coding: utf-8 -*-

'''
    Score-kit library for performing analytics and building scoring models

    We also have a description! Read in awe!
'''



from . import data, model, processor, woe, report, cross, ensemble
from .version import __version__

__all__ = ['data', 'model', 'processor', 'woe', 'report', 'cross', 'ensemble']
