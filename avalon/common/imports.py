"""
This module defines all of the 3rd party aliases that we use.
It doesn't really do anything except make it much more convenient to import names from pycharm.
"""

import numpy
from tqdm.auto import tqdm as _tqdm

# canonical names
np = numpy

# .auto is never even suggested, sigh
tqdm = _tqdm
