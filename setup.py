from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Distribution which always forces a binary package with platform name"""

    def has_ext_modules(self) -> bool:
        return True


setup(
    # Include pre-compiled extension
    # package_data={"packagename": ["_precompiled_extension.pyd"]},
    # distclass=BinaryDistribution,
)
