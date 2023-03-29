from setuptools import setup, find_packages
from distutils.extension import Extension
import re
import os
import codecs
# from Cython.Build import cythonize

here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with codecs.open(os.path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(
        r"^__version__ = ['\"]([^'\"]*)['\"]",
        version_file,
        re.M,
    )
    if version_match:
        return version_match.group(1)

    raise RuntimeError("Unable to find version string.")

# try:
#     from Cython.Distutils import build_ext
# except ImportError:
#     use_cython = False
# else:
#     use_cython = True

long_description="""
omniplot is a python module to draw a scientific plot with hassle free. It mainly focuses on bioinfomatics data.
It is intended to be used in jupyter lab environment. 
"""

cmdclass = { }
ext_modules = []#cythonize("omniplot/cython/chipseq_utils.pyx")

VERSION=find_version("omniplot", "_version.py")
setup(
    name='omniplot',
    version=VERSION,
    #version="0.2.3",
    description='To draw scientific plots easily',
    author='Koh Onimaru',
    author_email='koh.onimaru@gmail.com',
    url='https://github.com/koonimaru/omniplot',
    python_requires='>=3.8',
    packages=find_packages(),
    cmdclass = cmdclass,
    ext_modules=ext_modules,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Visualization',
        
    
    ],
    install_requires=[ 'numpy', 
                      'matplotlib',
                      'scipy',"seaborn",
                      "pandas","igraph",
                      "umap-learn",
                      "tensorflow",
                      "fastcluster","cvxopt",
                      "statsmodels",
                      "natsort",
                      "joblib","pyranges",
                      "ray",
                      "intervaltree","networkx","datashader","python-louvain", "scikit-fuzzy","scikit-image",
                      ],
    long_description=long_description,
)