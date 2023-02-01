from setuptools import setup, find_packages
from distutils.extension import Extension
import re
import os
import codecs
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

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = { }
ext_modules = [ ]

#print(find_version("deepgmap", "version.py"))
setup(
    name='omniplot',
    #version=VERSION,
    version="0.1.0",
    description='To draw scientific plots easily',
    author='Koh Onimaru',
    author_email='koh.onimaru@gmail.com',
    url='',

    packages=find_packages(),#['omniplot'],
    #provides=['omniplot'],
    #package_data = {
    #     '': ['enhancer_prediction/*', '*.pyx', '*.pxd', '*.c', '*.h'],
    #},
    #packages=find_packages(),
    cmdclass = cmdclass,
    ext_modules=ext_modules,
    classifiers=[
        'Development Status :: 0 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: Apache Software License ',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        
    
    ],
    install_requires=[ 'numpy', 
                      'matplotlib',
                      'scipy',"seaborn",
                      "pandas","igraph",
                      "umap-learn",
                      "tensorflow",
                      "fastcluster","cvxopt","statsmodels"
                      ],
    long_description=open('README.md').read(),
)