from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Fast and simple probablistic data matching package'
LONG_DESCRIPTION = 'This is a simple and fast data matching package developed by NYULH HEAL LAB'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="healmatcher", 
        version=VERSION,
        author="Joseph Shim",
        author_email="<joseph.shim.rok@gmail.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['splink','pandas','numpy'], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['probablistic match', 'probablistic data match', 'splink'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)