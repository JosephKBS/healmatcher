from setuptools import setup, find_packages

VERSION = '0.0.15' 
DESCRIPTION = 'Fast and simple probabilistic data matching package'

# Setting up
setup(
        name="healmatcher", 
        version=VERSION,
        author="Joseph Shim github.com/JosephKBS",
        author_email="<joseph.shim.rok@gmail.com>",
        description=DESCRIPTION,
   	    long_description=open("README.md", 'r').read(),
    	long_description_content_type='text/markdown',
        packages=find_packages(),
        install_requires=['splink','pandas','numpy'], 
        
        keywords=['probabilistic match', 'probabilistic data match', 'splink'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)