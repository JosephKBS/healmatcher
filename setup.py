from setuptools import setup, find_packages

VERSION = '0.0.49' 
DESCRIPTION = 'Fast and simple probabilistic data matching package'

# Setting up
setup(
        name="healmatcher", 
        version=VERSION,
        author="Joseph Shim",
        author_email="<joseph.shim.rok@gmail.com>",
        url = "https://github.com/JosephKBS/healmatcher",
        description=DESCRIPTION,
   	    long_description=open("README.md", 'r').read(),
    	long_description_content_type='text/markdown',
        packages=find_packages(),
        install_requires=['splink','pandas','numpy','IPython','datetime','pyarrow'], 
        
        keywords=['probabilistic match', 'probabilistic data match', 'splink'],
        include_package_data=True,
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)