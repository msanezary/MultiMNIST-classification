from setuptools import setup, find_packages

setup(
    name='MultiMNIST',  # Name of the package
    version='0.1.0',  # Version number
    author='Mohammed Said ANEZARY, Jad BOUTZIL',
    author_email='msanezary@gmail.com, boutzil.50@gmail.com', 
    description='A machine learning project to classify overlaid digits from the MNIST dataset.', 
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown',  # Type of the long description
    url='https://github.com/msanezary/MultiMNIST-classification',  # Project URL
    packages=find_packages(where='src'),  # List of all Python import packages that should be included in the Distribution Package
    package_dir={'': 'src'},  # Directory where find_packages looks for Python packages
    install_requires=[
        'numpy',
        'torch>=1.8',
        'torchvision>=0.9.0',
        'matplotlib',  # Only include if you're using plotting functions in your package
        'pickle-mixin'  # Include if necessary for Python 3 compatibility
    ], 
    classifiers=[
        'Development Status :: 3 - Alpha', 
        'Intended Audience :: Developers', 
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',  # Supported programming languages
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Minimum version requirement of the Python programming language
    include_package_data=True, 
    zip_safe=False,  # Unpacked the package will be and easier to work with
    scripts=['src/main.py']  # Script to be installed
)
