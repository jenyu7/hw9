import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name='hw9',
    version='0.0.1',
    author='Jen Yu, Sammy Shaker, Arun Chakravorty',
    author_email='jenyu@caltech.edu',
    description='Packaged code for BE/Bi103a HW9',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    install_requires=["numpy","pandas", "bokeh>=1.4.0", "scipy", "numba", "bebi103", "iqplot", "colorcet", 
    "holoviews", "tqdm"],
    classifiers=(
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ),
)