from setuptools import setup, find_packages



with open('README.md') as f:
    readme = f.read()

setup(
    name='ane_gpt',
    version='0.1.2',
    url='https://github.com/starxsky/NewANEGPT',
    description="Reference PyTorch implementation of GPT for Apple Neural Engine (ANE) deployment",
    long_description=readme,
    long_description_content_type='text/markdown',
    author='StarxSky & Apple Inc.',
    install_requires=[
        "torch>=1.10.0,<=1.11.0",
        "coremltools>=5.2.0",
        "transformers>=4.18.0",
        "protobuf>=3.1.0,<=3.20.1",
    ],
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
)
