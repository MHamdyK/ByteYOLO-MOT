from setuptools import setup, find_packages

setup(
    name='byteyolo-mot',
    version='0.1.0',
    description='Multi-Object Tracking with YOLOv8 + ByteTrack',
    author='Your Name',
    license='Apache-2.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        # this will read from requirements.txt at install time
        *open('requirements.txt').read().splitlines()
    ],
    entry_points={
        'console_scripts': [
            'byteyolo=cli:main'
        ],
    },
)
