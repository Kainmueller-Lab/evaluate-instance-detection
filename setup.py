from setuptools import setup

setup(
        name='evaluate-instance-detection',
        version='0.1',
        description='Evaluate instance detection.',
        url='https://github.com/Kainmueller-Lab/evaluate-instance-detection',
        author='Peter Hirsch',
        author_email='kainmuellerlab@mdc-berlin.de',
        license='MIT',
        install_requires=[
            'h5py',
            'numpy',
            'scipy',
            'toml',
            'zarr',
        ],
        packages=[
                'evaluateInstanceDetection',
        ]
)
