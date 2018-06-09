import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='tensionflow',
    version='0.0.1',
    author='Colin Fahy  ',
    author_email='colin@cfahy.com',
    description='Tensorflow framework for working with audio data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Curly-Mo/tensionflow',
    packages=setuptools.find_packages(),
    python_requires='>=3.4',
    classifiers=(
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ),
    install_requires=[
        'bidict ~= 0.14.2',
        'librosa ~= 0.6.1',
        'numpy ~= 1.14.3',
        'pandas ~= 0.23.0',
        'tensorflow ~= 1.8.0',
    ],
    extras_require={
        'test': [
            'tox>=3.0.0',
            'python-coveralls>=2.9.1',
            'pytest>=3.6.1',
            'pytest-cov>=2.5.1',
            'codecov>=2.0.15',
            'flake8>=3.5.0',
        ],
    }
)
