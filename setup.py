from setuptools import setup, find_packages

setup(
    name='cleanframe',
    version='0.1.0',
    author='Fayez Hesham',
    author_email='fayezhesham510@gmail.com',
    description='A Python library for cleaning and validating pandas DataFrames.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/fayezhesham/cleanframe',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Framework :: Pandas',
    ],
    python_requires='>=3.7',
    install_requires=[
        'pandas>=1.0.0',
    ],

    # package_data={
    #     'cleanframe': ['data/*.csv'],
    # },
    # include_package_data=True, # Set to True to include data files specified in MANIFEST.in
)