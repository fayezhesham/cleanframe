from setuptools import setup, find_packages
import io

with io.open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
setup(
    name='cleanframe',
    version='0.2.4',
    author='Fayez Hesham',
    author_email='fayezhesham510@gmail.com',
    description='A Python library for cleaning and validating pandas DataFrames.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/fayezhesham/cleanframe',
    include_package_data=True,
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.7',
    install_requires=[
        'pandas>=1.0.0',
    ],


)