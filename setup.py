# -*- coding: utf-8 -*-

desc = """\
pyTrigger - a software trigger typically used with DAQ systems

For a showcase see: https://github.com/ladisk/pyTrigger/blob/master/Showcase%20-%20pyTrigger.ipynb
=============
"""

from setuptools import setup, Extension
setup(name='pyTrigger',
      version='0.11',
      author='Janko Slavič',
      author_email='janko.slavic@fs.uni-lj.si',
      url='https://github.com/ladisk/pyTrigger',
      py_modules=['pyTrigger'],
      long_description=desc,
      install_requires=['numpy']
      )