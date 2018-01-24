# -*- coding: utf-8 -*-

desc = """\
pyTrigger - software trigger for data acquisition (with ring buffer).

For a showcase see: https://github.com/ladisk/pyTrigger/blob/master/Showcase%20-%20pyTrigger.ipynb
=============
"""

from setuptools import setup, Extension
setup(name='pyTrigger',
      version='0.12',
      author='Janko Slavič',
      author_email='janko.slavic@fs.uni-lj.si',
      url='https://github.com/ladisk/pyTrigger',
      py_modules=['pyTrigger'],
	  description='Software trigger for data acquisition (with ring buffer).',
      long_description=desc,
      install_requires=['numpy']
      )