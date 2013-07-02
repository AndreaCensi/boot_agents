import os
from setuptools import setup, find_packages

version = "1.2dev1"

description = """"""

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

long_description = read('README.rst')


setup(name='BootAgents',
      author="Andrea Censi",
      author_email="andrea@cds.caltech.edu",
      url='http://github.com/AndreaCensi/boot_agents',

      description=description,
      long_description=long_description,
      keywords="",
      license="",

      classifiers=[
        'Development Status :: 4 - Beta',
      ],

	    version=version,
      download_url='http://github.com/AndreaCensi/boot_agents/tarball/%s' % version,

      package_dir={'':'src'},
      packages=find_packages('src'),
      install_requires=[
        # 'scikits.image', # TODO: put conditional
        'astatsa',
        'BootOlympics>=1.0,<2'],
      setup_requires=['nose>=1.0'],
      tests_require=['nose>=1.0', 'rudolf', 'nose-progressive', 'nose-cov'],

       entry_points={
         'console_scripts': [
           'boot_agents_diff_actions = '
                'boot_agents.diffeo.analysis.main:main',
            ]
    }
)

