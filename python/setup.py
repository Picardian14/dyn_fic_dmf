from setuptools import setup, Extension
import os 
eigen_path = os.path.abspath('cpp/Eigen')

<<<<<<< HEAD
ext = Extension(
    '_DYN_FIC_DMF',
    libraries=['boost_python310', 'boost_numpy310'],
    sources=['fastdyn_fic_dmf/DYN_FIC_DMF.cpp'],
    include_dirs=[eigen_path]  # Add this line
)
=======
ext = Extension('_DYN_FIC_DMF',
                libraries = ['boost_python310', 'boost_numpy310'],
                sources   = ['fastdyn_fic_dmf/DYN_FIC_DMF.cpp'])

>>>>>>> a90fb3f94b86ce5dab55a458ae1f78206e758cf8
setup(name              = 'fastdyn_fic_dmf',
      version          = '0.1',
      description      = 'Fast Dynamic Mean Field simulator of neural dynamics',
      author           = 'Pedro A.M. Mediano',
      author_email     = 'pam83@cam.ac.uk',
      url              = 'https://gitlab.com/concog/fastdmf',
      long_description = open('../README.md').read(),
      package_data     = {'python/fastdyn_fic_dmf': ['DTI_fiber_consensus_HCP.csv']},
      install_requires = ['numpy'],
      ext_modules      = [ext],
      packages         = ['fastdyn_fic_dmf'])

