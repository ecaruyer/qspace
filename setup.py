from distutils.core import setup
import os


setup(name='qspace',
    description='Reconstruction tools for continuous q-space diffusion MRI.',
    version='0.1.dev',
    author='Emmanuel Caruyer',
    author_email='caruyer@gmail.com',
    url='http://www.emmanuelcaruyer.com/',
    scripts = [os.path.join('scripts', 'mspf_fit'),
               os.path.join('scripts', 'multishell')],
    package_data={'qspace.sampling' : ["data/jones_*.txt"]},
    packages=['qspace', 'qspace.bases', 'qspace.sampling', 'qspace.visu'])
