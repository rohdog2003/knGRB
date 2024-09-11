from setuptools import setup

setup(
    name='knGRB',
    version='1.0.0',    
    description='Python modules for calculating the spectra, electron distribution, and Compton Y parameter of Gamma Ray Burst afterglows including KN effects.',
    url='https://github.com/rohdog2003/knRGB',
    author='Coleman Rohde',
    author_email='rohdog2003@gmail.com',
    license='GNU GPLv3',
    packages=['knGRB'],
    install_requires=['mpi4py>=2.0',
                      'numpy',                     
                      'scipy'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU GPLv3 License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)