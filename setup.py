from setuptools import setup

setup(
    name='mlpj',
    version='0.1.0',    
    description='Tools for machine learning projects',
    url='https://github.com/bdanielby/mlpj',
    author='Bruno Daniel',
    license='MIT',
    packages=['mlpj'],
    install_requires=[
        'numpy', 'pandas', 'matplotlib'
    ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Manufacturing',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
    ],
)
