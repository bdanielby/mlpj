import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


github_url = 'https://github.com/bdanielby/mlpj'

setup(
    name='mlpj',
    version='0.2.1',
    description='Tools for machine learning projects',
    long_description=read('README.md'),
    long_description_content_type='text/markdown',
    url=github_url,
    author='Bruno Daniel',
    license='MIT',
    packages=['mlpj'],
    package_data={'mlpj': ['result_template.html']},
    include_package_data=True,
    install_requires=[
        'numpy', 'pandas', 'matplotlib', 'numba', 'lockfile', 'jinja2',
        'markupsafe',
    ],
    project_urls={
        'Homepage': github_url,
        'Source': github_url,
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
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
