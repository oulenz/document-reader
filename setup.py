from setuptools import setup, find_packages

setup(
    name='document_scanner',
    version='0.6.0',
    description='Extract content from a filled out document',
    url='https://github.com/epigramai/document_scanner',
    author='Oliver',
    author_email='oliver@epigram.ai',
    packages=find_packages('.')
    # TODO install_requires=['tfwrapper>=0.1.0-rc5']
)