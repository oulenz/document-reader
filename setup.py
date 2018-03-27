from setuptools import setup, find_packages

setup(
    name='document-reader',
    version='0.8.0',
    description='Extract content from a document',
    url='https://github.com/epigramai/document_reader',
    author='Oliver',
    author_email='oliver@epigram.ai',
    packages=find_packages('.')
    # TODO install_requires=
)