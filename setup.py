from setuptools import setup, find_packages
from typing import List

HYPNENATE = '-e .'
def get_requirements(file_path:str)->List[str]:
    requirments=[]
    with open(file_path) as file_obj:
        requirments=file_obj.readlines()
        requirments=[req .replace("\n","") for req in requirments]
        if HYPNENATE in requirments:
            requirments.remove(HYPNENATE)
    return requirments

setup(
    name='ML Project',
    version='0.1.0',
    author='sasidhar',
    author_email='sasidharreddy2105@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)