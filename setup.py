import os
from setuptools import setup, find_packages

__version__ = '1.0.0'
cwd = os.path.dirname(os.path.abspath(__file__))
# requirements = open(os.path.join(cwd, "requirements.txt"), "r").readlines()

setup(
    name='maha_tts',
    version=__version__,

    url='https://github.com/dubverse-ai/MahaTTS/tree/main',
    author='Dubverse AI',
    author_email='jaskaran@dubverse.ai',
    install_requires = [
        'einops',
        'transformers',
        'unidecode',
        'inflect'
    ],
    packages=find_packages(),
    py_modules=['maha_tts'],
)