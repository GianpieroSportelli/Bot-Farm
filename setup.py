#!/usr/bin/env python

from distutils.core import setup

setup(name='Bot Farm',
      version='0.1.0',
      description='Bot Farm is a framework for build your intelligent telegram bot',
      author='Gianpiero Sportelli',
      author_email='sportelligianpiero@gmail.com',
      packages=['bot_farm', 'bot_farm.bert'],
      install_requires=["pip>=20.0","requests", "beautifulsoup4", "tensorflow>=2.0", "pandas>=1.0.1", "nlpaug",
                        "matplotlib", "nltk", "tensorflow_hub>=0.7","telepot","datetime","tensorflow_model_optimization"]
      )
