# Modeling Mutual Intelligibility between Slavic Language Pairs
*by Tyler Renslow*

This repository contains all scripts and data used for my master's thesis.

The repository structure loosely follows that of the Team Data Science Process developed by Microsoft. More info can be found [here](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/overview).

## Software Dependencies
All scripts were written in Python 3, with additional packages used for different tasks.

Packages for data processing:
- [NLTK](https://www.nltk.org/install.html)
- [wikipedia](https://pypi.org/project/wikipedia/)
- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)

Packages for modeling:
- [TensorFlow](https://www.tensorflow.org/install) (code was written when v1.7 was latest, may be broken now)
- For training TensorFlow models on NVIDIA GPUs, follow the instructions at this [link](https://www.tensorflow.org/install/gpu).

TODO:

- refactor all paths in scripts
- check compatability with latest TensorFlow version
- find more efficient way to store scraped wikipedia articles, with the goal of making them easier to share and process
- store all large data in compressed files to save disk space
- reformat log files in a smart way to reflect which feature set used to train model

