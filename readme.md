# Use Random Selection for Now: Investigation of Few-Shot Selection Strategies in LLM-based Text Augmentation for Classification
This repository contains both the data and code for this paper. This repository is structured as follows:

**datasets** - contains folder for each of the datasets with the preprocessed train and test data, collected data for each LLM and random seed, and classifation training results for both in-distribution and out-of-distribution data.

**mistral_collect_scripts** - contains scripts for collecting data via sample selection strategies for *all* LLMs used 

**finetuning_scripts** -  contains scripts for finetuning classifiers for each of the cases mentioned in the paper itself.

**reqs.txt** - contains the python pip requirements for this project

## Citing

```
@misc{cegin2024userandomselectionnow,
      title={Use Random Selection for Now: Investigation of Few-Shot Selection Strategies in LLM-based Text Augmentation for Classification}, 
      author={Jan Cegin and Branislav Pecher and Jakub Simko and Ivan Srba and Maria Bielikova and Peter Brusilovsky},
      year={2024},
      eprint={2410.10756},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.10756}
}
```
