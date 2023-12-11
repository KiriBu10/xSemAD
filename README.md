# xSemAD: Explainable Semantic Anomaly Detection in Event Logs Using Sequence-to-Sequence Models
This repository contains the prototype of the approach, evaluation, and data as described in
Busch, K., Kampik, T., & Leopold, H. xSemAD: Explainable Semantic Anomaly Detection in Event Logs Using Sequence-to-Sequence Models. Under submission for the 2024 International Conference on Advanced Information Systems Engineering (CAISE 2024). 

## About the project
The identification of undesirable behavior in event logs is an important aspect of process mining that is often addressed by anomaly detection methods.  Traditional anomaly detection methods tend to focus on statistically rare behavior and neglect the subtle difference between rarity and undesirability. The introduction of semantic anomaly detection has opened a promising avenue by identifying semantically deviant behavior. This work addresses a gap in semantic anomaly detection, which typically indicates the occurrence of an anomaly without explaining the nature of the anomaly. We propose xSemAD, an approach that uses a sequence-to-sequence model to go beyond pure identification and provides extended explanations. Our model generates constraints from an event log and compares them with the observed event log. This approach not only helps understand the specifics of the undesired behavior, but also facilitates targeted corrective actions. Our experiments demonstrate the effectiveness of the proposed approach by showing that it is able to generate relevant constraints and outperform existing state-of-the-art semantic anomaly detection methods. This advance promises to refine the interpretability and practical utility of semantic anomaly detection in process mining and provide a more comprehensive solution for detecting and understanding undesirable behavior.

### Built with
* ![platform](https://img.shields.io/badge/platform-linux-brightgreen)
* ![GPU](https://img.shields.io/badge/GPU-2%20x%20Nvidia%20RTX%20A6000-red)
* ![python](https://img.shields.io/badge/python-black?logo=python&label=3.8.13)
* ![python](https://img.shields.io/badge/python-black?logo=python&label=3.7.16)

## Requirements
### To apply our approach (xSemAD)
Use at least ![python](https://img.shields.io/badge/python-black?logo=python&label=3.8.13)
1. clone this project <code>git clone</code> to get the repository
2. install the requirements with 
```sh
pip install -r requirements.txt
```
### To apply SVM or BERT from Caspary et al. 2023
Use at least ![python](https://img.shields.io/badge/python-black?logo=python&label=3.7.16)
1. clone this project <code>git clone</code> to get the repository
2. install the requirements in the ml-semantic-anomaly-detection folder with 
```sh
pip install -r requirements.txt
```
or 
```sh
pip install -r requirements_gpu.txt
```
Look [here](https://gitlab.uni-mannheim.de/processanalytics/ml-semantic-anomaly-dection) for more details.


## Project Organization



## Contact

Kiran Busch - kiran.busch@klu.org

## Find a bug?
If you found an issue or would like to submit an improvement to this project, please contact the authors. 


# Date of release
The completion of the repository is scheduled for December 11th, 2023.

