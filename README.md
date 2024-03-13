# xSemAD: Explainable Semantic Anomaly Detection in Event Logs Using Sequence-to-Sequence Models
This repository contains the prototype of the approach and evaluation as described in xSemAD: Explainable Semantic Anomaly Detection in Event Logs Using Sequence-to-Sequence Models. Under submission for the 2024 International Conference on Business Process Management (BPM 2024). 

## About the project
The identification of undesirable behavior in event logs is an important aspect of process mining that is often addressed by anomaly detection methods. 
Traditional anomaly detection methods tend to focus on statistically rare behavior and neglect the subtle difference between rarity and undesirability. The introduction of semantic anomaly detection has opened a promising avenue by identifying semantically deviant behavior.
This work addresses a gap in semantic anomaly detection, which typically indicates the occurrence of an anomaly without explaining the nature of the anomaly. We propose xSemAD, an approach that uses a sequence-to-sequence (seq2seq) model to go beyond pure identification and provides extended explanations. In essence, our approach learns constraints from a given process model repository and then checks whether these constraints hold in the considered event log. This approach not only helps understand the specifics of the undesired behavior, but also facilitates targeted corrective actions.
Our experiments demonstrate the effectiveness of the proposed approach by showing that it is able to generate relevant constraints and outperform existing state-of-the-art semantic anomaly detection methods. This advance promises to refine the interpretability and practical utility of semantic anomaly detection in process mining and provide a more comprehensive solution for detecting and understanding undesirable behavior.

### Built with
* ![platform](https://img.shields.io/badge/platform-linux-brightgreen)
* ![GPU](https://img.shields.io/badge/GPU-2%20x%20Nvidia%20RTX%20A6000-red)
* ![python](https://img.shields.io/badge/python-black?logo=python&label=3.8.13)
* ![python](https://img.shields.io/badge/python-black?logo=python&label=3.7.16)

## Requirements
### To apply our approach (xSemAD)
Use at least ![python](https://img.shields.io/badge/python-black?logo=python&label=3.8.13)
1. clone this project <code>git clone</code> to get the repository
2. install the requirements in the constraints-transformer folder with 
```sh
pip install -r requirements.txt
```
Make sure to adapt all path file names to your needs.
### To apply SVM or BERT from Caspary et al. 2023
Use at least ![python](https://img.shields.io/badge/python-black?logo=python&label=3.7.16)
1. clone [this](https://gitlab.uni-mannheim.de/processanalytics/ml-semantic-anomaly-dection) project <code>git clone</code> to get the repository
2. install the requirements in the ml-semantic-anomaly-detection folder with 
```sh
pip install -r requirements.txt
```
or 
```sh
pip install -r requirements_gpu.txt
```
Look [here](https://gitlab.uni-mannheim.de/processanalytics/ml-semantic-anomaly-dection) for more details.
Add the files from the Caspary2023 folder to that repository. Make sure to adapt all path file names to your needs.


## Project Organization
    ├── caspary2023                                      <- Source code for SVM/BERT.
    │   ├ 00_paper_generate_logs.py                      <- Script to generate event logs
    │   ├ 01_paper_generate_pairs_kb.py                  <- Script to generate the knowledgebase 
    │   ├ 02_paper_generate_noisy_test_logs.py           <- Script to noisy test event logs 
    │   ├ 03_paper_train_models_run_eval.py              <- Train the models 
    │   ├ 04_paper_predict_test_main_bert.py             <- Script to generate bert predictions on testset 
    │   └ 04_paper_predict_test_main_svm.py              <- Script to generate svm predictions on testset 
    ├── constraints-transformer
    │   ├── conversion                                   <- Contains bpmn analyzer, json2petrinet, and petrinet analyzer.    
    │   ├── evaluation                                   <- contains utility functions for evaluation.
    │   ├── labelparser                                  <- contains utility functions for label parsing.
    │   ├── results                                      <- Figures for the paper.
    │   ├ 00_run_preprocess_sapsam.py                    <- Script to preprocess the sapsam dataset.
    │   ├ 01_run_paper_preprocessing_data.py             <- Script to generate train,test, and validatrion set
    │   ├ 02_run_training.py                             <- Script to fine-tune FLAN-T5 
    │   ├ 04_run_generate_predictions_testset.py         <- Script to generate xSemAD predictions on testset 
    │   ├ 04_run_generate_predictions_validationset.py   <- Script to generate xSemAD predictions on validationset 
    │   ├ 08_run_declare_miner.py                        <- Script to run declare miner
    │   ├ 09_run_minerful.py                             <- Script to run MINERful
    │   ├ 10_paper_plots.ipynb                           <- Script to generate the PAPER PLOTS
    │   ├ 10_paper_paper_results.ipynb                   <- Script go generate the PAPER RESULTS
    │   ├ config.py                                      <- config file 
    │   └ requirements.txt                               <- requirements file
    ├── README.md                                        <- The top-level README for users of this project.
    └── LICENSE                                          <- License that applies to the source code in this repository.
    

## Contact


## Find a bug?
If you found an issue or would like to submit an improvement to this project, please contact the authors. 



