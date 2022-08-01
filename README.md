# MB-RGCN

## Introduction
**M**ulti **B**ehavior - **R**esidual **G**raph **C**onvolutional **N**etwork (**MB-RGCN**) explicitly utilizes the ordinal relationship across behaviors with a lightweight model structure. Particularly, a chain of behavior blocks is used to explicitly encode the cascading behavior relationship with residual connections. In addition, LightGCN is applied to capture features for each behavior block, and a multi-task learning mechanism is leveraged to facilitate the model learning. 

## Environment
The MB-RGCN model is implemented under the following development environment:
- Python 3.6
- tensorflow==1.14.0
- scipy==1.5.4
- CUDA==11.0 
- numpy==1.19.5

## Datasets
- **Raw data**
  - Tmall: https://tianchi.aliyun.com/dataset/dataDetail?dataId=649
  
- **Preprocessed data**

  The preprocessed data can be found in the `/datasets` folder
