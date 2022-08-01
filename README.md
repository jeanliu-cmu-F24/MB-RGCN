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
 
## Files structure
1. python scripts

a. `MB_RGCN.py`: This is the main structure of our MB-RGCN model, where we define the MB_RGCN recommender's class
b. ``:

## How to use the recommender model
1. Create a new folder namede `History` in your working directory

```bash
mkdir History
```

2. Run the `MB_RGCN.py` script
- Beibei
```bash
python MB_RGCN.py --gcn_list 3 3 3
```

- Tmall
```bash
python MB_RGCN.py --gcn_list 4 4 4 4 --data tmall
```

### Parameters
- *gcn_list*: The gcn_list argument cannot be skipped. You should use the long option argument `--gcn_list`, followed by a sequence of integers with space as the delimiter. The sequence of integers stands for the # of layers at **each behavior block**
  - for Beibei dataset, a integer sequence with a length of 3 shall be given, since the dataset comes with 3 behavior blocks by default
  - for Tmall dataset, a integer sequence with a length of 4 shall be given, since the dataset comes with 4 behavior blocks by default

## References
- The main body of our code (The recommender constructor, train and test pipeline, and the data handler) refers to the code by 
- The implementation of the LightGCN layer refers to the official open source code of the original LightGCN paper.
