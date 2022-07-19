# Cascading Residual Block for Multi-Behavior Recommender System

## Python Scripts
<code>Params.py</code> Model parameters

<code>DataHandler.py</code> The script used to handle the data

<code>CRG_multiTask_inner.py</code> The new model proposed by us with 1) LightGCN at each block 2) cascading residual blocks 3) multi-task loss function

<code>LightGCN.py</code> A vanilla version of LightGCN Model with "buy" as the target behavior

<code>CRG_multiTask.py</code> The MB-CRB model with a fully connected layer after the inner product at the final prediction stage -> this scripts prove that the FC layer at prediction stage is of no use.

<code>CRG_targetTaskLossFuncOnly.py</code> The MB-CRB model w/o multi-task loss function (only the loss for the target behavior is considered)
