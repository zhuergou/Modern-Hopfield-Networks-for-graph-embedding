
## Installation
1. Download the software package

2. Install software dependencies
- numpy
- networkx
- sklearn

3. Install Netlsd from https://github.com/xgfs/NetLSD


## Usage

### Graph classification with coarse graphs.

`main_classification.py` contains the experimental codes for graph classification for coarse graphs. 
The basic usage is 

```
python main_classification.py
```

Parameter options are

-dataset: MUTAG, ENZYMES, NCI1, NCI109, PROTEINS, PTC

-method: mgc, sgc, hopfield

-ratio, the ratio between coarse and original graphs n/N

The default setting is 
```
python main_classification.py --dataset MUTAG --method hopfield --ratio 0.2
```







