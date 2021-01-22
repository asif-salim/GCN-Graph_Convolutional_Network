This is a TensorFlow implementation of the paper 'Framework for Designing Filters of Spectral Graph Convolutional Neural Networks in the context of Regularization Theory", (https://arxiv.org/abs/2009.13801)

The implementation is an extension of the implementation codes of the paper by Thomas N. Kipf, Max Welling, [Semi-Supervised Classification with Graph Convolutional Networks](http://arxiv.org/abs/1609.02907) (ICLR 2017)

 ## Installation

```bash
python setup.py install
```

## Requirements
* tensorflow (>0.12) (Programming is done with version 1.9.0 in Anaconda environment in Windows OS)
* networkx

## Run the demo

```bash
cd gcn
python train.py  
``` 

## Results
-----------------------------------------------------------------
Model     |      CORA       |     CITESEER    |     PUBMED      |
-----------------------------------------------------------------
ManiReg   |  59.5           |  60.1           |  70.7           |
SemiEmb   |  59.0           |  59.6           |  71.1           |
LP        |  68.0           |  45.3           |  63.0           |
DeepWalk  |  67.2           |  43.2           |  65.3           |
ICA       |  75.1           |  69.1           |  73.9           |
Planetoid |  75.7           |  64.7           |  77.2           |
MLP       |  56.2           |  57.1           |  70.7           |
-----------------------------------------------------------------    
GCN       |  81.78 +- 0.64  |  70.73 +- 0.53  |  78.48 +- 0.58  |
IGCN      |  80.49 +- 1.58  |  68.86 +- 1.01  |  77.87 +- 1.55  |
ChebyNet  |  82.16 +- 0.74  |  70.46 +- 0.70  |  78.24 +- 0.43  |
GraphHeat |  81.38 +- 0.69  |  69.90 +- 0.50  |  75.64 +- 0.64  |
-----------------------------------------------------------------
Diffusion |  83.12 +- 0.37  |  71.17 +- 0.43  |  79.20 +- 0.36  |
1-step RW |  82.36 +- 0.34  |  71.05 +- 0.34  |  78.74 +- 0.27  |
2-step RW |  82.51 +- 0.22  |  71.18 +- 0.59  |  78.64 +- 0.20  |
3-step RW |  82.56 +- 0.24  |  71.21 +- 0.63  |  78.28 +- 0.36  |
Cosine    |  75.53 +- 0.52  |  67.29 +- 0.64  |  75.52 +- 0.53  |
-----------------------------------------------------------------

Commands to reproduce the results reported in the manuscript:
## CORA dataset
GCN      : python train.py --dataset cora --model gcn --hidden1 16 
ChebyNet : python train.py --dataset core --model gcn_cheby --hidden1 16 --max_degree 2
GraphHeat: python train.py --dataset cora --model graphheat --hidden1 64 --max_degree 2 --s 1
Diffusion: python train.py --dataset cora --model diffusion --hidden1 64 --max_degree 4 --s 1.1
1-step RW: python train.py --dataset cora --model psteprandomwalk --hidden1 64 --p 1 --a 2.9
2-step RW: python train.py --dataset cora --model psteprandomwalk --hidden1 64 --p 2 --a 10.1
3-step RW: python train.py --dataset cora --model psteprandomwalk --hidden1 64 --p 3 --a 22
cosine   : python train.py --dataset cora --model cosine --hidden1 32 --max_degree 1

## CITESEER dataset
GCN      : python train.py --dataset citeseer --model gcn --hidden1 16 
ChebyNet : python train.py --dataset citeseer --model gcn_cheby --hidden1 64 --max_degree 1
GraphHeat: python train.py --dataset citeseer --model graphheat --hidden1 32 --max_degree 2 --s 1
Diffusion: python train.py --dataset citeseer --model diffusion --hidden1 32 --max_degree 3 --s 1.1
1-step RW: python train.py --dataset citeseer --model psteprandomwalk --hidden1 32 --p 1 --a 2.7
2-step RW: python train.py --dataset citeseer --model psteprandomwalk --hidden1 32 --p 2 --a 8.3
3-step RW: python train.py --dataset citeseer --model psteprandomwalk --hidden1 32 --p 3 --a 18
cosine   : python train.py --dataset citeseer --model cosine --hidden1 64 --max_degree 1

## PUBMED dataset
GCN      : python train.py --dataset pubmed --model gcn --hidden1 16 
ChebyNet : python train.py --dataset pubmed --model gcn_cheby --hidden1 64 --max_degree 1
GraphHeat: python train.py --dataset pubmed --model graphheat --hidden1 64 --max_degree 2 --s 1
Diffusion: python train.py --dataset pubmed --model diffusion --hidden1 64 --max_degree 3 --s 0.9
1-step RW: python train.py --dataset pubmed --model psteprandomwalk --hidden1 64 --p 1 --a 3.7
2-step RW: python train.py --dataset pubmed --model psteprandomwalk --hidden1 32 --p 2 --a 10.8
3-step RW: python train.py --dataset pubmed --model psteprandomwalk --hidden1 64 --p 3 --a 12
cosine   : python train.py --dataset pubmed --model cosine --hidden1 32 --max_degree 1

## Note:
The architecture used in the commands is GCN with 2 layers of Graph Convolution(GC). To run the other architectures, change the "arch" variable in line 33 of train.py to
a) arch = GCN_1_layer   #-- for model with only one layer of GC.
b) arch = GCN_3_layer   #-- for model with three layers of GC.
c) arch = GCNDense      #-- for model with one layer GC followed by a dense layer.
d) arch = GCNDenseDense #-- for model with one layer GC followed by two layers of dense layers.




 
