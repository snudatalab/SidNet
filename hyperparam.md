
## Overview 
In this document, we briefly summarize the information on hyperparameters of competitors of SidNet, and describe how we searched for their values in the sign prediction task. 


### SRWR
* Model hyperparameters
    - c: restart probability (c is fixed to 0.15 as in [1])
    - beta: balance attenuation factor for "the enemy of my enemy is my friend"
    - gamma: balance attenuation factor for "the friend of my enemy is my friend"
  
* Searched ranges for hyperparameters
  -  beta: grid search in [0, 1] by 0.1
  -  gamma: grid search in [0, 1] by 0.1 
  
* Searched values of hyperparameters

| **Hyperparameter** | **Bitcoin-Alpha** | **Bitcoin-OTC** | **Wikipedia** | **Slashdot** | **Epinions** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `beta` | 0.8 | 0.9 | 0.6 | 0.7 | 0.8 |
| `gamma` | 0.6 | 0.7 | 1 | 1 | 0.9 |

### APPNP
* Model hyperparameters
  - K: number of diffusion steps (K is fixed to 10 as in [2] - due to its convergence property, its predictive performance tends to converge as K increases)
  - alpha: decaying ratio
    
* Searched ranges for hyperparameters
  -  alpha: grid search in [0, 1] by 0.1

* Searched values of hyperparameters

| **Hyperparameter** | **Bitcoin-Alpha** | **Bitcoin-OTC** | **Wikipedia** | **Slashdot** | **Epinions** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| `alpha` | 0.2 | 0.3 | 0.4 | 0.3 | 0.4 |

### ResGCN
ResGCN is a network consisting of multiple GCN layers which are consecutively tied with residual connections as proposed in [3].

* Hyperparameters
* Model hyperparameters
  - L: number of GCN layers
    
* Searched ranges for hyperparameters
  -  L: grid search in [1, 10] by 1

* Searched values of hyperparameters

| **Hyperparameter** | **Bitcoin-Alpha** | **Bitcoin-OTC** | **Wikipedia** | **Slashdot** | **Epinions** |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | `L` | 7 | 8 | 6 | 3 | 4 |

### SIDE
* Model hyperparameters
  - w: number of walks per node (w is fixed to 80 as in [4])
  - l: number of steps per walk (l is fixed to 40 as in [4])
  - n: noise sampling size (n is fixed to 20 as in [4])
  - k: context size

* Searched ranges for hyperparameters
  -  k: grid search in [1, 3, 5, 7]

* Searched values of hyperparameters
  
  | **Hyperparameter** | **Bitcoin-Alpha** | **Bitcoin-OTC** | **Wikipedia** | **Slashdot** | **Epinions** |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | `k` | 3 | 5 | 3 | 7 | 5 |
    
### SLF
* Model hyperparameters
  - p0: effect of no feedback (p0 is fixed to 0.001 as in [5])
  - n: sample size of null social relationships
  
* Searched ranges for hyperparameters
  -  n: [0, 5, 10, 20, 50, 100] 

* Searched values of hyperparameters
  
  | **Hyperparameter** | **Bitcoin-Alpha** | **Bitcoin-OTC** | **Wikipedia** | **Slashdot** | **Epinions** |
  | :---: | :---: | :---: | :---: | :---: | :---: |
  | `n` | 20 | 50 | 50 | 20 | 100 |


### SGCN
* Model hyperparameters
  - L: number of SGCN layers (L is fixed to 2 as in [6] - after L=2, its performance degrades)
  - lambda: weight of extended structural balance theory
    
* Searched ranges for hyperparameters
  -  lambda: [0, 3, 5, 7, 10]
    
* Searched values of hyperparameters

  | **Hyperparameter** | **Bitcoin-Alpha** | **Bitcoin-OTC** | **Wikipedia** | **Slashdot** | **Epinions** |
    | :---: | :---: | :---: | :---: | :---: | :---: |
  | `lambda` | 5 | 5 | 7 | 3 | 5 |

### SNEA
* Model hyperparameters
  - L: number of SGCN layers (L is fixed to 2 as in [7])
  - lambda: weight of extended structural balance theory

* Searched ranges for hyperparameters
  -  lambda: [0, 1, 2, 3, 4, 5, 6]

* Searched values of hyperparameters

  | **Hyperparameter** | **Bitcoin-Alpha** | **Bitcoin-OTC** | **Wikipedia** | **Slashdot** | **Epinions** |
      | :---: | :---: | :---: | :---: | :---: | :---: |
  | `lambda` | 4 | 4 | 5 | 3 | 6 |

### Optimizer Information
All methods except for SRWR used the logistic classifier for the sign prediction task at the final stage, which are trained by the Adam optimizer with the following parameters:
  - `weight-decay`: 1e-3
  - `learning-rate`: 0.01
  - `epochs`: 100


### References
[1] Jung, J., Jin, W., & Kang, U. (2020). Random walk-based ranking in signed social networks: model and algorithms. Knowledge and Information Systems, 62(2), 571-610.

[2] Klicpera, J., Bojchevski, A., & Günnemann, S. (2018, September). Predict then Propagate: Graph Neural Networks meet Personalized PageRank. In International Conference on Learning Representations.o

[3] Li, G., Muller, M., Thabet, A., & Ghanem, B. (2019). Deepgcns: Can gcns go as deep as cnns?. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 9267-9276).

[4] Kim, J., Park, H., Lee, J. E., & Kang, U. (2018, April). Side: representation learning in signed directed networks. In Proceedings of the 2018 World Wide Web Conference (pp. 509-518).

[5] Xu, P., Hu, W., Wu, J., & Du, B. (2019, July). Link prediction with signed latent factors in signed social networks. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (pp. 1046-1054).

[6] Derr, T., Ma, Y., & Tang, J. (2018, November). Signed graph convolutional networks. In 2018 IEEE International Conference on Data Mining (ICDM) (pp. 929-934). IEEE.

[7] Li, Y., Tian, Y., Zhang, J., & Chang, Y. (2020, April). Learning signed network embedding via graph attention. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 34, No. 04, pp. 4772-4779).