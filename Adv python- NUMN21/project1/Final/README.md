# labs_fmnn25

## Data

- load splitted data
- data is clean
- Data loader
  - shuffle epoch iterator
  - random sampling

## Feed Forward Network

Layer class

- Weight, bias, activation function
- Initialization
- Forward computation

Network class

- Allows choosing layer sizes, activation and loss functions
- Backpropagation
- SGD, Weight update

Activation & Loss functions

- forward computation
- backwards/derivatives

## Hyperparameters

- Layer sizes
- Learning rate

## Attack

- FGSM

## Contributions

- Dataloading with shuffle by Asifa and Jonathan
- Network was implemented separately, and then strengths of each solution was combined.
  - Weight and bias initialization by Jonathan
  - Layer class structure by Marcus
- Attack and hyperparmeter study class by Asifa.
- Test by Joel (and Marcus)
