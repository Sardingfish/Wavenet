## Example graphical outputs

![Filter graph](https://github.com/asogaard/Wavenet/blob/master/media/FilterGraph.png "Filter graph.")

### Configuration
* __Training data__: NeedleGenerator
* __Number of filter coefficients__: 2

### Description
The graphical output in this folder is produced using Example03, with input from Example02. These plots should give an indication of the wavenet training that the package facilitates.

* [./exampleInputs/](./exampleInputs/): Folder containing the 10 training examples used in generating the cost map shown in [FilterGraph.pdf](FilterGraph.pdf).
* [CostGraph.pdf](CostGraph.pdf): Graphs of the combined (regularisation and sparsity) cost as functions of the number of training updates, one for each random initialisation.
* [FilterGraph.pdf](FilterGraph.pdf): The evolution of the filter coefficient configurations, one for each random initialisation, overlayed on the cost map in two-dimensional filter coefficient space. Red markers are initial configurations (generated on the unit circle); blue markers are final configurations. Contours indicate regions of similar cost, computed for the inputs in [./exampleInputs/](./exampleInputs/).
* [FilterGraph.png](FilterGraph.png): Same as [FilterGraph.pdf](FilterGraph.pdf), but in different format.
* [Orthonormality.pdf](Orthonormality.pdf): Distributions of inner products for all basis functions in the best, final basis<sup>1</sup>. Should be peaked around 0 and 1.
* [bestBasis_1D.mov](bestBasis_1D.mov): Movie showing the evolution of the best basis<sup>1</sup> in one dimension.
* [bestBasis_2D.mov](bestBasis_2D.mov): Movie showing the evolution of the best basis<sup>1</sup> in two dimensions.

---

<sup>1</sup> By "best basis" we mean the basis, or initialisation i.e. evolution of bases, which has the lowest cost at the end of the training.