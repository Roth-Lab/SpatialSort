# SpatialSort
A spatially aware Bayesian clustering approach that allows for the incorporation of prior biological knowledge.

## Installation
1. Clone this repo
```commandline
git clone https://github.com/Roth-Lab/SpatialSort.git
```
2. Move into the repo and install SpatialSort
```commandline
pip install .
```
3. Run SpatialSort
```commandline
SpatialSort infer --exp-csv 'sample_exp.csv' --loc-csv 'sample_loc.csv' --rel-csv 'sample_rel.csv'
```

## Quick Documentation
### Required Inputs
Running SpatialSort requires at least three files which all have a common structure and two parameters. Let's talk about them one by one.
1. Expression Matrix: this is a cell by expression marker matrix. The first column of the matrix should indicate the patient id of the cell, not the cell id. An example is given in the `example\` folder. We input the csv file name as the following:
   * `--exp-csv 'sample_exp.csv'` : the file path of the csv file holding expression data.
2. Location Matrix: this is a cell by x,y coordinates matrix indicating cellular location. The first column of the matrix should indicate the patient id of the cell. The rest of the two columns are the X and Y coordinates of the cell. We input the csv file name as the following:
   * `--loc-csv 'sample_loc.csv'` : the file path of the csv file holding location data.
3. Relation Matrix: this is a matrix of cell relations indicating which cells are linked together in the patient-specific neighbour graph. The first column of the matrix should indicate the patient id of the cell. We input the csv file name as the following:
   * `--rel-csv 'sample_rel.csv'` : the file path of the csv file holding cell neighbour relations data.
4. `-k / --num-clusters` : this is the number of clusters to initialize clustering.
5. `-o / --out_dir`: this is the output directory to save results.

### Additional and Optional Inputs
6. `-t" / --num-iters`: this is the number of iterations to run inference, default at 500.
7. `-s / --seed`: this is a seed for random numbers, default is at random.
8. Prior Expression Matrix: this is a cluster by marker prior expression matrix with entries of either -1, 0, 1, 2 indicating a prior expectation of unknown, low, mid, high expression of a specific marker for each cluster.
   * `-p / --prior-csv`: the  file path of the csv file holding a cluster by marker prior expression matrix.
9. Anchor Matrix: this is a labeled cell by expression marker anchor matrix. 
   * `-a / --anchor-csv`: the file path of the csv file holding an anchor matrix. The first column of the matrix should indicate the word "known" repeated by the total number of cells. The last column name MUST be named as "label" and include all labels zero-based indexed.
10. `-l / --prec-scale`: the precision scale parameter for each marker, smaller the stronger, default at 0.1.

### Outputs
1. `x_hat.csv`: this is the clustering result from SpatialSort. The format is patient by cell, in which each row indicates all the cell labels for a patient ordered by the patient ordering in the expression matrix.
2. `x_last_iteration.csv`: this is the clustering result from the last iteration of SpatialSort.
3. `x_trace.csv`: this is the label trace matrix of SpatialSort. A cell by iteration matrix. 
4. `v_measure_trace.png`: this is v-measure trace map of the run.

### Example Run
Let us run SpatialSort on one of our example datasets. Here we are running in anchor mode on a semi-real biased dataset with parameters 11 clusters 500 iterations and seed equals to 2.
```commandline
SpatialSort infer --exp-csv 'example/bias_simulated_expression.csv' --loc-csv 'example/simulated_location.csv' --rel-csv 'example/simulated_relation.csv' -k 11 -o . -t 500 -s 2 -a 'example/bias_anchor.csv' --save-trace
```

### Troubleshooting:
1. SpatialSort is not running: double check that all matrices should have the same column name for the first column, e.g. file_id / flow_id, etc. Anchors should have the same name for all of the cells of the first column.
2. NAN values in expression matricies: SpatialSort supports nans, but having too many or columns of nans will deteriorate performance as expression values are not inferred.
3. SpatialSort is taking exceptionally long: consider downsampling the inputs.

## License
SpatialSort is licensed under the MIT License.

## Version
**0.1.0** - First release.


