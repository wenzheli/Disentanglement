# Disentanglement


### Description for files
- admm.m :  the main file for learning model parameteres, it is based on the note.
- modularity_dir:  the funtion uses to generate block-diagonal matrix. The input is the matrix, and output is the cluster indicates 
                for each dimensions. 
                
- *.mat :  this is the face image data sets used for testing. 


###TODO list:
* Verify sparse row rank regularization gives block-diagonal matrix (L_1 + nuclear_norm) If not, we need to seek for other regularization
* Need to make sure current formulation (linear autoencoder) works and is able to generate block-diagonal weight matrix. 
* Extend the current formulation to non-linear autoencoder by putting sigmoid function. (Derivation should be a bit complicated)
* Extend the current formulation to non-linear restricteid boltzmann machine. (ideally, we can test out 2 or 3 groups of hidden factors and see what features each group captures. The baseline could be the paper by Scott and Honglak. If our formulation can find the similar effects, then we win.  

The following are some advanced lists:
* Put the nonparametric prior to automatically discover the number of hidden factors. 
* Build deep architecture. 
* Some theoretical analysis. 


###Reference
[1] Richard, Emile, Pierre-AndrÃ© Savalle, and Nicolas Vayatis. "Estimation of simultaneously sparse and low rank matrices." arXiv preprint arXiv:1206.6474 (2012).

[2] Zhou, Ke, Hongyuan Zha, and Le Song. "Learning social infectivity in sparse low-rank networks using multi-dimensional hawkes processes." Proceedings of the Sixteenth International Conference on Artificial Intelligence and Statistics. 2013.

[3] Avron, Haim, et al. "Efficient and practical stochastic subgradient descent for nuclear norm regularization." arXiv preprint arXiv:1206.6384 (2012).

[4] Feng, Jiashi, et al. "Robust Subspace Segmentation with Block-Diagonal Prior." Computer Vision and Pattern Recognition (CVPR), 2014 IEEE Conference on. IEEE, 2014.

[5] CandÃ¨s, Emmanuel J., et al. "Robust principal component analysis?." Journal of the ACM (JACM) 58.3 (2011): 11.

[6] Hasselmo, M.E., Schnell, E. \& Barkai, E. (1995) Dynamics of learning
and recall at excitatory recurrent synapses and cholinergic modulation
in rat hippocampal region CA3. Journal of Neuroscience




