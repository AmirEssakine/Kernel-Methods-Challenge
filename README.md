# Kernel methods for machine learning inclass data challenge 2024 - 2025

This repository contains the handing for the MVA master's class: Kernel methods for machine learning: https://mva-kernel-methods.github.io/course-page/

This work has been done in a team of two: Amer Essakine and Liam Ludington.

The challenge was a classification task to predict wether a DNA sequence region is binding to a specific transcription factor. The implementations have been done from scratch and there was no use of built-in functions in sklearn or libsvm. It contains methods we tried in order to get the best accuracy score.

Needed libraries : numpy, pandas,multiprocessing, cvxopt, pickles, igraph,networkx,itertools, scipy, collections, copy and math

The implementation contains a range of general kernels like the gaussian kernel, polynomial kernel as well as the spectrum kernel and mismatch kernel and weighted mismatch kernel more specific to the problem. In addition, we implemented Convolutional Kitchen Sinks [1] which is an alternative method that approximates convolutional transformations with random filters, followed by a linear classifier. This reduces training complexity while retaining high predictive power. To improve results, we used multiple CKSs to estimate features by taking the sum of three kernels with different parameters. It also contains an implementation of kernel SVM and kernel PCA


[1] Alyssa Morrow, Vaishaal Shankar, Devin Petersohn, Anthony Joseph, Benjamin Recht, and Nir Yosef. Convolutional kitchen sinks for transcription factor binding site prediction. arXiv preprint arXiv:1706.00125, 2017.
