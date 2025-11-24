# CSC

Environment:

Python 3.8



Code Structure:

- data:
  - points.csv: data point
- code:
  - TSP: generate TSP solution
    - TSP_Pointerformer: generate the TSP solution using the Pointerformer method
  - clustering:
    - kruskal_clustering_balance: Balancing Kruksl algorithm
    - Kruskal_clustering: Kruskal algorithm
    - KMeans: K-Means algorithm
  - algorithm:
    - E-cluster-coo-kmeans-Pointer: generate the CSC solution with kmeans and Pointerformer method
    - E-cluster-coo-kruskal-Pointer: generate the CSC solution with kruskal and Pointerformer
    - E-cluster-coo-bkruskal-Pointer-gra: generate the CSC solution with Balancing Kruskal, Pointerformer and gradient descend（MAIN）
