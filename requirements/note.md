Both `requirements.txt` and `environment.yaml` describe the only environment used to run this code. Please use them as references and install the dependencies selectively.

we use python 3.13.11.

The key dependencies are:

```text
h5py
hdf5storage
mamba_ssm==1.0.1
torch==2.9.1
torch-kmeans==0.2.0
```

Versions of the other dependencies can be adjusted as needed.

After installing `torch-kmeans==0.2.0`, replace the installed `torch_kmeans/clustering/kmeans.py` with the `kmeans.py` provided in this directory. This modified version exposes the squared distance from each pixel to its assigned cluster center through `result.inertia[1]`, which is required by the indexing code.
