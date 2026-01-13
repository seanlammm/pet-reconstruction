# pet-reconstruction

## Environment Setup

```bash
conda create -n env-name  -c conda-forge libparallelproj parallelproj pytorch cupy cudatoolkit=11.8 tqdm python=3.10
```



## Implemented Reconstruction Algorithms

1. OSEM
1. MLAA



## Utilities

1. Coincidence raw data –> CDF
2. CastorId –> Sinogram (Michegram & SSRB)
3. Mumap –> ACF
4. ROOT –> CDF
5. gPET scatter CDF -> castor scf
