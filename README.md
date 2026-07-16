# stAge

Code and data to reproduce the results from ‘Multi-tissue spatial transcriptomics reveals biological age hotspots in mouse and human aging’.


### Abstract:
Aging is heterogeneous across tissues, yet regional aging patterns within tissues remain unclear. Here we develop stAge, a framework that quantifies localized transcriptomic age (tAge) from spatial transcriptomics data in mouse and human samples during natural aging and in response to injury, infection, neurodegeneration, and cancer. stAge captures age difference among samples and provides a single multi-tissue model to examine aging within and across organs. Across tissues and species, stAge uncovers spatial gradients of biological age and shows that injury and neurodegeneration induce pronounced age acceleration, with stronger responses in older organisms and partial normalization during tissue recovery. With age, tissues develop significant hotspots of accelerated aging and coldspots of preserved resilience. Hotspots are enriched for energy metabolism and immune aging signatures, whereas chromatin-related signatures are associated with coldspots. These findings show that aging is spatially structured within tissues and lay a foundation for spatially targeted rejuvenation strategies.


### Description
This repository contains a Jupyter notebook that enables simple use of the stAge method in Python (integrated_stAge.ipynb). These are the steps taken to generate spatial transcriptomic age predictions: 

1. Load H5AD dataset and set parameters
2. Optimal Resolution Search (ORS) or custom resolution
3. Apply stAge at optimal resolution
4. Display/Save results


The following files are all needed in the same directory as the notebook (all available in stAge/scripts):

- Scaled and YuGene EN pkl files
- Mus_musculus.gene_info
- st_utils.py & st_resol.py

More instructions are available in the notebook. Happy predicting!

<img width="200" height="240" alt="image" src="https://github.com/user-attachments/assets/2abd9e20-fd08-4321-84c1-973a3e522c92" />
