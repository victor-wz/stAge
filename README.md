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

More specific instructions are available in the notebook. 


<svg xmlns="http://www.w3.org/2000/svg" viewBox="-5 30 250 250" width="200" height="240">
  <defs>
    <!-- Soft glow -->
    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="4" result="coloredBlur"/>
      <feMerge>
        <feMergeNode in="coloredBlur"/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>

    <!-- Background gradient -->
    <radialGradient id="bg" cx="50%" cy="50%" r="80%">
      <stop offset="0%" stop-color="rgba(234, 239, 242, 0.93)"/>
      <stop offset="100%" stop-color="#3774b1ff"/>
    </radialGradient>

    <!-- Spot gradient -->
    <radialGradient id="spotGrad" cx="40%" cy="40%" r="80%">
      <stop offset="0%" stop-color="#3496ebff"/>
      <stop offset="100%" stop-color="#5bbff1ff"/>
    </radialGradient>

    <!-- Age clock gradient -->
    <linearGradient id="clockGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#ea004e"/>
      <stop offset="100%" stop-color="#ff6f91"/>
    </linearGradient>
  </defs>

  <!-- Base circle -->
  <circle cx="120" cy="120" r="110" fill="url(#bg)" />

  <!-- Subtle inner ring -->
  <circle cx="120" cy="120" r="102"
          stroke="#ffffff1a"
          stroke-width="2"
          fill="none"/>

  <!-- Spatial transcriptomics grid -->
  <g opacity="0.75">
    <circle cx="75" cy="100" r="7" fill="url(#spotGrad)"/>
    <circle cx="105" cy="100" r="7" fill="url(#spotGrad)"/>
    <circle cx="135" cy="100" r="7" fill="url(#spotGrad)"/>
    <circle cx="165" cy="100" r="7" fill="url(#spotGrad)"/>

    <circle cx="90" cy="120" r="7" fill="url(#spotGrad)"/>
    <circle cx="120" cy="120" r="7" fill="url(#spotGrad)"/>
    <circle cx="150" cy="120" r="7" fill="url(#spotGrad)"/>

    <circle cx="75" cy="140" r="7" fill="url(#spotGrad)"/>
    <circle cx="105" cy="140" r="7" fill="url(#spotGrad)"/>
    <circle cx="135" cy="140" r="7" fill="url(#spotGrad)"/>
    <circle cx="165" cy="140" r="7" fill="url(#spotGrad)"/>
  </g>

  <!-- Clock ring -->
  <circle cx="120" cy="120" r="65"
          stroke="url(#clockGrad)"
          stroke-width="7"
          fill="none"
          filter="url(#glow)"/>

  <!-- Clock hands -->
  <g stroke="url(#clockGrad)" stroke-width="6" stroke-linecap="round">
    <line x1="120" y1="120" x2="100" y2="90"  opacity="0.95"/>
    <line x1="120" y1="120" x2="160" y2="92"  opacity="0.95"/>
  </g>

  <!-- Center dot -->
  <circle cx="120" cy="120" r="6" fill="url(#clockGrad)" filter="url(#glow)"/>

  <!-- "stAge" label -->
  <text x="50" y="280"
        font-family="sans-serif"
        font-size="50"
        font-weight="600"
        fill="url(#clockGrad)"
        letter-spacing="1.5">
    stAge
  </text>

</svg>
