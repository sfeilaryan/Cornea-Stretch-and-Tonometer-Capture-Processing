# Ecole Polytechnique - Project Supervised by Professor Jean-Marc Allain - Extracting Characteristic Cornea Stretch From Tonometer Captures

The repository is dedicated to outlining a project I've worked on under the supervision of Professor Jean-Marc Allain at École Polytechnique. 

#### The main component of this repository is the project report, available in LaTeX and PDF form in the Project Report Directory.

We present below a copy of its abstract as a synopsis along with a brief description of what the repo contains so far and upcoming developments.

## Citation

If you use this report or the provided scripts in your research, please cite them using the following metadata.

```bibtex
@misc{sfeila_2024,
  author = {Ryan Sfeila},
  title = {Extracting Cornea Energy Proxy from Tonometer Captures},
  year = {2024},
  url = {https://github.com/sfeilaryan/Cornea-Stretch-and-Tonometer-Capture-Processing},
  supervisor = {Jean-Marc Allain},
  license = {BSD-3-Clause}
}
```

## Synopsis

This project comes after the statistical analysis done by Wu Yifan under Professor Jean-Marc Allain's supervision on tonometer captures from ophthalmologists containing movies of human patient corneas subjected to air puffs for an exam typically done to recover the intra-ocular pressure. The dataset contains healthy corneas and corneas diagnosed with keratoconus disease. The goal of this project is to build upon the image processing and superficial statistical measures of thickness and characteristic lengths extracted from the videos to try and extract an energy evolution of the corneas using discretization and exclusively the tonometer captures with no further information on experimental control parameters.

### Note: Code that is no longer useful and generated data that may be interesting but is no longer relevant in light of the latest developments of the project are all moved to Project Files -> Archived and Deprecated.

## Upcoming Changes

- The test with the vector along which to slide edge segments.

- Rewriting of the functions to generate waveform and potential distribution movies

## Potential Implementation

Machine Learning: Classifier Based on Stable Metrics Extracted by Wu Yifan (See Reading.)
The color mesh of the thickness variation looks like something I would pass to a trained model and perform transfer learning on (I've used YAMNet for audio-spectrogram embeddings before but this is equivalently doable for image recognition.) To be evaluated as this project concludes.
