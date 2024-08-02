# Ecole Polytechnique - Project Supervised by Professor Jean-Marc Allain - Extracting Characteristic Cornea Stretch From Tonometer Captures

The repository is dedicated to outlining a project I've worked on under the supervision of Professor Jean-Marc Allain at École Polytechnique. 

#### The main component of this repository is the project report.

We present below a copy of its abstract as a synopsis along with a brief description of what the repo contains so far and upcoming developments.

## Synopsis

This project comes after the statistical analysis done by Wu Yifan under Professor Jean-Marc Allain's supervision on tonometer captures from ophthalmologists containing movies of human patient corneas subjected to air puffs for an exam typically done to recover the intra-ocular pressure. The dataset contains healthy corneas and corneas diagnosed with keratoconus disease. The goal of this project is to build upon the image processing and superficial statistical measures of thickness and characteristic lengths extracted from the videos to try and extract an energy evolution of the corneas using discretization and exclusively the tonometer captures with no further information on experimental control parameters.

## Repository Description

The Project Docs Folder contains some of the notebooks used during the project - these have self-explanatory names. The sub-directories in the notebooks folder contain movies of waveforms, error evolutions, and the potential energy per cornea site evolution movies, for both the upper and middle waveforms. Messy and intermediate notebooks have been omitted for now, as well as the image processing and evolution tools script as they contain code that is yet to be refactored, but they will be available as the project evolves into its more conclusive phase outlined at the end of the report.

## Upcoming Changes

The test with the vector along which to slide edge segments.

## Potential Implementation

Machine Learning: Classifier Based on Stable Metrics Extracted by Wu Yifan (See Project Docs -> Academic Resources)
The color mesh of the thickness variation looks like something I would pass to a trained model and perform transfer learning on (I've used YAMNet for audio-spectrogram embeddings before but this is equivalently doable for image recognition.) To be evaluated as this project concludes.