# CDMAM-4.0-DLMO-research-science-tool
This RST contains a set of Python scripts for extracting and analyzing CDMAM 4.0 ROIs to evaluate DBT system image quality using DL-based observer.
The purpose of this test is to use a pre-trained, fixed deep learning (DL) model, which is part of the RST, as a starting point for training a new observer model with images collected using the subject device. This process evaluates the device's
performance in terms of 4-AFC proportion correct (PC) score through cross-validation. The fixed "baseline" [model](https://plaque.twinbrook.org/index.php/s/QqtXn25qpP7MstE) was trained by the FDA using approximately 80,000 ROIs and includes images from several major DBT manufacturers, collected with assistance from the AdvaMed association. The model represents a variety of older and newer DBT systems featuring different x-ray detector types, resolutions, reconstruction algorithms, post-processing
techniques, scanning geometries, and more.
