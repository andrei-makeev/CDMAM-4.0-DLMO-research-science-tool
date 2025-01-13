# CDMAM-4.0-DLMO-research-science-tool
This RST contains a set of Python scripts for extracting and analyzing CDMAM 4.0 ROIs to evaluate DBT system image quality using DL-based observer.
The purpose of this test is to use a pre-trained, fixed deep learning model, as a starting point for training a *new* observer model with images collected using the subject device. The software evaluates DBT performance in terms of 4-AFC proportion correct (PC) score through cross-validation. The fixed "baseline" [model](https://plaque.twinbrook.org/index.php/s/QqtXn25qpP7MstE) was trained by the FDA using approximately 80,000 CDMAM 4.0 ROIs and includes images from several major DBT manufacturers, collected with assistance from the AdvaMed association. The model represents a variety of older and newer DBT systems featuring different x-ray detector types, resolutions, reconstruction algorithms, post-processing techniques, scanning geometries, and more.
A series of DICOM DBT reconstruction volumes of the (CDMAM 4.0 + BR3D "swirl" background + PMMA) phantom assembly are used as an input to produce PC score and its standard error of the mean as a measure of system performance (image quality). There are three steps involved: 1) data acucquisition (described in detail in [data collection manual PDF](https://plaque.twinbrook.org/index.php/s/AZYWP2z9SBYyexd), 2) ROI extraction (automated or manual), and 3) running cross-validation for image quality perfomance assessment. Provided below are examples of hpw to execute the above tasks.
ROI extraction

To run the ROI extraction script in <u>manual</u> mode (with CDMAM 4.0 grid extreme vertices (x,y)-coordinates supplied by the user), use the following command in your terminal:

```bash
python3 read_cdmam_blobs.py -ctr_slc 24 <path_to_dicom_files> -m "(x~A~, y~A~, x~B~, y~B~, x~C~, y~C~, x~D~, y~D~)"



