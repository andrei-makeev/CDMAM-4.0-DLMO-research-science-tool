# CDMAM-4.0-DLMO-research-science-tool
This RST contains a set of Python scripts for extracting and analyzing CDMAM 4.0 ROIs to evaluate DBT system image quality using DL-based observer.
The purpose of this test is to use a pre-trained, fixed deep learning model, as a starting point for training a *new* observer model with images collected using the subject device. The software evaluates DBT performance in terms of 4-AFC proportion correct (PC) score through cross-validation. The "starting point" *baseline* [model](https://plaque.twinbrook.org/index.php/s/QqtXn25qpP7MstE) was trained by the FDA using approximately 80,000 CDMAM 4.0 ROIs and includes images from several major DBT manufacturers, collected with assistance from the AdvaMed association. The model represents a variety of older and newer DBT systems with different x-ray detector types, resolutions, reconstruction algorithms, post-processing techniques, scanning geometries, and more.
A series of DICOM DBT reconstruction volumes of the (CDMAM 4.0 + BR3D "swirl" background + PMMA) phantom assembly is used as an input to produce PC score and its standard error of the mean as a measure of system performance. There are three steps involved: 1) data acquisition (described in [data collection manual PDF](https://plaque.twinbrook.org/index.php/s/AZYWP2z9SBYyexd), 2) ROI extraction (manual or automated), and 3) cross-validation/inference for image quality perfomance assessment. Provided below are examples of how to execute the above tasks.
# ROI extraction
To run the ROI extraction script in **manual** mode (with CDMAM 4.0 grid outer rectangle vertex (*x,y*)-coordinates as determined visually in ImageJ):
```bash
python3 read_cdmam_blobs.py -ctr_slc 24 <path_to_dicom_files> -m "(x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D)"
```
where `ctr_slc` is the DBT plane in focus with numbering starting from zero, `(x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D)` are the CDMAM 4.0 grid outer rectangle corner coordinates in pixels. This method can be used if the data collection (for given PMMA thickness ) was done with CDMAM phantom position *unchanged* between DBT scans. 
[Sample dataset](https://plaque.twinbrook.org/index.php/s/ABCD) DICOM files for testing can be used with `"(396, 493, 1713, 499, 388, 2345, 1705, 2352)"`.
If the CDMAM position was accidently altered between the scans the user has an option to apply the script in manual mode as above, supplying new A, B, C, D coordinates of the CDMAM grid rectangle corners for each scan when position was changed, or run the script in **automatic** mode, in which the CV2 `SimpleBlobDetector` algorithm will need to be tuned to find the fiducial markers in the CDMAM image. This process is explained in the ROI extraction manual [PDF](https://plaque.twinbrook.org/index.php/s/ABCD) To run the ROI extraction script in **automatic** mode:
```bash
python3 read_cdmam_blobs.py -ctr_slc 24 <path_to_dicom_files> -a
```
Afer extraction process is complete it is recommended to visually inspect image patches in ImageJ by loading all `roi_*.png` files into ImageJ `File -> ImageSequence`. Good quality ROIs may have some of CDMAM 4.0 cell-separating lines visible in the ROIs on the squares borders, but no more than a few (2-4) pixels deep into ROI. No any other sumbols or characters present in CDMAM should be visible. Square ROI patches will have two zero-valued lines drawn to separate them into four qudrants. The central detail (signal) of the CDMAM should be approximately at the intersection of these lines, while the eccentrtic detail should be in one of the four quadrants.



