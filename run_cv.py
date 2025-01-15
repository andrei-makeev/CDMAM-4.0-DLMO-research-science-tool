##!/usr/bin/python3.10

import argparse
import numpy as np
import os
import glob
import random
from   datetime import datetime

#----- output filename (edit to reflect what was measured)

now= datetime.now()
formatted_date_time= now.strftime('%Y-%m-%d_%H-%M-%S')
f_res_name= f"cv_results_{formatted_date_time}_test_clr_added.txt"

n_rmv= 200  # 688  # number of random images to remove from CV dataset
n_inc= 200  # 500  # by how much to increment n_rmv in subsequent CV runs
n_pts= 3    # number of data points to evaluate DL-model performance vs. #ROIs in CV
n_avg= 4    # number of times to repeat CV for given #ROIs

#----- paths to 'baseline' model and testing dataset

#mname= '/home/makeev/ml/keras/cdmam_dlmo/modl/cdmam_fda+advamed-hlgc_60032_cyc_lr_no_frozen_layers.keras'
#fpath= '/home/makeev/ml/keras/cdmam_dlmo/data/hologic/new_br3d/6cm_pmma_16_random_dicoms_test_autmtc_ex'

def parse_arguments():
    
    parser= argparse.ArgumentParser(description= "Cross-validation runs to compute PC scores.")
    parser.add_argument("-m", "--model", required= True, help= "Path to the baseline model (e.g., <cdmam_fda+advamed.keras>).")
    parser.add_argument("-d", "--dpath", required= True, help= "Path to the test dataset (e.g., </home/makeev/cdmam_dlmo/data>).")
    
    return parser.parse_args()

#----- function to temporarily remove n files from dataset

def mv_nfiles(fpath, n):
    
    #----- check if 'temp' subfolder exists, if not create it
    
    if not os.path.exists(fpath+'/temp'):
        
        os.makedirs(fpath+'/temp')
        
    dt= datetime.utcnow()  # set seed to ensure randomness
    unix_now= (dt-datetime(1970, 1, 1)).total_seconds()
    random.seed(int(unix_now))    
    
    roi_lst= glob.glob(fpath+'/*.png')  # roi_*.png
    random.shuffle(roi_lst)
    
    roi_mov= roi_lst[0:n]
    
    ctr= 1
    for fname in roi_mov:
        
        cmd= 'mv '+fname+' '+fpath+'/temp/.'
        print(ctr, cmd)
        os.system(cmd)
        
        ctr+= 1

#----- parse command line arguments

args= parse_arguments()
mname= args.model  # baseline model
fpath= args.dpath  # path to test dataset
        
#----- (1) run CV with all data n_avg times

# N= len(glob.glob(fpath+'/roi_*.png'))
# f_res= open(f_res_name, 'w')
# f_res.write('CV using '+str(N)+' ROIs\n')
# f_res.close()

# cmd= 'python3 test__cdmam_model_oct.py '+fpath+' a '+f_res_name+' '+mname
# os.system(cmd)

for i in range(n_avg):  # repeat n_avg times to smooth out effect of random draw
        
    if i== 0:
        
        N= len(glob.glob(fpath+'/*.png'))  # roi_*.png
        f_res= open(f_res_name, 'w')
        f_res.write('CV using '+str(N)+' ROIs\n')
        f_res.close()
        
    cmd= 'python3 test_cdmam_model.py '+fpath+' a '+f_res_name+' '+mname
    os.system(cmd)

#----- (2) run CV with less data n_avg times

for j in range(n_pts):  # repeat n_pts times, e.g. if full set had 1000 ROIs, create 800, 600, 400 ROIs subsets for additional CV runs
    
    for i in range(n_avg):  # repeat n_avg times to smooth out effect of random draw
        
        #----- move n_rmv ROIs to 'temp' folder
        
        mv_nfiles(fpath, n_rmv)
        
        if i== 0:
            
            N= len(glob.glob(fpath+'/*.png'))  # roi_*.png
            f_res= open(f_res_name, 'a')
            f_res.write('CV using '+str(N)+' ROIs\n')
            f_res.close()
            
        cmd= 'python3 test_cdmam_model.py '+fpath+' a '+f_res_name+' '+mname
        os.system(cmd)
        
        #----- move ROIs back from 'temp' folder
        
        cmd= 'mv '+fpath+'/temp/*.png '+fpath+'/.'
        os.system(cmd)
            
    n_rmv+= n_inc
    
f_res.close()
