import glob
import random
import string
import imageio
import tifffile as tiff
import pydicom
import numpy as np
import cv2
import argparse

from detect_blobs import cdmam_grid_corners
from sklearn.preprocessing import MinMaxScaler
from termcolor import colored

#----- base filename for ROIs

base_str= 'hlgc_60mm_test'

#----- systematic offsets to ROI center coordinates (add if needed)

roi_ctr_off_hor= 0  # 3  # shift ROI centers left by 3 pixels
roi_ctr_off_ver= 2  # 1  # shift ROI centers up by 1 pixel

#----- parse command line arguments

def parse_arguments():
    
    parser= argparse.ArgumentParser(description="Process DICOM files and extract ROIs from specified regions.")
    parser.add_argument("-ctr_slc", type= int, required= True, help= "Central slice in the DBT volume to extract ROIs from.")
    parser.add_argument("dicom_path", type= str, help= "Path to the directory containing DICOM files.")
    group= parser.add_mutually_exclusive_group(required= True)
    group.add_argument("-m", metavar= "coords", type= str, help= "Manual mode: specify rectangle corners as (x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D).")
    group.add_argument("-a", action= "store_true", help= "Automatic mode: auto-detect rectangle corners.")
    args= parser.parse_args()
    
    #----- validate manual mode coordinates
    
    if args.m:
        
        try:
            
            coords= list(map(int, args.m.strip("()").split(",")))
            
            if len(coords)!= 8:
                
                raise ValueError("Exactly 8 integer coordinates must be provided for -m.")
            
            args.coords= coords  # store validated coordinates
            
        except ValueError as e:
            
            parser.error(f"Invalid coordinates format for -m: {e}")
            
    return args

def generate_random_string(length= 7):
    
    characters= string.ascii_letters + string.digits
    random_string= ''.join(random.choices(characters, k= length))
    
    return random_string

#----- v6: added 56 Au-signals diameter and thickness info in ROI names

DIA= np.array([2.00, 1.70, 1.40, 1.20, 1.00, 0.88, 0.77, 0.66, 0.57, 0.50, 0.42, 0.35, 0.30, 0.25])  # circular signal diameter, mm
THK= np.array([[0.078, 0.087, 0.094, 0.103],  # circular signal thickness, um
               [0.078, 0.089, 0.096, 0.105],
               [0.079, 0.090, 0.098, 0.106],
               [0.079, 0.090, 0.097, 0.106],
               [0.091, 0.099, 0.109, 0.129],
               [0.100, 0.109, 0.128, 0.147],
               [0.110, 0.130, 0.148, 0.168],
               [0.130, 0.149, 0.170, 0.191],
               [0.150, 0.171, 0.192, 0.208],
               [0.171, 0.193, 0.210, 0.240],
               [0.191, 0.208, 0.239, 0.264],
               [0.242, 0.269, 0.308, 0.341],
               [0.312, 0.351, 0.401, 0.446],
               [0.401, 0.448, 0.493, 0.549]])

def my_function():
    
    args= parse_arguments()    
    #print(colored(f'detcd blobs: {len(good_blobs)}', 'red'))
    
    print(colored('--------------------------------------------------------------------------------------------------', 'blue'))
    
    if args.a:
        
        print(colored("Mode: automatic", 'blue'))
        
    elif args.m:
        
        print(colored(f"Mode: manual, (A,B,C,D): {args.coords}", 'blue'))
        
    print(colored(f"Central slice: {args.ctr_slc}", 'blue'))
    print(colored(f"Path to DICOM files: {args.dicom_path}", 'blue'))
    
    print(colored('--------------------------------------------------------------------------------------------------', 'blue'))
    
    ctr_slc= args.ctr_slc
    
    for fname in glob.glob(args.dicom_path+'/*.dcm'):
        
        # dcm_id= fname[-11:].split('.')[0]  # unique 7-character substring from DICOM filename for identification
        dcm_id= base_str+'_'+generate_random_string()
        
        dcm=  pydicom.dcmread(fname)
        img= dcm.pixel_array
        slc= img[ctr_slc, :, :]
        
        print(fname)  # , '    ', dcm_id)
        
        #----- (1): use CV blob detection alg. to determine outer grid corner coordinates
        
        if args.a:
            
            #----- image min/max pixel values
            
            min_px= slc.min()
            max_px= slc.max()
            
            slc_byt= (slc - min_px) * (255.0 / (max_px - min_px))
            slc_byt= np.clip(slc_byt, 0, 255)
            slc_byt= slc_byt.astype(np.uint8)
            
            cv2.imwrite('8bits.png', slc_byt)
            
            #----- call blob detection module to find A, B, C, D coordinates and square size
            
            A, B, C, D, sq= cdmam_grid_corners(slc_byt)
            
            # print(A)
            # print(B)
            # print(C)
            # print(D)
            # print(sq)
            
        #----- (2): define CDMAM outer grid corner coordinates by hand (provided CDMAM position remained intact during scans)
        
        elif args.m:
            
            A= (args.coords[0], args.coords[1])  # A             B
            B= (args.coords[2], args.coords[3])  #      CDMAM
            C= (args.coords[4], args.coords[5])  #     PHANTOM
            D= (args.coords[6], args.coords[7])  # C             D            
            
            # print(A)
            # print(B)
            # print(C)
            # print(D)
                    
        # print('1/2-ROI size:', siz_hlf)
        
        point1= np.array(A)  # top horizontal line (left to right)
        point2= np.array(B)
        f0= lambda t: point1+t*(point2-point1)
        
        point3= np.array(A)  # left vertical line (top to bottom)
        point4= np.array(C)
        f1= lambda t: point3+t*(point4-point3)
        
        point5= np.array(B)  # right vertical line (top to bottom)
        point6= np.array(D)
        f2= lambda t: point5+t*(point6-point5)
        
        #----- absolute distances, px
        
        AB= B[0]-A[0]
        AC= C[1]-A[1]
        BD= D[1]-B[1]
        
        if args.m:
            
            sq= AB/16  # square (cell) size, px
        
        gp= sq/2                     # gap between blocks, px
        siz_hlf= int(round(sq)/2)-4  # ROI (with CDMAM cell) half size
        
        #print(sq)
        
        #----- localize 40 ROIs in block "0.50-2.00"
        
        gt= ['2', '1', '3', '1',
             '3', '1', '0', '0',
             '1', '0', '2', '0',
             '2', '3', '1', '3',
             '1', '1', '0', '0',
             '3', '2', '0', '2',
             '1', '2', '3', '2',
             '0', '3', '3', '1',
             '1', '1', '0', '3',
             '2', '0', '2', '1']
        
        #----- initialize (0-1) scaler
        
        # scaler= MinMaxScaler()
        
        N= 10
        M= 16
        
        # print('----- BLOCK-1 cell coordinates -----')
        
        ctr= 0
        for i in range(N):
            for j in range(M):
                
                x_off= sq/2+sq*j
                y_off= sq/2+sq*i
                
                x_rel= x_off/AB
                y_re1= y_off/AC
                y_re2= y_off/BD
                
                pt_vert_l= f1(y_re1)
                pt_vert_r= f2(y_re2)
                fm= lambda t: pt_vert_l+t*(pt_vert_r-pt_vert_l) # horizonal line passing through cell centers
                                                                # moving UP for each new row of cells
                cc= fm(x_rel)  # cell center
                yc= cc[0].astype(int)+roi_ctr_off_hor
                xc= cc[1].astype(int)+roi_ctr_off_ver
                
                #----- crop an ROI (only four leftmost columns with brightest signals)
                
                if j >= 12:
                    
                    # print('ROI location: i= {:d}, x= {:.2f}, y= {:.2f}'.format(ctr, cc[0], cc[1]))
                    
                    dia= DIA[i]
                    thk= THK[i, j-12]
                    # print(dia, thk)
                    
                    roi= slc[xc-siz_hlf:xc+siz_hlf+1, yc-siz_hlf:yc+siz_hlf+1]  # odd number of pixels
                    # roi_scl= scaler.fit_transform(roi)                        # rescale ROI to (0-1)
                    roi[siz_hlf, :]= 0; roi[:, siz_hlf]= 0                      # draw 0-value lines separating quadrants
                    roi= np.int16(roi)
                    
                    # fname_out= 'roi_blk2.00_'+dcm_id+'_'+str(ctr+1).zfill(3)+'_q'+str(gt[ctr])+'.png'                
                    # fname_out= 'roi_'+dcm_id+'_'+'{:.3f}'.format(dia)+'_'+'{:.3f}'.format(thk)+'_q'+str(gt[ctr])+'.tif'
                    fname_out= 'roi_'+dcm_id+'_'+'{:.3f}'.format(dia)+'_'+'{:.3f}'.format(thk)+'_q'+str(gt[ctr])+'.png'
                    
                    #cv2.imwrite(fname_out, roi)
                    imageio.imwrite(fname_out, roi)
                    # tiff.imwrite(fname_out, roi_scl, dtype= roi_scl.dtype)
                    
                    ctr+= 1
                    
        #----- localize 16 ROIs in block "0.25-0.42"
        
        gt= ['3', '1', '3', '1',
             '3', '1', '1', '3',
             '3', '2', '3', '2',
             '0', '0', '0', '2']
        
        N= 4
        
        # print('----- BLOCK-2 cell coordinates -----')
        
        ctr= 0
        for i in range(N):
            for j in range(M):
                
                x_off= sq/2+sq*j
                y_off= sq/2+sq*i + (sq*10+gp)
                
                x_rel= x_off/AB
                y_re1= y_off/AC
                y_re2= y_off/BD
                
                pt_vert_l= f1(y_re1)
                pt_vert_r= f2(y_re2)
                fm= lambda t: pt_vert_l+t*(pt_vert_r-pt_vert_l)
                
                cc= fm(x_rel)  # cell center
                yc= cc[0].astype(int)+roi_ctr_off_hor
                xc= cc[1].astype(int)+roi_ctr_off_ver
                
                #----- crop an ROI (only four rightmost columns with brightest signals)
                
                if j >= 12:
                    
                    # print('ROI location: i= {:d}, x= {:.2f}, y= {:.2f}'.format(ctr, cc[0], cc[1]))
                    
                    dia= DIA[i+10]
                    thk= THK[i+10, j-12]
                    #print(dia, thk)
                    
                    roi= slc[xc-siz_hlf:xc+siz_hlf+1, yc-siz_hlf:yc+siz_hlf+1]  # odd number of pixels
                    # roi_scl= scaler.fit_transform(roi)                        # rescale ROI to (0-1)
                    roi[siz_hlf, :]= 0; roi[:, siz_hlf]= 0                      # draw 0-value lines separating quadrants
                    roi= np.int16(roi)
                    
                    # fname_out= 'roi_blk0.50_'+dcm_id+'_'+str(ctr+1).zfill(3)+'_q'+str(gt[ctr])+'.png'                
                    # fname_out= 'roi_'+dcm_id+'_'+'{:.3f}'.format(dia)+'_'+'{:.3f}'.format(thk)+'_q'+str(gt[ctr])+'.tif'
                    fname_out= 'roi_'+dcm_id+'_'+'{:.3f}'.format(dia)+'_'+'{:.3f}'.format(thk)+'_q'+str(gt[ctr])+'.png'
                    
                    # cv2.imwrite(fname_out, roi)
                    imageio.imwrite(fname_out, roi)
                    # tiff.imwrite(fname_out, roi_scl, dtype= roi_scl.dtype)
                    
                    ctr+= 1
                    
#----- extract CDMAM signal ROIs from *.dcm files

#for fname in glob.glob('1.2.840.113681.3232246842.1711111762.4392.12192.1_73200000_000620_171339558800e3.dcm'):
#for fname in glob.glob('1.2.840.113681.*.dcm'):

my_function()
