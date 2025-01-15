import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import matplotlib.ticker as plticker
from   matplotlib.lines import Line2D

#----- set print options to suppress line breaks

np.set_printoptions(threshold= np.inf, linewidth= np.inf)

#fname= 'cv_results_2024-11-21_01-00-50.txt'
#fname= 'cv_results_2024-11-20_19-06-27.txt'  # Siemens B.brilliant
#fname= 'cv_results_2024-11-21_01-00-50.txt'  # Siemens Revelation
#fname= 'cv_results_2024-11-22_15-32-11_no_holgc_test_holgc_new_br3d_6cm.txt'
#fname= 'cv_results_2024-12-04_13-14-29_no_holgc_test_holgc_new_br3d_6cm_cyc_LR.txt'
#fname= 'cv_results_2024-12-24_17-11-36_no_holgc_test_holgc_new_br3d_6cm_cyc_LR_stage1+stage2_frozen.txt'
#fname= 'cv_results_2024-12-25_17-42-12_no_holgc_test_holgc_new_br3d_6cm_cyc_LR_stage1+stage2+stage3_frozen.txt'
#fname= 'cv_results_2024-12-26_18-27-16_no_holgc_test_holgc_new_br3d_6cm_cyc_LR_first4_conv2_layers_frozen.txt'
#fname= 'cv_results_2024-12-30_12-34-19_no_holgc_test_holgc_new_br3d_6cm_cyc_LR_all_layers_trainable.txt'
#fname= 'cv_results_2025-01-03_19-58-51_no_holgc_test_holgc_new_br3d_6cm_cyc_LR_all_layers_trainable_18_random_dicoms.txt'
#fname= 'cv_results_2025-01-06_14-05-05_no_holgc_test_holgc_new_br3d_6cm_cyc_LR_all_layers_trainable_18_random_dicoms_LR=1e-05.txt'
#fname= 'cv_results_2025-01-07_14-12-16_no_holgc_test_holgc_new_br3d_6cm_cyc_LR_all_layers_trainable_18_random_dicoms_LR=1e-05_correl_noise_added.txt'
#fname= 'cv_results_2025-01-10_16-41-51_no_holgc_test_holgc_new_br3d_6cm_cyc_LR_all_layers_test_manual_ex.txt'
#fname= 'cv_results_2025-01-10_19-36-54_no_holgc_test_holgc_new_br3d_6cm_cyc_LR_all_layers_test_autmtc_ex.txt'

def parse_arguments():
    
    parser= argparse.ArgumentParser(description= "Visualize and save CV results.")
    parser.add_argument("-f", "--file_inp", required= True, help= "Path to the CV results file (e.g., <cv_results_file.txt>).")
    parser.add_argument("-o", "--file_out", required= True, help= "output PNG image (e.g., <output_filename.png>).")
    
    return parser.parse_args()

def extract_nroi(fname):
    
    values= []
    with open(fname, 'r') as f:
        
        for line in f:
            
            matches= re.findall(r'CV using (\d+) ROIs', line)
            
            if matches:
                
                values.extend(map(int, matches))
                
    return np.sort(np.array(values))

#----- parse command line arguments

args= parse_arguments()
fname_inp= args.file_inp  # input text file with CV results
fname_out= args.file_out  # output PNG graphics file

#----- array with number of ROIs used in CV training

nt_array= extract_nroi(fname_inp)

#----- read PC values for n_pts data points

pc_arr= []
n_rpt= 4  # number of repeated CV runs

f_res= open(fname_inp, 'r')
lines= f_res.readlines()
lines_pc= list(range(1, 5))+list(range(6, 10))+list(range(11, 15))+list(range(16, 20))

ctr= 0
for line in lines:
    
    if ctr in lines_pc:
        
        curr_line= [float(num) for num in line.strip().split()]
        pc_arr.append(curr_line)
        
    ctr+= 1
    
#----- split 'pc_arr' list into chunks of 4 and merge each chunk into a single array

pc_arr_spl= [np.concatenate(pc_arr[i:i+n_rpt]) for i in range(0, len(pc_arr), n_rpt)]

# for idx, arr in enumerate(pc_arr_spl):  # check resulting arrays
    
#     print(f'Array {idx+1}:\n{arr}')

pc0= np.round(np.mean(pc_arr_spl[3]), 3)  # round to 3 decimal points
pc1= np.round(np.mean(pc_arr_spl[2]), 3)
pc2= np.round(np.mean(pc_arr_spl[1]), 3)
pc3= np.round(np.mean(pc_arr_spl[0]), 3)

se0= np.round(np.std(pc_arr_spl[3])/np.sqrt(len(pc_arr_spl[3])), 3)
se1= np.round(np.std(pc_arr_spl[2])/np.sqrt(len(pc_arr_spl[2])), 3)
se2= np.round(np.std(pc_arr_spl[1])/np.sqrt(len(pc_arr_spl[1])), 3)
se3= np.round(np.std(pc_arr_spl[0])/np.sqrt(len(pc_arr_spl[0])), 3)

pc_array= [pc0, pc1, pc2, pc3]
sd_array= [se0, se1, se2, se3]

#print(pc_array, sd_array)

plt.errorbar(nt_array, pc_array, yerr= sd_array, fmt= 'o', fillstyle= 'none', color= mcd.CSS4_COLORS['red'], linewidth= 1.5, capsize= 3, label= '5 cm PMMA')  # ^ triangle marker
#plt.title('Siemens Revelation: CDMAM + $\mathregular{swirl^{*}}$ + PMMA')
#plt.title('Siemens B.brilliant: CDMAM+BR3D+50mm PMMA')
plt.title('System \'A\': test with CDMAM+BR3D+50mm PMMA', color= 'blue')
plt.xlabel('#ROIs in 10-fold CV')
plt.ylabel('PC$_{4-AFC}$')
#plt.ylabel('PC')
#plt.legend(loc= 'lower right', framealpha= 1.0)
plt.ylim(0.0, 1.0)
#plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
plt.grid(which= 'major', axis= 'y')

#----- add minor grid lines to y-axis

intl= np.arange(0.1, 1, 0.1)
locn= plticker.MultipleLocator(base= 0.1)
plt.gca().yaxis.set_minor_locator(locn)
plt.grid(which= 'minor', axis= 'y', linestyle= 'dashed')  # , linewidth= 0.6)
plt.ylim(0.5, 1.025)

#----- add PC values in text box

N1= nt_array[0]

textstr= f"""
$PC_{{N={nt_array[0]}}}$: {pc0:.3f}$\pm${se0:.3f}
$PC_{{N={nt_array[1]}}}$: {pc1:.3f}$\pm${se1:.3f}
$PC_{{N={nt_array[2]}}}$: {pc2:.3f}$\pm${se2:.3f}
$PC_{{N={nt_array[3]}}}$: {pc3:.3f}$\pm${se3:.3f}
"""

# txt1= f"$PC_{{N={nt_array[0]}}}$: {pc0}$\pm${se0}"
# txt2= f"$PC_{{N={nt_array[1]}}}$: {pc1}$\pm${se1}"
# txt3= f"$PC_{{N={nt_array[2]}}}$: {pc2}$\pm${se2}"
# txt4= f"$PC_{{N={nt_array[3]}}}$: {pc3}$\pm${se3}"

# textstr= """
# $PC_{txt1}$=
# PC_2= ...
# PC_3= ...
# PC_4= ...
# """

#----- add text box to plot
props= dict(boxstyle= 'round', facecolor= 'wheat', alpha=0.5)
plt.text(0.95, 0.05, textstr, transform=plt.gca().transAxes, fontsize= 9, bbox= props, verticalalignment= 'bottom', horizontalalignment='right')

#----- save plot into PDF

plt.rcParams['figure.dpi']= 600
plt.rcParams['savefig.dpi']= 600
#plt.savefig('dlmo_test_hlgc_new_swirls_cyc_LR.png', bbox_inches= 'tight')
plt.savefig(fname_out, bbox_inches= 'tight')
