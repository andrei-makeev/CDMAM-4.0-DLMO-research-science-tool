import cv2
import numpy as np
from termcolor import colored

def cdmam_grid_corners(img):
    
    #----- takes DBT slice with CDMAM 4.0 and BR3D swirl slab and returns
    #----- four extreme grid corner corrdinates (A, B, C, and D) along
    #----- with CDMAM square size for ROI extraction. Expected input
    #----- type - grayscale 0-255
    
    # image= cv2.imread(img)
    
    image= img          # input: 0-255 grayscale image    
    # image= 255-image  # invert image so that blobs are dark
    
    #----- CV2 blob detector parameters
    
    params= cv2.SimpleBlobDetector_Params()
    
    params.minThreshold= 13  # 2
    params.maxThreshold= 100
    params.thresholdStep= 1
    
    params.filterByArea= True
    params.minArea= 30
    params.maxArea= 70
    # params.filterByCircularity= True 
    # params.minCircularity= 0.5
    # params.filterByConvexity= True
    # params.minConvexity= 0.5
    
    params.filterByColor= True  # detect light blobs (255)
    params.blobColor= 255
    
    #----- parameters defining "good" rows of blobs in CDMAM image
    
    expected_blob_count= 6
    min_x_spacing= 160  # min distance between blobs horizontally
    
    detector= cv2.SimpleBlobDetector_create(params)
    
    #----- do some pre-porcessing to optimize blob detection
    
    # blurrd= cv2.GaussianBlur(image, (3, 3), 0)
    # thresh= cv2.adaptiveThreshold(blurrd, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # edges=  cv2.Canny(thresh, 100, 200)
    
    #----- top-hat enhancement
    
    kernel= cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    tophat_img= cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    cv2.imwrite('tophat.png', tophat_img)
    
    #----- detect blobs
        
    # keypoints= detector.detect(image)
    # keypoints= detector.detect(blurrd)
    keypoints= detector.detect(tophat_img)
    
    print('total blobs:', len(keypoints))
    
    blob_points= [(int(k.pt[0]), int(k.pt[1])) for k in keypoints]
    
    #----- sort blobs based on their y-coordinates
    
    blob_points_sorted_by_y= sorted(blob_points, key= lambda p: p[1])
    
    y_threshold= 10  # expected vertical space within which blobs
                     # belonging to a particular "row" are localized
    
    #----- group blobs into rows based on y-coordinate similarity
    
    rows= []
    current_row= [blob_points_sorted_by_y[0]]
    
    for i in range(1, len(blob_points_sorted_by_y)):
        
        if abs(blob_points_sorted_by_y[i][1] - blob_points_sorted_by_y[i - 1][1]) <= y_threshold:
            
            current_row.append(blob_points_sorted_by_y[i])
            
        else:
            
            rows.append(current_row)
            current_row= [blob_points_sorted_by_y[i]]
            
    rows.append(current_row)
    
    #----- sort blobs in each row by their x-coordinate
    
    rows_sorted_by_x= [sorted(row, key= lambda p: p[0]) for row in rows]
    
    # for index, row in enumerate(rows_sorted_by_x):
    #     print(f"ROW{index}: {row}")
    
    #----- now we can filter out blobs based on expected number of blobs per row and regular spacing
        
    filtered_rows= []    
    for row in rows_sorted_by_x:
        
        if len(row) >= expected_blob_count:  # check if spacing between blobs is roughly uniform
            
            spacings= [abs(row[i][0] - row[i - 1][0]) for i in range(1, len(row))]
            
            # if all(min_x_spacing - 10 <= spacing <= min_x_spacing + 10 for spacing in spacings):
            # if all(min_x_spacing - 20 <= spacing <= min_x_spacing*2 + 20 for spacing in spacings):
            if all(min_x_spacing - 20 <= spacing for spacing in spacings):
                
                filtered_rows.append(row)
                
    top7= filtered_rows[0]
    bot8= filtered_rows[-1]
    
    #----- flatten filtered rows into a list of GOOD blobs
    
    good_blobs= [blob for row in filtered_rows for blob in row]
    bgr_image= cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)  # convert grayscale to BGR for color markers
    
    #for pt in good_blobs:    # (1) draw only "good" blobs that passed filtering criteria
    for pt in blob_points:  # (2) draw all blobs found (for debugging)
        
        cv2.circle(bgr_image, pt, 8, (0, 255, 255), -1)  # color blobs        
        # cv2.circle(image, pt, 5, (255, 255, 255), -1)  # white blobs
        
    cv2.imwrite('blobs.png', bgr_image)
    
    if(len(good_blobs)== 42):
        
        print('detcd blobs:', len(good_blobs))
        
    else:
        
        print(colored(f'detcd blobs: {len(good_blobs)}', 'red'))
        
    top7= sorted(top7, key= lambda x: x[0])
    bot8= sorted(bot8, key= lambda x: x[0])
    
    # print('----- top 7 markers -----')
    
    # for point in top7:
        
    #     x, y= point
    #     print(f"({x:.2f}, {y:.2f})")
        
    # print('----- bottom 8 markers -----')
    
    # for point in bot8:
        
    #     x, y= point
    #     print(f"({x:.2f}, {y:.2f})")
        
    M1= np.array(top7[0])
    M2= np.array(top7[-1])
    M3= np.array(bot8[0])
    M4= np.array(bot8[-1])
    
    M12= M2[0]-M1[0]
    sq=  M12/12
    
    # print('# good blobs:', len(good_blobs), sq)
    
    f0= lambda t: M1-t*(M2-M1)
    M0= f0(sq*4/M12)  # determine missing (top left) marker point in CDMAM 4.0
    
    #----- hardwired rel. offsets from four CDMAM circular markers closest to four outmost grid corners
    
    dx1= 0.0014002553118510905*(M2[0]-M0[0])  # (-2, 22) from M0
    dy1= 0.0107135903556761810*(M3[1]-M0[1])
    
    dx2= 0                                    # (0, 22) from M2
    dy2= 0.0107209262860323610*(M4[1]-M2[1])
    
    dx3= 0.0020998895840529380*(M4[0]-M3[0])  # (-3, -21) from M3
    dy3= 0.0102266089758727180*(M3[1]-M0[1])
    
    dx4= 0                                    # (0, 21) from M4
    dy4= 0.0102336114548490720*(M4[1]-M2[1])
    
    #----- corner A coordinates
    
    f0= lambda t: M0-t*(M2-M0)
    f1= lambda t: M0+t*(M3-M0)
    
    AX= np.round(f0(dx1/(M2[0]-M0[0]))[0]).astype(int)
    AY= np.round(f1(dy1/(M3[1]-M0[1]))[1]).astype(int)
    A= (AX, AY)
    
    # print(f"M0= ({M0[0]}, {M0[1]})")
    # print(f"A= ({AX}, {AY})")
    
    #----- corner B coordinates
    
    f0= lambda t: M2+t*(M2-M0)
    f1= lambda t: M2+t*(M4-M2)
    
    BX= np.round(f0(dx2/(M2[0]-M0[0]))[0]).astype(int)
    BY= np.round(f1(dy2/(M4[1]-M2[1]))[1]).astype(int)
    B= (BX, BY)
    
    # print(f"M2= ({M2[0]}, {M2[1]})")
    # print(f"B= ({BX}, {BY})")
    
    #----- corner C coordinates
    
    f0= lambda t: M3-t*(M4-M3)
    f1= lambda t: M3-t*(M3-M0)
    
    CX= np.round(f0(dx3/(M4[0]-M3[0]))[0]).astype(int)
    CY= np.round(f1(dy3/(M3[1]-M0[1]))[1]).astype(int)
    C= (CX, CY)
    
    # print(f"M3= ({M3[0]}, {M3[1]})")
    # print(f"C= ({CX}, {CY})")
    
    #----- corner D coordinates
    
    f0= lambda t: M4+t*(M4-M3)
    f1= lambda t: M4-t*(M4-M2)
    
    DX= np.round(f0(dx4/(M4[0]-M3[0]))[0]).astype(int)
    DY= np.round(f1(dy4/(M4[1]-M2[1]))[1]).astype(int)
    D= (DX, DY)
    
    # print(f"M4= ({M4[0]}, {M4[1]})")
    # print(f"D= ({DX}, {DY})")
    
    return A, B, C, D, sq
    # return (100, 100), (200, 200), (300, 300), (400, 400), 78.0
