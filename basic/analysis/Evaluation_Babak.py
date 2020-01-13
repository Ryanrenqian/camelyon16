# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:09:32 2016

@author: Babak Ehteshami Bejnordi

Evaluation code for the Camelyon16 challenge on cancer metastases detecion
"""

import openslide
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import measure
import math
import os
import sys
from xml.etree import ElementTree as ET    
from xml.etree.ElementTree import Element, SubElement
from xml.dom import minidom
from sklearn.metrics import roc_curve, roc_auc_score
import csv   
   
def computeEvaluationMask(maskDIR, resolution, level):
    """Computes the evaluation mask.
    
    Args:
        maskDIR:    the directory of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made
        
    Returns:
        evaluation_mask
    """
    slide = openslide.open_slide(maskDIR)
    dims = slide.level_dimensions[level]
    pixelarray = np.zeros(dims[0]*dims[1], dtype='uint')
    pixelarray = np.array(slide.read_region((0,0), level, dims))
    distance = nd.distance_transform_edt(255 - pixelarray[:,:,0])
    Threshold = 75/(resolution * pow(2, level) * 2) # 75µm is the equivalent size of 5 tumor cells
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity = 2) 
    return evaluation_mask
    
    
def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)
    
    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object 
        should be less than 275µm to be considered as ITC (Each pixel is 
        0.243µm*0.243µm in level 0). Therefore the major axis of the object 
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.
        
    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made
        
    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    all_tumor_centroids = {}
    max_label = np.amax(evaluation_mask)    
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = [] 
    threshold = 275/(resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i+1)
        else:
            label = 'Label ' + str(i+1)
            all_tumor_centroids[label] = properties[i].centroid
    return Isolated_Tumor_Cells, all_tumor_centroids


def readCSVContent(csvDIR):
    """Reads the data inside CSV file
    
    Args:
        csvDIR:    The directory including all the .csv files containing the results.
        Note that the CSV files should have the same name as the original image
        
    Returns:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
    """
    Xcorr, Ycorr, Probs = ([] for i in range(3))
    csv_lines = open(csvDIR,"r").readlines()
    for i in range(len(csv_lines)):
        line = csv_lines[i]
        elems = line.rstrip().split(',')
        try:
            float(elems[0])
            Probs.append(float(elems[0]))
            Xcorr.append(int(float(elems[1])))
            Ycorr.append(int(float(elems[2])))
        except ValueError:
            print "Not float"

    return Probs, Xcorr, Ycorr
    
def readCSVContentEval1(csvDIR):
    """Reads the data inside CSV file
    
    Args:
        csvDIR:    The directory including all the .csv files containing the results.
        Note that the CSV files should have the same name as the original image
        
    Returns:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
    """
    Probs = dict()
    csv_lines = open(csvDIR,"r").readlines()
    for i in range(len(csv_lines)):
        line = csv_lines[i]
        elems = line.rstrip().split(',')
        try:
            Probs[elems[0]] = float(elems[1])
        except ValueError:
            print "Not float"

    return Probs

         
def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells, all_tumor_centroids, level):
    """Generates true positive and false positive stats for the analyzed image
    
    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        is_tumor:   A boolean variable which is one when the case cotains tumor
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
        level:      The level at which the evaluation mask was made
         
    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections
        
        TP_probs:   A list containing the probabilities of the True positive detections
        
        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)
        
        detection_summary:   A python dictionary object with keys that are the labels 
        of the lesions that should be detected (non-ITC tumors) and values
        that contain detection details [confidence score, X-coordinate, Y-coordinate]. 
        Lesions that are missed by the algorithm have an empty value.
        
        FP_summary:   A python dictionary object with keys that represent the 
        false positive finding number and values that contain detection 
        details [confidence score, X-coordinate, Y-coordinate]. 
    """
    
    max_label = np.amax(evaluation_mask)
    FP_probs = [] 
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}  
    FP_summary = {}
    for i in range(1,max_label+1):
        if i not in Isolated_Tumor_Cells:
            label = 'Label ' + str(i)
            detection_summary[label] = [0, all_tumor_centroids[label][1]*pow(2, level), all_tumor_centroids[label][0]*pow(2, level)]        
     
    FP_counter = 0       
    if (is_tumor):
        for i in range(0,len(Xcorr)):
            if (Ycorr[i]/pow(2, level) < evaluation_mask.shape[0] and Xcorr[i]/pow(2, level) < evaluation_mask.shape[1]):
                HittedLabel = evaluation_mask[Ycorr[i]/pow(2, level), Xcorr[i]/pow(2, level)]
                if HittedLabel == 0:
                    FP_probs.append(Probs[i])
                    key = 'FP ' + str(FP_counter)
                    FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                    FP_counter+=1
                elif HittedLabel not in Isolated_Tumor_Cells:
                    if (Probs[i]>TP_probs[HittedLabel-1]):
                        label = 'Label ' + str(HittedLabel)
                        detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                        TP_probs[HittedLabel-1] = Probs[i]                                     
    else:
        for i in range(0,len(Xcorr)):
            FP_probs.append(Probs[i]) 
            key = 'FP ' + str(FP_counter)
            FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]] 
            FP_counter+=1
            
    num_of_tumors = max_label - len(Isolated_Tumor_Cells);        ++++++
    return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary
 

def computeBootstrapSet(FROC_data, num_of_slides_to_count_FPs, number_of_files, num_of_bootstrap_samples = 1000):
    """Generates bootstrap sets from the original data  
    Args:
        total_fps_lists: List of false positives from all bootstrap instances
        total_sns_lists: List of sensitivities from all bootstrap instances
        
    Returns:
        total_fps_lists: List of false positives from all bootstrap instances
        total_sns_lists: List of sensitivities from all bootstrap instances
    """      
    # list of boot straped Fps and sensitivities
    total_fps_lists = []
    total_sns_lists = []
    
    # a single FROC bootstrap instance
    FROC_data_bootstrap = np.zeros((4, number_of_files), dtype=np.object)   
    for boot_strap_num in xrange(num_of_bootstrap_samples):
        rand_samples = np.random.randint(number_of_files, size=number_of_files)
        for samples in xrange(number_of_files):
            FROC_data_bootstrap[0][samples] = FROC_data[0][rand_samples[samples]]
            FROC_data_bootstrap[1][samples] = FROC_data[1][rand_samples[samples]]
            FROC_data_bootstrap[2][samples] = FROC_data[2][rand_samples[samples]]
            FROC_data_bootstrap[3][samples] = FROC_data[3][rand_samples[samples]]
        FPs_bootstrap, sensitivity_bootstrap, _ = computeFROC(FROC_data_bootstrap, num_of_slides_to_count_FPs)
        total_fps_lists.append(FPs_bootstrap)
        total_sns_lists.append(sensitivity_bootstrap)
    return total_fps_lists, total_sns_lists
    
def interpolateBootstrapedFROC_curves(total_fps_lists, total_sns_lists, num_of_bootstrap_samples):
    """Interpolates the FROC data from all bootstrap instances
    
    Args:
        total_fps_lists: List of false positives from all bootstrap instances
        total_sns_lists: List of sensitivities from all bootstrap instances
        
    Returns:
        all_fps:  Interpolate FPS
        interp_sens: Interpolated TPS
    """  
    FROC_minFP = 0.001
    FROC_maxFP = 8
    all_fps = np.linspace(FROC_minFP, FROC_maxFP, num=10000)
    
    # Interpolate all FROC curves at this points
    interp_sens = np.zeros((num_of_bootstrap_samples,len(all_fps)), dtype = 'float32')
    for i in range(num_of_bootstrap_samples):
        current_tps = total_sns_lists[i]
        current_fps = total_fps_lists[i]
        interp_sens[i,:] = np.interp(all_fps, current_fps[::-1], current_tps[::-1])
    return all_fps, interp_sens
        
        
def computeMeanCI(interp_sens, confidence = 0.95):
    """Generates the sensitivity data for the upper, average, and lower bounds of the 
    FROC plot with confidence interval
    
    Args:
        interp_sens: Interpolated sensitivities for all the boot stap instances
        confidence: the value for computation of confidence interval
        
    Returns:
        sens_mean:  Mean sensitivity points
        sens_lb: lower bound sensitivity data
        sens_up: upper bound sensitivity data
    """    
    sens_mean = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    sens_lb   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    sens_up   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    
    Pz = (1.0-confidence)/2.0
        
    for i in range(interp_sens.shape[1]):
        # get sorted vector
        vec = interp_sens[:,i]
        vec.sort()

        sens_mean[i] = np.average(vec)
        sens_lb[i] = vec[math.floor(Pz*len(vec))]
        sens_up[i] = vec[math.floor((1.0-Pz)*len(vec))]

    return sens_mean,sens_lb,sens_up  
    
def getCPM(fps, sens, all_probs, fixedFPs):
    """compute average sensitivity over multiple fixed false positive rates l
    
    Args:
        fps:   A list containing the average number of false positives
        per image for different thresholds
        
        sens:  A list containig overall sensitivity of the system
        for different thresholds
        
    Returns:
        mean_sens:  The average sensitivity at the given false positive points in FixedFPs
        fixed_sens: A list containing sensitivity values at each of the points in FixedFPs 
    """         
    ThreshProps = np.zeros(len(fixedFPs))
    fixed_sens = [0.0] * len(fixedFPs)
    for i,fixedFP in enumerate(fixedFPs):
        diffPrior = max(fps) # initialize with max number
        for j,fp in enumerate(fps):
            diffCurr = abs(fp-fixedFP)
            if j == 0:
                fixed_sens[i] = sens[j]
                ThreshProps[i] = all_probs[j]
                diffPrior = diffCurr  
            if diffCurr < diffPrior:
                fixed_sens[i] = sens[j]
                ThreshProps[i] = all_probs[j]
                diffPrior = diffCurr
              
    mean_sens = np.mean(fixed_sens)
    return mean_sens, fixed_sens, ThreshProps

    
def computeFROC(FROC_data, num_of_slides_to_count_FPs):
    """Generates the data required for plotting the FROC curve
    
    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image
         
    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds
        
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """
    
    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist] 
    
    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())    
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs)/float(num_of_slides_to_count_FPs)
    total_sensitivity = np.asarray(total_TPs)/float(sum(FROC_data[3]))      
    return  total_FPs, total_sensitivity, all_probs
   
   
def plotFROC(total_FPs, total_sensitivity, mean_sens_final_score, result_folder, team_name = 'Team Name', Website_ID = 'Team Name'):
    """Plots the FROC curve
    
    Args:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds
        
        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
         
    Returns:
        -
    """    
    fig = plt.figure()
    ax = plt.gca()
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    #plt.plot(total_FPs, total_sensitivity, label='FROC curve (Final score = %0.4f)' %mean_sens_final_score ) 
    plt.plot(total_FPs, total_sensitivity, label='FROC curve (Final score = %0.4f\n,Max Recall=%0.4f)' %(mean_sens_final_score,max(total_sensitivity))   ) 
    plt.grid(b=True, which='both')
    plt.xlabel('Average Number of False Positives', fontsize=12)
    plt.ylabel('Metastasis detection sensitivity', fontsize=12)  
    title = 'FROC curve - ' + team_name
    fig.suptitle(title, fontsize=12, position=(0.5, 1.0))      
    plt.legend(loc="lower right") 
    plt.ylim((0,1))
    plt.xlim((0,8))
    #team = team_name.replace("\n", "")
    plt.savefig(os.path.join(result_folder, "FROC_%s.png" % Website_ID), bbox_inches='tight', dpi=300) 
    #print "Max Recall = %.4f" %total_sensitivity[len(total_FPs)-1]


def plotFROCplotBootstrapedFROC(all_fps, sens_lb, sens_mean, sens_up, result_folder, team_name = 'Team Name'):
    """Plots the FROC curve with confidence interval
    
    Args:
        all_fps:      A list containing the average number of false positives
        per image for different thresholds
        sens_mean:  Mean sensitivity points
        sens_lb: lower bound sensitivity data
        sens_up: upper bound sensitivity data
        team_name: Group Name
    Returns:
        -
    """    
    fig1 = plt.figure()
    ax = plt.gca()
    clr = 'b'
    #plt.plot(total_FPs, total_sensitivity, color=clr, label="%s" % CADSystemName, lw=2)
    plt.plot(all_fps, sens_mean, label="mean")
    plt.plot(all_fps, sens_lb, color=clr, ls=':')
    plt.plot(all_fps, sens_up, color=clr, ls=':')
    ax.fill_between(all_fps, sens_lb, sens_up, facecolor=clr, alpha=0.05)        
    ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
    plt.ylim((0,1))
    plt.xlim((0,8))
    plt.xlabel('Average Number of False Positives', fontsize=12)
    plt.ylabel('Metastasis detection sensitivity', fontsize=12)  
    title = 'Free response receiver operating characteristic curve \n with 95% confidence interval - ' + team_name
    fig1.suptitle(title, fontsize=12, position=(0.5, 1.0))
    plt.grid(b=True, which='both')
    plt.savefig(os.path.join(result_folder, "FROC_with_CI_%s.png" % team_name), bbox_inches='tight', dpi=300)    
    
    
def plotROC(fpr, tpr, AUC_eval1, output, team_name = 'Team Name', Website_ID = 'Team Name'):
    """Plots the FROC curve with confidence interval
    
    Args:
        all_fps:      A list containing the average number of false positives
        per image for different thresholds
        sens_mean:  Mean sensitivity points
        sens_lb: lower bound sensitivity data
        sens_up: upper bound sensitivity data
        team_name: Group Name
    Returns:
        -
    """        
    print 'Final score of the system for evaluation one is: ' + str(AUC_eval1)
    fig1 = plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (AUC = %0.4f)' % AUC_eval1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    title = 'ROC curve - ' + team_name
    fig1.suptitle(title, fontsize=12, position=(0.5, 1.0))
    plt.grid(b=True, which='both')
    plt.legend(loc="lower right")   
    #team = team_name.replace("\n", "")
    plt.savefig(os.path.join(output, "ROC_%s.png" % Website_ID), bbox_inches='tight', dpi=300)
    
    
        
if __name__ == "__main__":

    # ground truth mask folder
    mask_folder = r'/media/CUDisk1/hjlin/Backup/Camelyon2016/2016ISBI/CAMELYON16/Testset/Ground_Truth/Masks'
    eval_mask_dir = r'/media/CUDisk1/hjlin/Backup/Camelyon2016/2016ISBI/CAMELYON16/Testset/precomputed_masks'
    # if you have not precomputed evaluation masks and stored it to disk load_eval should be false, see line 478
    load_eval = False
    # submission folder containing two folders 
    corepath = r"."
    output = corepath + r"/Result_Final"
    if not os.path.exists(output):
        os.mkdir(output)
        
    result_folder = corepath + r"/Evaluation2"
    # csv file should be named val1.csv
    csvDIR_eval1 = corepath + r"/Evaluation1/val1.csv"
    Website_ID = 'CULab'
    team_name = 'CULAB3'
    result_file_list = []
    result_file_list += [each for each in os.listdir(result_folder) if each.endswith('.csv')]
    
    CountOnlyInNeg = 1
    EVALUATION_MASK_LEVEL = 5 # Image level at which the evaluation is done
    L0_RESOLUTION = 0.243 # pixel resolution at level 0
    
    FROC_data = np.zeros((4, len(result_file_list)-2), dtype=np.object)
    FP_summary = np.zeros((2, len(result_file_list)-2), dtype=np.object)
    detection_summary = np.zeros((2, len(result_file_list)-2), dtype=np.object)
    
    caseNum = 0 
    number_of_negatives = 0
    for case in result_file_list:
        if (case[0:-4] != 'Test_114') and (case[0:-4] != 'Test_049'):
            print 'Evaluating Performance on image:', case[0:-4]
            sys.stdout.flush()
            csvDIR = os.path.join(result_folder, case)
            Probs, Xcorr, Ycorr = readCSVContent(csvDIR)
            maskDIR = os.path.join(mask_folder, case[0:-4]) + '_Mask.tif'
            is_tumor = os.path.isfile(maskDIR)     
            
            if (is_tumor):
                eval_mask_dir_file = os.path.join(eval_mask_dir, case[0:-4]) + '_Mask.npy'
                if not load_eval:
                    evaluation_mask = computeEvaluationMask(maskDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
                    np.save(eval_mask_dir_file, evaluation_mask)
                else:
                    evaluation_mask = np.load(eval_mask_dir_file)
                ITC_labels, all_tumor_centroids = computeITCList(evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
            else:
                evaluation_mask = 0
                ITC_labels = []
                all_tumor_centroids = {}
                number_of_negatives += 1
                
               
            FROC_data[0][caseNum] = case
            FP_summary[0][caseNum] = case
            detection_summary[0][caseNum] = case
            if (CountOnlyInNeg and is_tumor):
                _, FROC_data[2][caseNum], FROC_data[3][caseNum], detection_summary[1][caseNum], FP_summary[1][caseNum] = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels, all_tumor_centroids, EVALUATION_MASK_LEVEL) 
                FROC_data[1][caseNum] = []
            else:
                FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum], detection_summary[1][caseNum], FP_summary[1][caseNum] = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels, all_tumor_centroids, EVALUATION_MASK_LEVEL)

            
            XML_directory = os.path.join(output, case[0:-4]) + '.xml'
            Cut_thresh = 1
            #writeResultsToXML(detection_summary[1][caseNum], FP_summary[1][caseNum], XML_directory, Cut_thresh)
            caseNum += 1
    
    # Compute FROC curve 
    

    if (CountOnlyInNeg):
        num_of_slides_to_count_FPs = number_of_negatives
    else:
        num_of_slides_to_count_FPs = len(FROC_data[0])
        
        
    total_FPs, total_sensitivity, all_probs = computeFROC(FROC_data, num_of_slides_to_count_FPs)
    fixedFPs = [0.25, 0.5, 1, 2, 4, 8]

    mean_sens_final_score, fixed_sens, ThreshProps = getCPM(total_FPs, total_sensitivity, all_probs, fixedFPs)
    if (total_FPs[0]<8):
        total_FPs = np.insert(total_FPs, 0, 8)
        total_sensitivity = np.insert(total_sensitivity, 0, total_sensitivity[0])
        
    plotFROC(total_FPs, total_sensitivity, mean_sens_final_score, output, team_name, Website_ID)

#    # Computing ROC    
    eval1_probs = readCSVContentEval1(csvDIR_eval1) 
    eval1_probs_final = []
    GT_Labels = []
    caseNum = 0    
    for case in result_file_list:
        if True:#(case[0:-4] != 'Test_049'):
            print 'Evaluating Performance on image:', case[0:-4]
            sys.stdout.flush()
            maskDIR = os.path.join(mask_folder, case[0:-4]) + '_Mask.tif'
            is_tumor = os.path.isfile(maskDIR) 
            eval1_probs_final.append(eval1_probs[case[0:-4]])
            if (is_tumor):
                GT_Labels.append(1)
            else:
                GT_Labels.append(0)
        caseNum += 1                
    
    fpr, tpr, _ = roc_curve(GT_Labels, eval1_probs_final)          
    AUC_eval1 = roc_auc_score(GT_Labels, eval1_probs_final)    
    plotROC(fpr, tpr, AUC_eval1, output, team_name, Website_ID)

      
    csv_write_dir = os.path.join(output, "scores_%s.csv" % Website_ID)  
    CSV_content = [['FPs/WSI', 'Sensitivity'], ['0.25',fixed_sens[0]], ['0.5', fixed_sens[1]] , ['1',  fixed_sens[2]] , ['2', fixed_sens[3]], ['4',fixed_sens[4]], ['8',fixed_sens[5]]]
    with open(csv_write_dir, 'wb') as testfile:
        csv_writer = csv.writer(testfile)
        for y in range(len(CSV_content[0])):
            csv_writer.writerow([x[y] for x in CSV_content])
        
        
        
