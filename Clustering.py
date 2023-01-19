import pydicom
import os
import fnmatch
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from time import perf_counter
import nibabel as nib
import csv
import os
from os.path import exists
import numpy as np
from time import perf_counter



# This function reads the dicom files of the PET image and extracts the necessary TACs for the analysis
# file_path is a path to the PET image file which has been chosen to be clustered
def prepare_TACs(file_path):

    start_time = perf_counter()  # start_time and end_time record time spent on fetching image and extracting TACs
    tac_array = []  # array for extracted TACs
    tac_array_inds = []  # array for indexes of TACs, i.e. the (x,y,z) placings

    file_list = fnmatch.filter(os.listdir(file_path), '*.dcm')  # file_list includes all dicom files in the image folder

    tmp_file = pydicom.dcmread(file_path + '/' + file_list[0])  # this opens the first dicom file in order to extract the dimensions of the image
    x_dim = tmp_file[0x0028, 0x0010].value      # Rows
    y_dim = tmp_file[0x0028, 0x0011].value      # Columns
    z_dim = tmp_file[0x0054, 0x0081].value      # Number of Slices
    frame_dim = tmp_file[0x0054, 0x0101].value  # Number of Time Slices

    # Check dimension and file numbers
    if(z_dim * frame_dim != len(file_list)):
        print('Dimension does not match!')

    image_data = np.empty([x_dim, y_dim, z_dim, frame_dim])  # Allocate memory for image data

    # Read all image files in file_list
    for k in range(len(file_list)):

        tmp_file = pydicom.dcmread(file_path + '/' + file_list[k])  # read dicom file
        image_index = tmp_file[0x0054, 0x1330].value  # Image Index
        tmp_image_data = tmp_file.pixel_array * tmp_file[0x0028, 0x1053].value + tmp_file[0x0028, 0x1052].value # voxel values

        # Find z index
        z_ind = image_index % z_dim
        if z_ind == 0:
            z_ind = z_dim
        z_ind = z_ind - 1

        # Find frame number
        frame_ind = np.floor((image_index - 1) /z_dim).astype(int)

        # Store voxel values to image_data
        for x_ind in range(x_dim):
            for y_ind in range(y_dim):
                image_data[x_ind, y_ind, z_ind, frame_ind] = tmp_image_data[x_ind, y_ind]

    # Define binary mask to include only voxels with values bigger than
    mean_per_voxel = np.mean(image_data, axis=3)
    binary_mask_for_mean = mean_per_voxel > np.mean(mean_per_voxel)

    # Gather TACs and their indexes
    for x_ind in range(x_dim):
        for y_ind in range(y_dim):
            for z_ind in range(z_dim):
                if binary_mask_for_mean[x_ind, y_ind, z_ind]:
                    tac_array.append(image_data[x_ind, y_ind, z_ind, :])
                    tac_array_inds.append((x_ind, y_ind, z_ind))

    end_time = perf_counter()
    print("Preparing the image took " + str(round(end_time - start_time, 0)) + " seconds" + "\n")

    return tac_array, tac_array_inds


# This function executes the clusterings with PCA+KMeans, ICA+KMeans and Gaussian mixture model
# 'image' includes the filtered TACs from the specific PET image that come from prepare_TACs()
# 'perform_pca', 'perform_ica' and 'perform_gmm' are boolean type. It ables you to choose which clustering methods are executed.
# 'cluster_number' is a list with number of clusters for each method
# returns 'label_list' which includes the cluster labels for each method. Having the same label number means that the TACs belong to the same cluster
def clustering_TACs(image, perform_pca, perform_ica, perform_gmm, cluster_number):

    label_list = []  # stores the label arrays from each clustering method
    processing_time = []  # stores the processing times for each clustering

    # PCA+KMeans clustering
    if perform_pca:
        print("PCA+KM")
        PCA_start = perf_counter()  # PCA_start and PCA_end record time spent on PCA and KMeans
        PCA_clustering = PCA(n_components=cluster_number[0]).fit_transform(image)  # Principal Component Analysis
        PCA_labels = KMeans(n_clusters=cluster_number[0]).fit_predict(PCA_clustering)  # KMeans clustering
        PCA_end = perf_counter()
        print(str(round(PCA_end - PCA_start, 0)) + " seconds" + "\n")
        processing_time.append(round(PCA_end - PCA_start, 0))
        label_list.append(PCA_labels)  # save labels

    # ICA+KMeans
    if perform_ica:
        print("ICA+KM")
        ICA_start = perf_counter()  # ICA_start and ICA_end record time spent on ICA and KMeans
        ICA_clustering = FastICA(n_components=cluster_number[1]).fit_transform(image)  # Independent Component Analysis
        ICA_labels = KMeans(n_clusters=cluster_number[1]).fit_predict(ICA_clustering)  # KMeans clustering
        ICA_end = perf_counter()
        print(str(round(ICA_end - ICA_start, 0)) + " seconds" + "\n")
        processing_time.append(round(ICA_end - ICA_start, 0))
        label_list.append(ICA_labels)  # save labels

    # Gaussian mixture model
    if perform_gmm:
        print("GMM")
        GMM_start = perf_counter()  # GMM_start and GMM_end record time spent on PCA and KMeans

        # GMM might not converge if the cluster number is high. The clustering is done again until it succeeds
        no_success = True
        while no_success:
            try:
                gmm_model = GaussianMixture(n_components=cluster_number[2]).fit(np.array(image))  # Gaussian mixture model clustering
                GMM_labels = gmm_model.predict(image)  # extract labels from the model
                GMM_end = perf_counter()
                print(str(round(GMM_end - GMM_start, 0)) + " seconds" + "\n")
                processing_time.append(round(GMM_end - GMM_start, 0))
                label_list.append(GMM_labels)  # save labels
                no_success = False
            except:
                # if error occurs, try again
                print('GMM was not successful')

    return label_list, processing_time



# This function compares the clusters created by one clustering method with manually segmented masks and saves results to a csv file
# 'image' includes the filtered TACs from the specific PET image that come from prepare_TACs()
# 'image_inds' are the indexes of TACs that come from prepare_TACs()
# 'labels' are the labels from a clustering with one of the chosen methods
# 'manual_nifti_list' includes manual segmentations
# 'manual_TAC_list' includes the average TACs of manual segmentations
# 'clustering_time' includes the processing times of each clustering method
# 'respath' indicates where the results are to be saved
# 'image_name' is the name of the PET image and it will take part in naming the csv file
# 'method_name' is the name of the clustering method and it will take part in naming the csv file
# 'number_of_clusters' is the number of cluster used in clustering and it will take part in naming the csv file
def cluster_vs_manual_mask(image, image_inds, labels, manual_nifti_list, manual_TAC_list,
                           clustering_time, respath, image_name, method_name, number_of_clusters):

    # turn the nifti images so that their dimensions and directions are the same way as the original dicom image
    heart_mask = np.transpose(np.flip(manual_nifti_list[0], 2), (1, 0, 2))
    brain_mask = np.transpose(np.flip(manual_nifti_list[1], 2), (1, 0, 2))
    kidney_mask = np.transpose(np.flip(manual_nifti_list[2], 2), (1, 0, 2))
    manual_masks = [heart_mask, brain_mask, kidney_mask]

    # Unique labels of the clusters
    cluster_label = list(np.unique(labels))

    jaccard_index = [0, 0, 0]  # create list for saving best jaccard indexes
    # create cluster image from each cluster label
    for k in range(len(cluster_label)):
        indexes = np.where(labels == cluster_label[k])  # fetch indexes of the same label
        mask_image = np.empty([heart_mask.shape[0], heart_mask.shape[1], heart_mask.shape[2]])
        sum_for_average = np.zeros([len(image[0])])
        for ind in indexes[0]:
            tmp = image_inds[ind]
            mask_image[tmp[0], tmp[1], tmp[2]] = 1
            sum_for_average = sum_for_average + image[ind]

        # calculate TP, FP and FN
        ROIs = ['Heart', 'Brain', 'Kidney']
        for roi_index in range(len(ROIs)):
            TP = 0
            FN = 0
            FP = 0
            for x_array in range(len(mask_image)):
                for y_array in range(len(mask_image[x_array])):
                    if (np.all(manual_masks[roi_index][x_array, y_array, :] == 0) and np.all(
                            mask_image[x_array, y_array, :] == 0)):
                        continue
                    else:
                        for z_array in range(len(manual_masks[roi_index][x_array, y_array, :])):
                            if manual_masks[roi_index][x_array, y_array, z_array] != 0:
                                if mask_image[x_array, y_array, z_array] != 0:
                                    TP = TP + 1
                                else:
                                    FP = FP + 1
                            else:
                                if mask_image[x_array, y_array, z_array] != 0:
                                    FN = FN + 1
            # calculate jaccard
            # update the best cluster and the best jaccard score if they're better than the previous chosen cluster and jaccard score for the ROI
            if TP / (TP + FP + FN) > jaccard_index[roi_index]:
                if ROIs[roi_index] == 'Heart':
                    heart_cluster = mask_image
                    heart_ave = sum_for_average / len(indexes[0])
                    jaccard_index[roi_index] = TP / (TP + FP + FN)
                    heart_TP = TP
                    heart_FN = FN
                    heart_FP = FP
                if ROIs[roi_index] == 'Brain':
                    brain_cluster = mask_image
                    brain_ave = sum_for_average / len(indexes[0])
                    jaccard_index[roi_index] = TP / (TP + FP + FN)
                    brain_TP = TP
                    brain_FN = FN
                    brain_FP = FP
                if ROIs[roi_index] == 'Kidney':
                    kidney_cluster = mask_image
                    kidney_ave = sum_for_average / len(indexes[0])
                    jaccard_index[roi_index] = TP / (TP + FP + FN)
                    kidney_TP = TP
                    kidney_FN = FN
                    kidney_FP = FP

    # make results
    cluster_masks = [heart_cluster, brain_cluster, kidney_cluster]
    averages = [heart_ave, brain_ave, kidney_ave]
    rmse_list = []
    for roi in range(len(ROIs)):
        cluster = cluster_masks[roi]
        manual = manual_masks[roi]

        # find image slice with most voxels included to mask
        pre_count = np.count_nonzero(manual[0, :, :])
        slice_number = 0
        for slice_index in range(1, heart_load.shape[0]):
            if np.count_nonzero(manual[slice_index, :, :]) > pre_count:
                pre_count = np.count_nonzero(manual[slice_index, :, :])
                slice_number = slice_index

        # save a slice of manual mask to nifti, if it hasn't been saved already
        if not exists(respath + '/' + image_name + '_manual_' + ROIs[roi] + '_' + str(slice_number) + '.nii'):
            cluster_image = nib.Nifti1Image(manual[slice_number, :, :], np.eye(4))
            nib.save(cluster_image, respath + '/' + image_name + '_manual_' + ROIs[roi] + '_' + str(slice_number))

        # save a slice of cluster mask to nifti
        cluster_image = nib.Nifti1Image(cluster[slice_number, :, :], np.eye(4))
        nib.save(cluster_image, respath + '/' + image_name + '_' + method_name + '_' + str(number_of_clusters) + '_' +
                 ROIs[roi] + '_' + str(slice_number))

        # RMSE
        rmse = mean_squared_error(manual_TAC_list[roi], averages[roi], squared=False)
        rmse_list.append(round(rmse, 2))

    # gather the resulting data into a dataframe
    data_heart = ['Heart', round(jaccard_index[0], 2), heart_TP,
                  len(heart_mask) * len(heart_mask[0]) * len(heart_mask[0][0]) - heart_TP - heart_FN - heart_FP,
                  heart_FP, heart_FN, rmse_list[0], ' ', 'TAC [Bq/ml]']
    data_heart.extend(averages[0])
    data_heart.extend([' ', clustering_time])

    data_brain = ['Brain', round(jaccard_index[1], 2), brain_TP,
                  len(brain_mask) * len(brain_mask[0]) * len(brain_mask[0][0]) - brain_TP - brain_FN - brain_FP,
                  brain_FP, brain_FN, rmse_list[1], ' ', 'TAC [Bq/ml]']
    data_brain.extend(averages[1])
    data_brain.extend([' ', ' '])

    data_kidney = ['Kidney', round(jaccard_index[2], 2), kidney_TP,
                   len(kidney_mask) * len(kidney_mask[0]) * len(kidney_mask[0][0]) - kidney_TP - kidney_FN - kidney_FP,
                   kidney_FP, kidney_FN, rmse_list[2], ' ', 'TAC [Bq/ml]']
    data_kidney.extend(averages[2])
    data_kidney.extend([' ', ' '])

    tac_dataframe = pd.DataFrame({image_name: data_heart, method_name: data_brain, str(number_of_clusters): data_kidney})

    # create first column
    time_column = [' ', 'Jaccard Index', 'Number of True Positives', 'Number of True Negatives',
                   'Number of False Positives', 'Number of False Negatives', 'RMSE between TACs', ' ']
    time_column.extend(manual_TAC_list[3])
    time_column.extend([' ', 'Processing time(s)'])

    # insert column to be the first column in the dataframe
    tac_dataframe.insert(0, 'Image, method and number of clusters', time_column, True)

    # Save to csv
    resfile = respath + "/" + image_name + '_' + method_name + '_' + str(number_of_clusters) + '.csv'
    tac_dataframe.to_csv(resfile, index=False)


# Define paths
datapath = "path/to/my/data"
validationpath = "/path/to/gold_standard"
respath = "/where/I/want/results"

# Define which images to analyse and how many clusters to use
image_name = [FILL YOUR IMAGE IDS HERE]
number_of_clusters = [25, 22, 18]  # See DecideClusterNumber.py
datafolders = ["PT_" + img + ".pet" for img in image_name]

# Analyse the 30 images
for file_index in range(len(image_name)):
    tac_data, tac_data_inds = prepare_TACs(datapath + '/' + datafolders[file_index])

    # import manual TACs
    os.chdir(validationpath + '/' + image_name[file_index])
    TACs = pd.read_excel(image_name[file_index] + '.xlsx')
    heart_TAC = TACs['Heart\n(Bq/ml)'][4:].to_numpy()
    brain_TAC = TACs['Brain\n(Bq/ml)'][4:].to_numpy()
    kidney_TAC = TACs['Kidney\n(Bq/ml)'][4:].to_numpy()
    times = TACs['Unnamed: 0'][3:]
    TAC_list = [heart_TAC, brain_TAC, kidney_TAC, times]

    # import niftis of manual segmentations
    heart_load = nib.load(image_name[file_index] + '_heart.img').get_fdata()
    brain_load = nib.load(image_name[file_index] + '_brain.img').get_fdata()
    kidney_load = nib.load(image_name[file_index] + '_kidney.img').get_fdata()
    load_list = [heart_load, brain_load, kidney_load]

    # cluster and evaluate
    label_list, process_time = clustering_TACs(tac_data, True, True, True, number_of_clusters)
    label_file = respath + "/" + image_name[file_index] + "_Labels.npy"
    np.save(label_file, label_list)
    print('clustering ready')
    print('PCA results')
    cluster_vs_manual_mask(tac_data, tac_data_inds, label_list[0], load_list, TAC_list, process_time[0],
                           respath, image_name[file_index], 'PCA', number_of_clusters[0])
    print('ICA results')
    cluster_vs_manual_mask(tac_data, tac_data_inds, label_list[1], load_list, TAC_list, process_time[1],
                           respath, image_name[file_index], 'ICA', number_of_clusters[1])
    print('GMM results')
    cluster_vs_manual_mask(tac_data, tac_data_inds, label_list[2], load_list, TAC_list, process_time[2],
                           respath, image_name[file_index], 'GMM', number_of_clusters[2])

