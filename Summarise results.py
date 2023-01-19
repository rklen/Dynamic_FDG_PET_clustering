import os
from os.path import exists
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


# Helper function that extract the Jaccard indices from all images, cluster sizes, and methods for the given organ
def ExtractJaccard(respath, clustersizes, methods, organ):

    all_content = os.listdir(respath)
    images = [name for name in all_content if os.path.isdir(os.path.join(respath, name))]

    # Initialise Jaccard table (cols: PCA_5, ICA_5, GMM_5, PCA_10, ICA_10, GMM_10, PCA_15, ...)
    jaccard = np.empty((len(images), len(clustersizes) * len(methods)), "double")
    methodprefix = methods * len(clustersizes)
    clustersuffix = [element for element in clustersizes for i in range(len(methods))]
    colnames = [i + "_" + str(j) for i, j in zip(methodprefix, clustersuffix)]
    rownames = images

    # Fill the Jaccard table
    for i in range(len(images)):
        colindex = 0
        for c in clustersizes:
            for m in methods:
                filename = images[i] + "_" + m + "_" + str(c) + ".csv"
                filepath = respath + "/" + images[i] + "/" + filename
                if exists(filepath):
                    fullres = pd.read_csv(filepath)
                    organjaccards = list(fullres.iloc[1])
                    organindex = list(fullres.iloc[0]).index(organ)
                    jaccard[i, colindex] = float(organjaccards[organindex])
                else:
                    jaccard[i, colindex] = np.NaN
                colindex = colindex + 1

    # Convert to pandas data frame with named rows and columns
    jaccard_table = pd.DataFrame(jaccard, columns=colnames, index=rownames)
    return jaccard_table



def ExtractProcessingTime(respath, clustersizes, methods):

    all_content = os.listdir(respath)
    images = [name for name in all_content if os.path.isdir(os.path.join(respath, name))]

    # Initialise Jaccard table (cols: PCA_5, ICA_5, GMM_5, PCA_10, ICA_10, GMM_10, PCA_15, ...)
    times = np.empty((len(images), len(clustersizes) * len(methods)), "double")
    methodprefix = methods * len(clustersizes)
    clustersuffix = [element for element in clustersizes for i in range(len(methods))]
    colnames = [i + "_" + str(j) for i, j in zip(methodprefix, clustersuffix)]
    rownames = images

    # Fill the Jaccard table
    for i in range(len(images)):
        colindex = 0
        for c in clustersizes:
            for m in methods:
                filename = images[i] + "_" + m + "_" + str(c) + ".csv"
                filepath = respath + "/" + images[i] + "/" + filename
                if exists(filepath):
                    fullres = pd.read_csv(filepath)
                    timerow = list(fullres.iloc[60])
                    times[i, colindex] = float(timerow[1])
                else:
                    times[i, colindex] = np.NaN
                colindex = colindex + 1

    # Convert to pandas data frame with named rows and columns
    times_table = pd.DataFrame(times, columns=colnames, index=rownames)
    return times_table



##### The 30 test images

# Define where to find the image result files and what methods and cluster sizes to look for
respath = "path/to/my/results"
methods = ["PCA", "ICA", "GMM"]
clustersizes = [18, 22, 25]

# Collect Jaccard indices for different donors
jaccard_heart = ExtractJaccard(respath, clustersizes, methods, "Heart")
jaccard_brain = ExtractJaccard(respath, clustersizes, methods, "Brain")
jaccard_kidney = ExtractJaccard(respath, clustersizes, methods, "Kidney")

# Combine into one table
jaccard_combined = jaccard_heart.to_numpy()
jaccard_combined[:, 0:3] = jaccard_heart.loc[:, ["PCA_25", "ICA_22", "GMM_18"]]
jaccard_combined[:, 3:6] = jaccard_brain.loc[:, ["PCA_25", "ICA_22", "GMM_18"]]
jaccard_combined[:, 6:9] = jaccard_kidney.loc[:, ["PCA_25", "ICA_22", "GMM_18"]]

# Calculate means over images
jaccard_combined.mean(axis=0)
# PCA_heart: 0.48678571, ICA_heart: 0.37571429, GMM_heart: 0.43035714,
# PCA_brain: 0.17107143, ICA_brain: 0.02214286, GMM_brain: 0.10142857,
# PCA_kidney: 0.22357143, ICA_kidney: 0.0375, GMM_kidney: 0.27535714

# Save collected Jaccard indices
col_names = ["PCA_heart", "ICA_heart", "GMM_heart", "PCA_brain", "ICA_brain", "GMM_brain",
             "PCA_kidney", "ICA_kidney", "GMM_kidney"]
jaccard = pd.DataFrame(jaccard_combined, columns=col_names, index=list(jaccard_heart.index))
jaccard.to_csv(respath + "/Jaccard30.csv")

# Collect processing times for different images and drop NA columns
times = ExtractProcessingTime(respath, clustersizes, methods)
times = times.dropna(axis=1, how="all")
times.mean()  # GMM: 1318.166667, ICA: 57.300000, PCA: 79.466667
times.std()   # GMM: 300.532296, ICA: 6.998276, PCA: 13.860719
times.to_csv(respath + "/ProcessingTimes30.csv")

###### Statistical testing of obtained Jaccards

from scipy import stats

jaccard = pd.read_csv("path/to/my/results/Jaccard30.csv", index_col=0)

# Initialise p-value table
pvals = np.full(fill_value=1.0, shape=(3, 3))
colnames = ["heart", "brain", "kidney"]
rownames = ["PCA_vs_ICA", "PCA_vs_GMM", "ICA_vs_GMM"]

# Calculate p-values using Wilcoxon signed-rank test
pvals[0, 0] = stats.wilcoxon(jaccard.loc[:, "PCA_heart"], jaccard.loc[:, "ICA_heart"]).pvalue    # < 0.05
pvals[0, 1] = stats.wilcoxon(jaccard.loc[:, "PCA_brain"], jaccard.loc[:, "ICA_brain"]).pvalue    # < 0.05
pvals[0, 2] = stats.wilcoxon(jaccard.loc[:, "PCA_kidney"], jaccard.loc[:, "ICA_kidney"]).pvalue  # < 0.05
pvals[1, 0] = stats.wilcoxon(jaccard.loc[:, "PCA_heart"], jaccard.loc[:, "GMM_heart"]).pvalue    # < 0.05
pvals[1, 1] = stats.wilcoxon(jaccard.loc[:, "PCA_brain"], jaccard.loc[:, "GMM_brain"]).pvalue    # < 0.05
pvals[1, 2] = stats.wilcoxon(jaccard.loc[:, "PCA_kidney"], jaccard.loc[:, "GMM_kidney"]).pvalue  # >= 0.05
pvals[2, 0] = stats.wilcoxon(jaccard.loc[:, "ICA_heart"], jaccard.loc[:, "GMM_heart"]).pvalue    # >= 0.05
pvals[2, 1] = stats.wilcoxon(jaccard.loc[:, "ICA_brain"], jaccard.loc[:, "GMM_brain"]).pvalue    # < 0.05
pvals[2, 2] = stats.wilcoxon(jaccard.loc[:, "ICA_kidney"], jaccard.loc[:, "GMM_kidney"]).pvalue  # < 0.05

# Convert into data frame and save p-values
ptable = pd.DataFrame(pvals, columns=colnames, index=rownames)
ptable.to_csv("path/Pvalues.csv")

###### Proportion of clustered voxels

images = [FILL IMAGE IDS HERE]
resprefix = "result/path/"

# Extract number and proportion of clustered voxels from each image
clustered_voxels = []
clustered_proportion = []
for i in images:
    filename = resprefix + i + "/" + i + "_Labels.npy"
    clusters = np.load(filename)
    clustered = clusters.shape[1]
    clustered_voxels.append(clustered)
    clustered_proportion.append(clustered/2605056)

# Calculate mean proportion
np.mean(clustered_proportion)  # 0.2035628536712198
np.std(clustered_proportion)   # 0.016227963293128846

# Save np.array(a).T.tolist()
voxel_summary = pd.DataFrame(np.array([clustered_voxels, clustered_proportion]).T,
                             columns=["ClusteredVoxels", "ClusteredVoxelProportion"],
                             index=images)
voxel_summary.to_csv("result/path/ClusteredVoxelInfo.csv")