# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 21:16:25 2017

@author: kawashimat, modified by jfriedrich
"""

import matplotlib.pyplot as plt
import numpy as np
from tifffile import imread
from sys import path, argv
path.append('functions/')
import Volt_Imfunctions as im
import Volt_ROI as ROI

# call as:  python Get_Timecourse_JF.py [dataset 1-3] [use_NMF 0/1]
try:
    dataset = ['06152017Fish1-2/', '10192017Fish2-1/', '10192017Fish3-1/'][int(argv[1]) - 1]
except:
    dataset = '06152017Fish1-2/'
try:
    use_NMF = bool(int(argv[2]))
except:
    use_NMF = False

pathname = '/mnt/home/jfriedrich/voltage/' + dataset
datadir = pathname + 'data_local/'
imdir = pathname + ('figNMF/' if use_NMF else 'fig/')

stack_fname = datadir + "registered.tif"  # "registered-partial.tif"


stack = imread(stack_fname)
image_len = stack.shape[0]
ave = stack.mean(axis=0).astype('float32')

ROI_list = np.load(datadir + "ROI_info.npy")['pixel_yx_list']
ROI_list = list(map(np.transpose, ROI_list))
nROI = len(ROI_list)


try:  # load results if existent
    ROI_info = np.load(imdir + "ROI_info_optimized.npy")
except:  # generate and save results
    for n in range(0, nROI):
        print("Neuron %d" % n)
        inds = ROI_list[n]
        if n == 0:
            ROI_info = ROI.optimize_trace(stack, ave, (inds[0], inds[1]), use_NMF=use_NMF)
        else:
            ROI_info = np.append(ROI_info, ROI.optimize_trace(stack, ave, (inds[0], inds[1]),
                                                              use_NMF=use_NMF))
    np.save(imdir + "ROI_info_optimized.npy", ROI_info)

T = len(stack)
dims = stack.shape[1:]


############################################################################

# import pandas as pd
# import scipy

# ROIs = np.zeros((nROI,) + dims, dtype='float32')
# for n in range(0, nROI):
#     ROIs[n, ROI_info['ROI_Y'][n], ROI_info['ROI_X'][n]] = ROI_info['Weight_final'][n]
# ROIs /= np.maximum(np.linalg.norm(ROIs.reshape(nROI, -1), 2, 1), 1e-9)[:, None, None]

# overlap = list([np.where(x > 0)[0]
#                 for x in ((ROIs.reshape(nROI, -1)
#                            .dot(ROIs.reshape(nROI, -1).T) > 0) - np.eye(nROI))])

# Yr = stack.reshape(T, -1)


# #######################################
# # low rank decomposition
# def blockCD(Yr, A, C=None, iters=5):
#     U = A.dot(Yr.T)
#     V = A.dot(A.T)
#     if C is None:
#         C = A[:-1].dot((Yr - Yr.mean(0).astype('float32')).T)
#         C = np.concatenate([C, U[-1:]])
#     for _ in range(iters):
#         # Cold = C.copy()
#         for m in np.where(V.diagonal())[0]:  # range(len(U)):  # neurons and background
#             C[m] += (U[m] - V[m].dot(C)) / V[m, m]
#             # C[m] = np.maximum(C[m], 0)
#         # print(np.linalg.norm(C-Cold))
#     return C


# use_blockCD = False
# # get traces
# b = stack.mean(0).astype('float32').ravel()
# b /= np.sqrt(b.dot(b))
# A = np.concatenate([ROIs.reshape(nROI, -1), b.reshape(1, -1)])
# if use_blockCD:
#     C = blockCD(Yr, A)
# else:
#     C = scipy.linalg.lstsq(A.T, Yr.T)[0]  # numpy would be very slow !!!
# # update background shape
# U = C.dot(Yr)
# V = C.dot(C.T)
# for _ in range(5):
#     A[-1] += ((U[-1] - V[-1].dot(A)) / V[-1, -1])
# # update traces
# if use_blockCD:
#     C = blockCD(Yr, A)
# else:
#     C = scipy.linalg.lstsq(A.T, Yr.T)[0]  # numpy would be very slow !!!

# B = stack - C[:-1].T.dot(A[:-1]).reshape(stack.shape)

# for n in range(0, nROI):
#     print("Neuron %d" % n)
#     inds = ROI_list[n]
#     if n == 0:
#         if len(overlap[n]):
#             ROI_info2 = ROI.optimize_trace(
#                 B + np.outer(C[n], A[n]).reshape(stack.shape), ave, (inds[0], inds[1]))
#         else:
#             ROI_info2 = ROI_info[n]
#     else:
#         if len(overlap[n]):
#             ROI_info2 = np.append(ROI_info2, ROI.optimize_trace(
#                 B + np.outer(C[n], A[n]).reshape(stack.shape), ave, (inds[0], inds[1])))
#         else:
#             ROI_info2 = np.append(ROI_info2, ROI_info[n])


# #######################################
# # estimate background from projection of rawdata on shape

# tcourses = ROIs.reshape(nROI, -1).dot(Yr.T)
# kernel = np.ones((51,)) / 51
# divider = np.convolve(np.ones(T), kernel, mode='same')
# tcourses_detrend = np.zeros_like(tcourses)
# for i in range(nROI):
#     tcourses_detrend[i] = np.convolve(tcourses[i], kernel, mode='same') / divider
# tcourses_zeroed = tcourses - tcourses_detrend

# B = stack - tcourses_zeroed.T.dot(ROIs.reshape(nROI, -1)).reshape(stack.shape)

# for n in range(0, nROI):
#     print("Neuron %d" % n)
#     inds = ROI_list[n]
#     if n == 0:
#         if len(overlap[n]):
#             ROI_info2 = ROI.optimize_trace(
#                 B + np.outer(tcourses_zeroed[n], ROIs[n].ravel())
#                 .reshape(stack.shape), ave, (inds[0], inds[1]))
#         else:
#             ROI_info2 = ROI_info[n]
#     else:
#         if len(overlap[n]):
#             ROI_info2 = np.append(ROI_info2, ROI.optimize_trace(
#                 B + np.outer(tcourses_zeroed[n], ROIs[n].ravel())
#                 .reshape(stack.shape), ave, (inds[0], inds[1])))
#         else:
#             ROI_info2 = np.append(ROI_info2, ROI_info[n])

# #######################################
# # estimate background from projection of rawdata on shape

# tcourses = ROIs.reshape(nROI, -1).dot(Yr.T)
# tcourses_detrend = np.zeros_like(tcourses)
# for i in range(nROI):
#     tcourses_detrend[i] = np.array(pd.Series(tcourses[i]).rolling(
#         window=150, min_periods=75, center=True).quantile(0.8))
# tcourses_zeroed = tcourses - tcourses_detrend

# B = stack - tcourses_zeroed.T.dot(ROIs.reshape(nROI, -1)).reshape(stack.shape)

# for n in range(0, nROI):
#     print("Neuron %d" % n)
#     inds = ROI_list[n]
#     if n == 0:
#         if len(overlap[n]):
#             ROI_info3 = ROI.optimize_trace(
#                 B + np.outer(tcourses_zeroed[n], ROIs[n].ravel())
#                 .reshape(stack.shape), ave, (inds[0], inds[1]))
#         else:
#             ROI_info3 = ROI_info[n]
#     else:
#         if len(overlap[n]):
#             ROI_info3 = np.append(ROI_info3, ROI.optimize_trace(
#                 B + np.outer(tcourses_zeroed[n], ROIs[n].ravel())
#                 .reshape(stack.shape), ave, (inds[0], inds[1])))
#         else:
#             ROI_info3 = np.append(ROI_info3, ROI_info[n])


############################################################################


active_cell = ROI_info['active']
active_n = 0
plt.figure(1, figsize=(16, 8))
for i, t in enumerate(ROI_info):
    plt.subplot(1, 2, 1)
    plt.ylim(-0.1, 0.2 * nROI + 0.1)
    if t['active'] == 0:
        plt.plot(np.arange(len(t['norm_tcourse1'])) / 300,
                 t['norm_tcourse1'] - 1 + 0.2 * i, color=(0.3, 0.3, 0.3))
    else:
        plt.plot(np.arange(len(t['norm_tcourse1'])) / 300,
                 t['norm_tcourse1'] - 1 + 0.2 * i, color=(1, 0, 0))

    if t['active'] == 1:
        active_n += 1
        tlimit = t['tlimit2']
        spikes = np.where(t['spike_tcourse2'] > 0)[0]

        plt.subplot(1, 2, 2).plot(np.arange(tlimit) / 300,
                                  t['norm_tcourse2'][:tlimit] - 1 + 0.2 * (active_n - 1))
        plt.subplot(1, 2, 2).plot(np.arange(tlimit, len(t['norm_tcourse2'])) / 300,
                                  t['norm_tcourse2'][tlimit:len(t['norm_tcourse2'])] -
                                  1 + 0.2 * (active_n - 1), color=(0.3, 0.3, 0.3))

        for s in range(len(spikes)):
            plt.subplot(1, 2, 2).plot(spikes[s] / 300, 0.15 +
                                      0.2 * (active_n - 1), 'ko', markersize=1)
        plt.ylim(-0.1, 0.2 * active_n + 0.1)

plt.tight_layout()
plt.savefig(imdir + "activity_timecourse.pdf")


active_cells = np.where(active_cell)[0]
active_tcourse = np.zeros((active_n, image_len))
active_spike_tcourse = np.zeros((active_n, image_len))

for i in range(len(active_cells)):
    active_tcourse[i, :] = ROI_info[active_cells[i]]['norm_tcourse2']
    active_spike_tcourse[i, :] = ROI_info[active_cells[i]]['spike_tcourse2']

np.save(imdir + "active_tcourse.npy", active_tcourse)
np.save(imdir + "active_spike_tcourse.npy", active_spike_tcourse)


plt.figure(2, figsize=(18, 6))
plt.subplot(131).imshow(im.imNormalize(ave.astype('float'), 99.9), cmap='gray')

img_color = np.tile(im.imNormalize(ave.astype('float'), 99.9)[:, :, None], (1, 1, 3))
for n in range(len(active_cells)):
    inds = ROI_list[active_cells[n]]
    img_color[inds[0].astype('int'), inds[1].astype('int'), 0] = 1
    img_color[inds[0].astype('int'), inds[1].astype('int'), 1] = 0
    img_color[inds[0].astype('int'), inds[1].astype('int'), 2] = 0

plt.subplot(132).imshow(img_color)

ax = plt.subplot(133)
tmp = np.zeros(ave.shape)
for n in range(len(active_cells)):
    inds = ROI_list[active_cells[n]]
    tmp[inds[0].astype('int'), inds[1].astype('int')] = 1

plt.imshow(tmp, cmap='gray')

for n in range(len(active_cells)):
    inds = ROI_list[active_cells[n]]
    location = (inds[0].mean(), inds[1].mean())
    ax.text(location[1], location[0], n, fontsize=12, color='r')

plt.tight_layout()
plt.savefig(imdir + "active_ROI.pdf")


f = plt.figure(3, figsize=(12, 6))
nrow = np.ceil(len(active_cells) / 5)
for c, ind in enumerate(active_cells):
    t = ROI_info[ind]
    if t['active'] == 1:
        spikes = np.where(t['spike_tcourse2'] == 1)[0]
        spikes = spikes[(spikes > 10) & (spikes < image_len - 11)]
        spike_matrix = np.zeros((len(spikes), 21))
        for i in range(len(spikes)):
            spike_matrix[i, :] = t['norm_tcourse2'][spikes[i] - 10:spikes[i] + 11]
        ave = spike_matrix.mean(axis=0)
        std = spike_matrix.std(axis=0)

        ax = plt.subplot(nrow, 5, c + 1)
        plt.plot(np.arange(-10, 11) / 300 * 1000, spike_matrix.T, color=(0.8, 0.8, 0.8))
        plt.plot(np.arange(-10, 11) / 300 * 1000, ave)
        plt.title('cell %d' % ind)

plt.tight_layout(.2)
plt.savefig(imdir + "spike_shape.pdf")


plt.figure(4, figsize=(12, 12 / 5 * nrow))
for c, ind in enumerate(active_cells):
    R = ROI_info[ind]
    W = R['Weight_final']
    X = R['ROI_X']
    Y = R['ROI_Y']
    plt.subplot(nrow, 5, c + 1)
    A = np.zeros(dims).T
    A[(X, Y)] = W
    plt.imshow(A)
    plt.ylim(X.min() - 1, X.max() + 1)
    plt.xlim(Y.min() - 1, Y.max() + 1)
    plt.title('cell %d' % ind)

plt.tight_layout(.2)
plt.savefig(imdir + "neural_shape.pdf")
