import numpy as np
import copy
from matplotlib import pyplot as plt
from utils import readMhd, readCsv, getImgWorldTransfMats, convertToImgCoord, extractCube
from readNoduleList import nodEqDiam
from imageio import imwrite
from itertools import *
import os
import cv2

dispFlag = False

imgs = []

# Create Folders

# define the name of the directory to be created
path1 = "../Dataset/images/GGO"
path2 = "../Dataset/images/PartSolid"
path3 = "../Dataset/images/Solid"

# define the access rights
access_rights = 0o755

try:
    os.mkdir(path1, access_rights)
    os.mkdir(path2, access_rights)
    os.mkdir(path3, access_rights)
except OSError:
    print ("Creation of the directory %s failed" % path1)
    print ("Creation of the directory %s failed" % path2)
    print ("Creation of the directory %s failed" % path3)
else:
    print ("Successfully created the directory %s" % path1)
    print ("Successfully created the directory %s" % path2)
    print ("Successfully created the directory %s" % path3)

# Read nodules csv
csvlines = readCsv('../trainset_csv/trainNodules_gt.csv')
header = csvlines[0]
nodules = csvlines[1:]

lndloaded = -1
for n in nodules:
        nod = float(n[header.index('Nodule')])
        if nod > 0: #only get nodule cubes for nodules>0
                ctr = np.array([float(n[header.index('x')]), float(n[header.index('y')]), float(n[header.index('z')])])
                lnd = int(n[header.index('LNDbID')])
                rads = list(map(int,list(n[header.index('RadID')].split(','))))
                radfindings = list(map(int,list(n[header.index('RadFindingID')].split(','))))
                finding = int(n[header.index('FindingID')])
                texts = list(map(float,list(n[header.index('Text')].split(','))))
                #texts = int(n[header.index('Text')])

                print(lnd,finding,rads,radfindings)
                
                # Read scan
                if lnd!=lndloaded:
                        [scan,spacing,origin,transfmat] =  readMhd('../Dataset/LNDb-{:04}.mhd'.format(lnd))
                        transfmat_toimg,transfmat_toworld = getImgWorldTransfMats(spacing,transfmat)
                        lndloaded = lnd
                
                # Convert coordinates to image
                ctr = convertToImgCoord(ctr,origin,transfmat_toimg)                
                
                for rad,radfinding,text in zip(rads,radfindings,texts):
                        # Read segmentation mask
                        # [mask,_,_,_] =  readMhd('masks/LNDb-{:04}_rad{}.mhd'.format(lnd,rad))

                        # Extract cube around nodule
                        scan_cube = extractCube(scan,spacing,ctr)
                        # masknod = copy.copy(mask)
                        # masknod[masknod!=radfinding] = 0
                        # masknod[masknod>0] = 1
                        # mask_cube = extractCube(masknod,spacing,ctr)
                        
                        # Display mid slices from resampled scan/mask
                        if dispFlag:
                                fig, axs = plt.subplots(2,3)
                                axs[0,0].imshow(scan_cube[int(scan_cube.shape[0]/2),:,:])
                                # axs[1,0].imshow(mask_cube[int(mask_cube.shape[0]/2),:,:])
                                axs[0,1].imshow(scan_cube[:,int(scan_cube.shape[1]/2),:])
                                # axs[1,1].imshow(mask_cube[:,int(mask_cube.shape[1]/2),:])
                                axs[0,2].imshow(scan_cube[:,:,int(scan_cube.shape[2]/2)])
                                # axs[1,2].imshow(mask_cube[:,:,int(mask_cube.shape[2]/2)])
                                plt.show()
                        
                        # Save scan cubes
                        if (lnd, finding) not in imgs:
                                imx = np.zeros(scan_cube[int(scan_cube.shape[0]/2),:,:].shape)
                                imy = np.zeros(scan_cube[:,int(scan_cube.shape[1]/2),:].shape)
                                imz = np.zeros(scan_cube[:,:,int(scan_cube.shape[2]/2)].shape)
                                imx = cv2.normalize(scan_cube[int(scan_cube.shape[0]/2),:,:], imx, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                                imy = cv2.normalize(scan_cube[:,int(scan_cube.shape[1]/2),:], imy, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                                imz = cv2.normalize(scan_cube[:,:,int(scan_cube.shape[2]/2)], imz, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                                #imwrite('../Dataset/images/LNDb-{:04d}_finding{}_x.png'.format(lnd, finding), imx)
                                #imwrite('../Dataset/images/LNDb-{:04d}_finding{}_y.png'.format(lnd, finding), imy)
                                #imwrite('../Dataset/images/LNDb-{:04d}_finding{}_z.png'.format(lnd, finding), imz)
                                imgs.append((lnd, finding))

                                img = np.zeros((80, 80, 3))
                                img[:,:,0] = imx
                                img[:,:,1] = imy
                                img[:,:,2] = imz

                                img = cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                                if text < 2.3:
                                        imwrite('../Dataset/images/GGO/LNDb-{:04d}_finding{}.png'.format(lnd, finding), img)
                                elif text >= 2.3 and text <= 3.6:
                                        imwrite('../Dataset/images/PartSolid/LNDb-{:04d}_finding{}.png'.format(lnd, finding), img)
                                elif text > 3.6:
                                        imwrite('../Dataset/images/Solid/LNDb-{:04d}_finding{}.png'.format(lnd, finding), img)

                        # Save mask cubes
                        # np.save('mask_cubes/LNDb-{:04d}_finding{}_rad{}.npy'.format(lnd,finding,rad),mask_cube)                
                
                
