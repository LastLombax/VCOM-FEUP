import numpy as np
import copy
from matplotlib import pyplot as plt
from utils import readMhd, readCsv, getImgWorldTransfMats, convertToImgCoord, extractCube
from readNoduleList import nodEqDiam
import os
import sys
np.set_printoptions(threshold=sys.maxsize)

dispFlag = True

# define the name of the directory to be created
path1 = "../Dataset/mask_cubes"

# define the access rights
access_rights = 0o755

try:
    os.mkdir(path1, access_rights)
except OSError:
    print ("Creation of the directory %s failed" % path1)
else:
    print ("Successfully created the directory %s" % path1)

# Read nodules csv
csvlines = readCsv('../trainset_csv/trainNodules.csv')
header = csvlines[0]
nodules = csvlines[1:]

cube_size = 80
voxel_size = 63.75

lndloaded = -1
for n in nodules:
        vol = float(n[header.index('Volume')])
        if nodEqDiam(vol)>3: #only get nodule cubes for nodules>3mm
                ctr = np.array([float(n[header.index('x')]), float(n[header.index('y')]), float(n[header.index('z')])])
                lnd = int(n[header.index('LNDbID')])
                rad = int(n[header.index('RadID')])
                finding = int(n[header.index('FindingID')])
                
                print(ctr, lnd,finding,rad)
                        
                # Read scan
                if lnd!=lndloaded:
                        [scan,spacing,origin,transfmat] =  readMhd('../Dataset/LNDb-{:04}.mhd'.format(lnd))
                        transfmat_toimg,transfmat_toworld = getImgWorldTransfMats(spacing,transfmat)
                        lndloaded = lnd
                
                # Convert coordinates to image
                ctr = convertToImgCoord(ctr,origin,transfmat_toimg)                
                
                # Read segmentation mask
                [mask,_,_,_] =  readMhd('../Dataset/masks/LNDb-{:04}_rad{}.mhd'.format(lnd,rad))
                
                # Extract cube around nodule
                scan_cube = extractCube(scan,spacing,ctr)
                masknod = copy.copy(mask)
                masknod[masknod!=finding] = 0 #is radfinding on other file.
                masknod[masknod>0] = 1
                mask_cube = extractCube(masknod,spacing,ctr, cube_size, voxel_size)

                print(mask_cube)
                
                # Display mid slices from resampled scan/mask
                if dispFlag:
                        fig, axs = plt.subplots(2,3)
                        axs[0,0].imshow(scan_cube[int(scan_cube.shape[0]/2),:,:])
                        axs[1,0].imshow(mask_cube[int(mask_cube.shape[0]/2),:,:])

                        axs[0,1].imshow(scan_cube[:,int(scan_cube.shape[1]/2),:])
                        axs[1,1].imshow(mask_cube[:,int(mask_cube.shape[1]/2),:])

                        axs[0,2].imshow(scan_cube[:,:,int(scan_cube.shape[2]/2)])
                        axs[1,2].imshow(mask_cube[:,:,int(mask_cube.shape[2]/2)])    
                        plt.show()
                
                # Save mask cubes
                np.save('../Dataset/mask_cubes/LNDb-{:04d}_finding{}.npy'.format(lnd,finding),mask_cube)                