# main version, only for video, include RetinaFace (face detection),  face parsing (face occlusion detection), Sort (object tracking), ArcFace (face verfication)
import argparse
from retinaface import RetinaFace
# from .ibug.face_parsing.parser import FaceParser as RTNetPredictor
from .sort.sort import *
# from arcface import ArcFace
from .RetinaFaceWrapper import detect_face
from skimage.transform import estimate_transform, warp
import cv2, os, math, random, shutil, copy
import numpy as np
from glob import glob
import torch
IMAGE_SIZE = 224
CHANGE_DETECT_RATIO = 30.0
BATCH_SIZE = 32 # if the number of image in the dir is smaller than BATCH_SIZE, we will delete the dir.
def getPSNR(I1, I2):
    max_pixel = 255.0
    mse = np.mean((I1 - I2) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    return psnr

def fileList(pathK):
    filelist = {}
    n = 0
    for root, folders, files in os.walk(pathK):
        for file in files:
            n += 1
            filelist[file] = file
    return filelist

def compare(pathL1, pathL2):
    for k1 in pathL1:
        dict1 = fileList(k1)
        for k2 in pathL2:
            dict2 = fileList(k2)
            for key1 in dict1:
                for key2 in dict2:
                    if dict1[key1] == dict2[key2]:
                        return True
    return False
 

def preprocessing(device="cuda", input="/home/cine/Downloads/acting.mp4", output="/home/cine/Downloads/TestVideo/acting/", isExclude448=False):
    ckpt = "/mnt/hdd/CTCNet/CTCNet/preprocessing/ibug/face_parsing/rtnet/weights/rtnet50-fcn-14.torch"
    face_detector = RetinaFace()
    mot_tracker = Sort()
    video_path = input
    videoName = os.path.splitext(os.path.split(video_path)[-1])[0]
    if output == "":
        save_path = "dataset/video_sequence/%s_sequence/" % videoName
    else:
        save_path = os.path.join(output)
    os.makedirs(save_path, exist_ok=True)
      
    print("starting..")
    cap = cv2.VideoCapture(video_path)

    count = 0; startTag = False; idTag = 1; preSaveCount = [0]; sequenceList = {1: 1}; deletedID = {}
    imageW = cap.get(cv2.CAP_PROP_FRAME_WIDTH);  imageH = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    os.makedirs(os.path.join(save_path, str(1) + "_" + str(sequenceList[1])), exist_ok=True)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            count += 1
            break
        count += 1
        if count == 1:
            prevFrame = frame
            # idDir = {}
        if count % 10 == 1:
            print(f'{count} frame')
        
        # RetinaFace (face detection)
        regins, detectTag = detect_face(face_detector, frame, scale=1.5, isExclude448=isExclude448)

        # pass the image size is smaller than 448
        if len(regins) == 0:
            if detectTag:
                print("Detected face is smaller than 448 in ", count)
            else:
                print("No face in ", count)
            if startTag == False:
                preSaveCount[0] = count
            prevFrame = frame
            continue

        regins = np.array(regins)

        # Sort (object tracking)
        track_bbs_ids = mot_tracker.update(np.array(regins))
        prevFrame = frame
        ids = np.array(track_bbs_ids[:, 4])
        track_bbs_ids = np.array(track_bbs_ids)

        previousID = np.array(track_bbs_ids[:, 4])

        for i, d in enumerate(track_bbs_ids):
            newID = int(track_bbs_ids[i, 4])
            if newID in deletedID:
                continue
            ###########crop image#################
            src_pts = np.array(
                [[d[0], d[1]], [d[0], d[3]],
                 [d[2], d[1]]])

            DST_PTS = np.array([[0, 0], [0, IMAGE_SIZE - 1], [IMAGE_SIZE - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)

            x_image = frame / 255.
            dst_image = warp(x_image, tform.inverse, output_shape=(IMAGE_SIZE, IMAGE_SIZE))
            dst_image = dst_image * 255.
            dst_image = np.maximum(np.minimum(dst_image, 255), 0)
            ##########################################
            if newID > idTag:
                while (len(preSaveCount) < newID):
                    preSaveCount.extend([count])
                    sequenceList[len(preSaveCount)] = max(sequenceList.values()) + 1

                    os.makedirs(os.path.join(save_path, str(len(preSaveCount)) + "_" + str(sequenceList[len(preSaveCount)])),
                                exist_ok=True)
                idTag = newID

            elif count - preSaveCount[newID - 1] > 1:
                if len(glob(os.path.join(save_path, str(newID) + "_" + str(sequenceList[newID])) + '/*.png')) > 0:

                    sequenceList[newID] = max(sequenceList.values()) + 1
                    os.makedirs(os.path.join(save_path, str(newID) + "_" + str(sequenceList[newID])),
                                exist_ok=True)

            preSaveCount[newID - 1] = count
            cv2.imwrite(
                os.path.join(save_path, str(newID) + "_" + str(sequenceList[newID]), f'{count:04d}.png'),
                dst_image)
    
    torch.cuda.empty_cache()