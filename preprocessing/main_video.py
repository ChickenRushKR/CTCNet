# main version, only for video, include RetinaFace (face detection),  face parsing (face occlusion detection), Sort (object tracking), ArcFace (face verfication)
import argparse
from retinaface import RetinaFace
from .ibug.face_parsing.parser import FaceParser as RTNetPredictor
from .sort.sort import *
from arcface import ArcFace
from .RetinaFaceWrapper import detect_face
from skimage.transform import estimate_transform, warp
import cv2, os, math, random, shutil, copy
import numpy as np
from glob import glob
IMAGE_SIZE = 448
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
 
def main(args):
    ckpt = "ibug/face_parsing/rtnet/weights/rtnet50-fcn-14.torch"
    face_detector = RetinaFace()
    face_parser = RTNetPredictor(device=args.device, ckpt=ckpt, num_classes=14)
    mot_tracker = Sort()
  
    video_path = args.input
    videoName = os.path.splitext(os.path.split(video_path)[-1])[0]
    if args.output == "":
        save_path = "dataset/video_sequence/%s_sequence/" % videoName
    else:
        save_path = os.path.join(args.output, "%s_sequence/" % videoName)
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
        
        # RetinaFace (face detection)
        regins, detectTag = detect_face(face_detector, frame, isExclude448=args.isExclude448)

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
        smallRegins = copy.deepcopy(regins)
        smallRegins[:, 0] = (regins[:, 0] * 224 / imageW)
        smallRegins[:, 1] = (regins[:, 1] * 224 / imageH)
        smallRegins[:, 2] = (regins[:, 2] * 224 / imageW)
        smallRegins[:, 3] = (regins[:, 3] * 224 / imageH)

        #  face parsing (face occlusion detection)
        masks = face_parser.predict_img(cv2.resize(frame,(224,224)), smallRegins.astype(int), rgb=False)

        # mouth, eyes, nose
        # 0 : background 1 : skin (including face and scalp) 2 : left_eyebrow 3 : right_eyebrow 4 : left_eye 5 : right_eye 6 : nose
        # 7 : upper_lip 8 : inner_mouth 9 : lower_lip 10 : hair 11 : left_ear 12 : right_ear 13 : glasses
        for i in range(len(masks)-1, -1,-1):
            if len(np.where(masks[i]==4)[0])==0 or len(np.where(masks[i]==5)[0])==0 or len(np.where(masks[i]==6)[0])==0 \
                    or len(np.where(masks[i]==7)[0])==0 or len(np.where(masks[i]==9)[0])==0:
                # print("Face has occlusion")
                regins = np.delete(regins, i, axis=0)

        if len(regins) == 0:
            print("All faces have occlusion")
            if startTag == False:
                preSaveCount[0] = count
            prevFrame = frame
            continue
        else:
            if not startTag:
                startFrame = count
            startTag = True
            psnrV = getPSNR(prevFrame, frame)

            # Sort (object tracking)
            track_bbs_ids = mot_tracker.update(np.array(regins))
            prevFrame = frame
        ids = np.array(track_bbs_ids[:, 4])
        track_bbs_ids = np.array(track_bbs_ids)
 
        if psnrV < CHANGE_DETECT_RATIO and startFrame!=count:
            preDeletedID = deletedID
            deletedID = {}
            for i, nowID in enumerate(ids):
                if nowID in previousID and int(nowID) not in preDeletedID:
                    deletedID[int(nowID)] = int(nowID)
                    continue

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

    cap.release()
    for s in sorted(glob(os.path.join(save_path, "*"))):
        numb = len(glob(os.path.join(s, '*.png')))
        if numb < BATCH_SIZE:  # if the number of image in the dir is smaller than BATCH_SIZE, we will delete it.
            print(s)
            shutil.rmtree(s)

  
    arcFaceModel = ArcFace.ArcFace(model_path=args.arcFaceModel)
    for k in range(2, idTag + 1):

        path = glob(os.path.join(save_path, str(k) + "_*")) #
        if len(path) == 0:
            continue
        path.sort(key=lambda x: int(x.split("/")[-1].split('_')[-1]))
        imageListT = []
        for p in path:
            imageListT.extend(glob(p + '/*.png'))
        imageListT = sorted(imageListT)
        changeId = False
        for m in range(1, k):
            pathPre = glob(os.path.join(save_path, str(m) + "_*"))
            if len(pathPre) <= 0:
                continue

            pathPre.sort(key=lambda x: int(x.split("/")[-1].split('_')[-1]))
            if compare(path, pathPre):
                continue
            imageList = []
            for p in pathPre:
                imageList.extend(glob(p + '/*.png'))
            preId = m

            imageList = sorted(imageList)

            if len(imageList) >= 3:
                a1 = sorted(random.sample(range(len(imageList)), 3))
            else:
                a1 = range(len(imageList))
            if len(imageListT) >= 3:
                a2 = sorted(random.sample(range(len(imageListT)), 3))
            else:
                a2 = range(len(imageListT))
            trueNumber = 0
            # ArcFace (face verfication)
            for i in a1:
                for j in a2:
                    print(os.path.split(imageList[i])[-1], " vs ", os.path.split(imageListT[j])[-1])

                    embs = arcFaceModel.calc_emb([imageList[i], imageListT[j]])
                    dist = arcFaceModel.get_distance_embeddings(embs[0], embs[1])

                    if dist < 1.24:
                        print("True")
                        trueNumber += 1
            if len(a1)==0 or len(a2)==0:
                print(imageList, path)
                print(imageListT, pathPre)
                continue
            if trueNumber / (len(a1) * len(a2)) > 0.8:
                changeId = True
                print("change ID")

                for p in path:
                    tpath = os.path.join(save_path, str(preId) + "_" + p.split('_')[-1])
                    if os.path.exists(tpath):
                        imageAll = glob(p + "/*.png")
                        for img in imageAll:
                            print(os.path.join(tpath, os.path.split(img)[-1]))
                            cv2.imwrite(os.path.join(tpath, os.path.split(img)[-1]), cv2.imread(img))

                        shutil.rmtree(p)
                    else:
                        print("renmae folder ", p, " to ", tpath)
                        os.rename(p, tpath)
            if changeId:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='set device, cpu for using cpu')
    parser.add_argument('--input', default="/home/cine/Downloads/acting.mp4", type=str, help='video path')
    parser.add_argument('--output', default="/home/cine/Downloads/TestVideo/acting/", type=str, help='save sequence path')
    parser.add_argument('--arcFaceModel', default="model.tflite", type=str, help='arcFace pretrained model')
    parser.add_argument('--isExclude448', default=False, type=str, help='exclude the image size which is smaller than 448')

    main(parser.parse_args())
