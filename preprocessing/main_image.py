# main_image version, only for image, include RetinaFace (face detection),  face parsing (face occlusion detection)
import argparse
from retinaface import RetinaFace
from ibug.face_parsing.parser import FaceParser as RTNetPredictor

from RetinaFaceWrapper import detect_face
from skimage.transform import estimate_transform, warp
import cv2, os, math,  copy
import numpy as np
from glob import glob

IMAGE_SIZE = 448


def main(args):
    ckpt = "ibug/face_parsing/rtnet/weights/rtnet50-fcn-14.torch"
    face_detector = RetinaFace()
    face_parser = RTNetPredictor(device="cuda:1", ckpt=ckpt, num_classes=14)
    image_path = args.input
    if args.output == "":
        save_path = "imagePreprocessingResult/"
    else:
        save_path = args.output
    os.makedirs(save_path, exist_ok=True)
    print("starting..")


    for  imagepath in sorted(glob(image_path+'*.jpg')+glob(image_path+'*.png')):

        img = cv2.imread(imagepath)
        name = os.path.splitext(os.path.split(imagepath)[-1])[0]
        if os.path.exists( os.path.join(save_path, name+'.jpg')):
            continue
        imageH, imageW = img.shape[:2]
        regins, detectTag = detect_face(face_detector, img)

        # pass the image size is smaller than 448
        if len(regins) == 0:
            if detectTag:
                print("Detected face is smaller than 448 in ", imagepath)
            else:
                print("No face in ", imagepath)
            continue

        regins = np.array(regins)
        smallRegins = copy.deepcopy(regins)
        smallRegins[:, 0] = (regins[:, 0] * 224 / imageW)
        smallRegins[:, 1] = (regins[:, 1] * 224 / imageH)
        smallRegins[:, 2] = (regins[:, 2] * 224 / imageW)
        smallRegins[:, 3] = (regins[:, 3] * 224 / imageH)

        #  face parsing (face occlusion detection)
        masks = face_parser.predict_img(cv2.resize(img,(224,224)), smallRegins.astype(int), rgb=False)

        # mouth, eyes, nose
        # 0 : background 1 : skin (including face and scalp) 2 : left_eyebrow 3 : right_eyebrow 4 : left_eye 5 : right_eye 6 : nose
        # 7 : upper_lip 8 : inner_mouth 9 : lower_lip 10 : hair 11 : left_ear 12 : right_ear 13 : glasses
        for i in range(len(masks)-1, -1,-1):
            if len(np.where(masks[i]==4)[0])==0 or len(np.where(masks[i]==5)[0])==0 or len(np.where(masks[i]==6)[0])==0 \
                    or len(np.where(masks[i]==7)[0])==0 or len(np.where(masks[i]==9)[0])==0:
                print("Face has occlusion")
                regins = np.delete(regins, i, axis=0)

        if len(regins) == 0:
            print("All faces have occlusion")
            continue


        d = regins[0]
        src_pts = np.array(
            [[d[0], d[1]], [d[0], d[3]],
             [d[2], d[1]]])

        DST_PTS = np.array([[0, 0], [0, IMAGE_SIZE - 1], [IMAGE_SIZE - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        x_image = img / 255.
        dst_image = warp(x_image, tform.inverse, output_shape=(IMAGE_SIZE, IMAGE_SIZE))
        dst_image = dst_image * 255.
        dst_image = np.maximum(np.minimum(dst_image, 255), 0)


        cv2.imwrite( os.path.join(save_path, name+'.jpg'), dst_image)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda', type=str, help='set device, cpu for using cpu')
    parser.add_argument('--input', default="/media/cine/First/LGAI_Dataset/CelebAHQ-Dataset/", type=str, help='image path')
    parser.add_argument('--output', default="/media/cine/First/LGAI_Dataset/CelebAHQ/images/", type=str, help='save image path')

    main(parser.parse_args())
