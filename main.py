import argparse, subprocess, os, sys
from preprocessing.class_video import preprocessing
from utils.class_crop_mouth_from_video import *
from utils.face_detector import HRNet
import face_alignment
from utils.utils import *
from eval_singlevideo import *
import numpy as np
import natsort
import cv2
from tqdm import tqdm

def lmk_vis(image, preds):
    for i in range(68):
        cv2.circle(image, (int(preds[i][0]),int(preds[i][1])), 2, (255,0,0), -1)
        # cv2.circle(image, (int(preds[0][i][0]),int(preds[0][i][1])), 2, (255,0,0), -1)
    cv2.imwrite('vis.png', image)

def main(args):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
    face_detector_hrnet = HRNet()
    # filename
    inputpath = args.input
    filename = inputpath.split('/')[-1].split('.')[0]
    filedir = os.path.dirname(inputpath)

    # output dir
    outputpath = os.path.join(args.save, filename)
    os.makedirs(outputpath, exist_ok=True)


    if args.preproc == '':
        print("Start preprocessing")
        # convert input video to 25 fps
        video_25fps_name = f"{filedir}/{filename}_25fps.mp4"
        command = f"rm -f {video_25fps_name}"
        output = subprocess.call(command, shell=True, stdout=None)
        command = f"ffmpeg -y -i {inputpath} -filter:v fps=fps=25 {video_25fps_name}"
        output = subprocess.call(command, shell=True, stdout=None)
        
        # detect faces from video
        preproc_path = os.path.join(outputpath, 'preproc')
        preprocessing(device="cuda:1", input=video_25fps_name, output=preproc_path)
        print("Preprocessing complete. Please check result.")
        print("Please restart this program. (with args.preproc='/path/to/preproc/1_1/)")
        exit()


    # landmark label generation
    preproc_path = args.preproc
    seq = preproc_path.split('/')[-2]
    file_list = os.listdir(preproc_path)
    file_list = natsort.natsorted(file_list)
    lmks_list = []
    print("Landmark detection start.")
    for i in tqdm(range(len(file_list))):
        file = file_list[i]
        file_path = os.path.join(preproc_path, file)
        image = cv2.imread(file_path)
        preds = fa.get_landmarks(np.array(image))
        # lmk_vis(image,lmks)
        lmks_list.append(preds)
    lmks_list = np.array(lmks_list)
    lmk_path = f'{outputpath}/{seq}'
    save2npz(lmk_path, data=lmks_list)

    # # speaker information
    csvfile = open(os.path.join(outputpath, str(filename) + '.csv'), 'w')
    for i in range(args.num):
        csvfile.write(seq + ',0\n')
    csvfile.close()

    # # # frames to video
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    face_video_name = f"{outputpath}/{seq}.mp4"
    out = cv2.VideoWriter(face_video_name, fourcc, 25.0, (224,224))
    file_list = os.listdir(preproc_path)
    file_list = natsort.natsorted(file_list)
    for file in file_list:
        file_path = os.path.join(preproc_path, file)
        image = cv2.imread(file_path)
        out.write(image)
    out.release()
    
    # # extract audio from video file
    audio_name = f"{filedir}/{filename}.wav"
    command = f"rm -f {audio_name}"
    output = subprocess.call(command, shell=True, stdout=None)
    command = f"ffmpeg -i {inputpath} -vn -ar 16000 -ac 1 -ab 192k -f wav {audio_name}"
    output = subprocess.call(command, shell=True, stdout=None)

    # # detect mouth region
    face_video_name = f"{outputpath}/{seq}.mp4"
    lmk_path = f"{outputpath}/{seq}.npz"
    csvfile_path = os.path.join(outputpath, str(filename) + '.csv')
    mouthroi_path = os.path.join(outputpath,"mouthroi")

    crop_mouth(video_direc=face_video_name, landmark_direc=lmk_path, filename_path=csvfile_path, save_direc=mouthroi_path)

    # ctcnet start
    audio_path = f"{filedir}/{filename}.wav"
    mouthroi_path = os.path.join(outputpath,"mouthroi",f'{seq}.npz')
    save_path = os.path.join(outputpath,f'{seq}.wav')
    evaluate(audio_path, mouthroi_path, save_path, args.num, device="cuda:1")
    
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Main function")
    parser.add_argument('-i', '--input', type=str, default='test_videos/interview.mp4', help="Input video")
    parser.add_argument('-p', '--preproc', type=str, default='/mnt/hdd/CTCNet/CTCNet/result/interview/preproc/1_1/', help="preprocessing path") 
    # parser.add_argument('-p', '--preproc', type=str, default='', help="preprocessing path") 
    parser.add_argument('-s', '--save', type=str, default='./result/', help="Output data save path")
    # parser.add_argument('-c', '--conf', type=str, default='', help="configure path")
    parser.add_argument('--num', type=int, default='1', help="number of spearkers in video")
    argument = parser.parse_args()
    main(argument)