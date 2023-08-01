import numpy as np

import argparse
from .HRNet.lib import models
# import HRNet.lib.models as models
from .HRNet.lib.config import config, update_config
from .HRNet.lib.core.evaluation import decode_preds
import torch
import cv2
class HRNet:
    def __init__(self):
        parser = argparse.ArgumentParser(description='Real-time webcam demo')

        parser.add_argument('--cfg', help='experiment configuration filename', type=str,
                            default='/mnt/hdd/CTCNet/CTCNet/utils/HRNet/experiments/300w/face_alignment_300w_hrnet_w18.yaml')
        parser.add_argument('--model-file', help='model parameters', type=str,
                            default='/mnt/hdd/CTCNet/CTCNet/utils/HRNet/New_HR18-300W.pth')
        #
        args = parser.parse_args()
        update_config(config, args)


        config.defrost()
        config.MODEL.INIT_WEIGHTS = False
        config.freeze()
        self.model = models.get_face_alignment_net(config)

        # load model
        state_dict = torch.load(args.model_file)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        # return args

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.size = 256
        self.center = [self.size / 2, self.size / 2]
        self.scale = self.size / 200

    def detect(self, image):
        h, w, _ = image.shape
        frame = cv2.resize(image, [self.size, self.size])

        input = np.copy(frame).astype(np.float32)
        input = (input / 255.0 - self.mean) / self.std
        input = torch.from_numpy(input).permute(2, 0, 1).unsqueeze(0)

        output = self.model(input)
        score_map = output.data.cpu()

        preds = decode_preds(score_map, torch.tensor([self.center]), torch.tensor([self.scale]), [64, 64])

        return np.array([[x*h/self.size, y*w/self.size] for x,y in preds[0]])
