###
# Author: Kai Li
# Date: 2022-04-03 08:50:42
# Email: lk21@mails.tsinghua.edu.cn
# LastEditTime: 2022-04-03 18:02:56
###
###
# Author: Kai Li
# Date: 2021-06-21 23:29:31
# LastEditors: Please set LastEditors
# LastEditTime: 2021-11-07 23:17:39
###

from typing import OrderedDict
from nichang.videomodels import VideoModel
from nichang.models.ctcnet import CTCNet
from nichang.datas.transform import get_preprocessing_pipelines
import os
import soundfile as sf
import torch
import yaml
import argparse
import numpy as np
from torch.utils import data
import warnings
warnings.filterwarnings("ignore")

def evaluate(audio_path, mouth_path, save_path, num_of_speakers, device):
    with open("local/lrs2_conf.yml") as f:
        conf = yaml.safe_load(f)
    conf["exp_dir"] = os.path.join(
        "exp", conf["log"]["exp_name"])
    conf["audionet"].update({"n_src": 1})

    model_path = os.path.join(conf["exp_dir"], "checkpoints/last.ckpt")
    model_path = "local/vox2_best_model.pt"
    sample_rate = conf["data"]["sample_rate"]
    audiomodel = CTCNet(sample_rate=sample_rate, **conf["audionet"], device=device)
    ckpt = torch.load(model_path, map_location="cpu")['state_dict']
    audiomodel.load_state_dict(ckpt)
    videomodel = VideoModel(**conf["videonet"])

    # Handle device placement
    audiomodel.eval()
    videomodel.eval()
    audiomodel.cuda()
    videomodel.cuda()
    model_device = next(audiomodel.parameters()).device

    # Randomly choose the indexes of sentences to save.
    torch.no_grad().__enter__()
    # for idx in range(1, num_of_speakers+1):
    spk, sr = sf.read(audio_path, dtype="float32")
    mouth = get_preprocessing_pipelines()["val"](np.load(mouth_path)["data"])
    
    # Forward the network on the mixture.
    target_mouths = torch.from_numpy(mouth).to(model_device)
    mix = torch.from_numpy(spk).to(model_device)
    # import pdb; pdb.set_trace()
    mouth_emb = videomodel(target_mouths.unsqueeze(0).unsqueeze(1).float())
    est_sources = audiomodel(mix[None, None], mouth_emb)

    # gt_dir = save_path
    # import pdb; pdb.set_trace()
    sf.write(save_path, est_sources.squeeze(0).squeeze(0).cpu().numpy(), 16000)
        # import pdb; pdb.set_trace()s


# if __name__ == "__main__":
#     from nichang.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

#     args = parser.parse_args()

#     with open("local/lrs2_conf.yml") as f:
#         def_conf = yaml.safe_load(f)

#     arg_dic = parse_args_as_dict(parser)
#     def_conf.update(arg_dic['main_args'])
#     main(def_conf)
