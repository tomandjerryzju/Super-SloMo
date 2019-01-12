import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import model
import dataloader
import matplotlib.pyplot as plt
import numpy as np
import random
import statistics as st
from math import log10
from IPython.display import clear_output, display
import datetime
from tensorboardX import SummaryWriter
from PIL import Image
import os
from webp_create import create_webp


# Choose batchsize as per GPU/CPU configuration
# This configuration works on GTX 1080 Ti
TRAIN_BATCH_SIZE = 6
VALIDATION_BATCH_SIZE = 1

# Path to dataset folder containing train-test-validation folders
DATASET_ROOT = "dataset"


# If resuming from checkpoint, set `trainingContinue` to True and set `checkpoint_path`
TRAINING_CONTINUE = True
CHECKPOINT_PATH = 'SuperSloMo.ckpt'


writer = SummaryWriter('log')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flowComp = model.UNet(6, 4)
flowComp.to(device)
ArbTimeFlowIntrp = model.UNet(20, 5)
ArbTimeFlowIntrp.to(device)


trainFlowBackWarp      = model.backWarp(352, 352, device)
trainFlowBackWarp      = trainFlowBackWarp.to(device)
validationFlowBackWarp = model.backWarp(640, 352, device)
validationFlowBackWarp = validationFlowBackWarp.to(device)


dict1 = torch.load(CHECKPOINT_PATH, map_location='cpu')
ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
flowComp.load_state_dict(dict1['state_dictFC'])


# Channel wise mean calculated on adobe240-fps training dataset
mean = [0.429, 0.431, 0.397]
std  = [1, 1, 1]
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([transforms.ToTensor(), normalize])

validationset = dataloader.SuperSloMo(root=DATASET_ROOT + '/test', transform=transform, randomCropSize=(640, 352), train=False)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=VALIDATION_BATCH_SIZE, shuffle=False)


negmean = [x * -1 for x in mean]
revNormalize = transforms.Normalize(mean=negmean, std=std)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])


def validate():
    # For details see training.
    flag = 1
    count = 0
    with torch.no_grad():
        for validationIndex, (validationData, _) in enumerate(validationloader, 0):
            frame0 = validationData[0]
            frame1 = validationData[-1]

            I0 = frame0.to(device)
            I1 = frame1.to(device)
            # IFrame = frameT.to(device)

            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:, :2, :, :]
            F_1_0 = flowOut[:, 2:, :, :]

            image_collector = []
            for t_ in range(0, 7):
                print t_
                t_ = torch.Tensor([t_]).type(torch.int64)

                fCoeff = model.getFlowCoeff(t_, device)

                F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
                F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

                g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)
                g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)

                intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))

                F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
                F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
                V_t_0 = F.sigmoid(intrpOut[:, 4:5, :, :])
                V_t_1 = 1 - V_t_0

                g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)
                g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)

                wCoeff = model.getWarpCoeff(t_, device)

                Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (
                wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)

                image_collector.append(Ft_p)

            save_path = os.path.join('/Users/hyc/workspace/Super-SloMo/intrpOut', str(validationIndex))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            torchvision.utils.save_image((revNormalize(frame0[0])), os.path.join(save_path, str(count).zfill(4) + '.jpg'))
            count += 1
            for jj, image in enumerate(image_collector):
                torchvision.utils.save_image((revNormalize(image.cpu()[0])), os.path.join(save_path, str(count).zfill(4) + '.jpg'))
                count += 1
            torchvision.utils.save_image((revNormalize(frame1[0])), os.path.join(save_path, str(count).zfill(4) + '.jpg'))

            count = 0
            save_path_gt = os.path.join('/Users/hyc/workspace/Super-SloMo/gt', str(validationIndex))
            if not os.path.exists(save_path_gt):
                os.makedirs(save_path_gt)
            torchvision.utils.save_image((revNormalize(frame0[0])),
                                         os.path.join(save_path_gt, str(count).zfill(4) + '.jpg'))
            count += 1
            for jj, image in enumerate(validationData[1:-1]):
                torchvision.utils.save_image((revNormalize(image.cpu()[0])),
                                             os.path.join(save_path_gt, str(count).zfill(4) + '.jpg'))
                count += 1
            torchvision.utils.save_image((revNormalize(frame1[0])),
                                         os.path.join(save_path_gt, str(count).zfill(4) + '.jpg'))

            create_webp(save_path, '{}.gif'.format(validationIndex), min_size=320.0)
            create_webp(save_path_gt, '{}_gt.gif'.format(validationIndex), min_size=320.0)
            break
            # For tensorboard
            # if (flag):
            #     retImg = torchvision.utils.make_grid(
            #         [revNormalize(frame0[0]), revNormalize(frameT[0]), revNormalize(Ft_p.cpu()[0]),
            #          revNormalize(frame1[0])], padding=10)
            #     flag = 0

    # return retImg


# c = validate()
# print(c.size())
# plt.imshow(c.permute(1, 2, 0).numpy())
# plt.show()
if __name__ == "__main__":
    validate()
