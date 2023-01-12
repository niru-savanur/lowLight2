import sys, os
import visdom
import time
import argparse
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import scipy.misc as misc
from torch.utils import data
from tqdm import tqdm
from loader import get_loader, get_data_path
from networks import get_model
import cv2





inputFolder = './input'
outputFolder = '/home/kevin/Desktop/model_test/output' # SeeInDark2'

Files = os.listdir(inputFolder)
numFiles = len(Files)


def test(args, model):
    model.eval()
    with th.no_grad():
        for i in range(numFiles):
            print i
            name = inputFolder + '/' + Files[i]
            print(name)

            input = misc.imread(name)
            input = misc.imresize(input, (512,768))
            input = input.astype(float) / 255.
            input = np.transpose(input, (2,0,1))
            input = np.expand_dims(input, 0)
            input = th.from_numpy(input).float()
            input = input.cuda()
            print input.shape
            xc,xmul,res = model(input)
            r = res.data[0].cpu().numpy()
            r = np.transpose(r, (1,2,0))
            print r.shape
            misc.imsave(outputFolder+'/'+Files[i].replace('png','jpg'), r)

def netParams(model):
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p
    return total_paramters

def mainFunc(args):
    model = get_model(args.arch)
    print(args.arch)
    model.cuda()
    total_paramters = netParams(model)
    print('Total network parameters: ' + str(total_paramters))
    
    if os.path.isfile(args.resume):
        checkpoint = th.load(args.resume)
        model.load_state_dict(checkpoint['model_state'])


    else:
       print("No checkpoint found at '{}'".format(args.resume))
    test(args, model)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s', 
                        help='Architecture to use [\'fcn8s, unet, segnet etc\']')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')
    args = parser.parse_args()
    mainFunc(args)
