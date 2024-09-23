import pygame
from pygame.locals import *
import numpy as np
import torch
import copy
import time

from simulator import definition as armdef
from diffusion_model import ModelForXY, ModelForTheta, steps, extract, normalize, denormalize


# pathの点を順番に移動する
# while_sleep_timeは点を移動する間sleepする時間
# 滑らかに移動をさせるために, 1ステップ前の入力信号に数ステップ分ノイズを加え, 数ステップ分デノイズしている
def move(path, while_sleep_time=0):
    pygame.init()
    display = pygame.display.set_mode((armdef.width, armdef.height))
    model = ModelForXY(steps)
    model.load_state_dict(torch.load("data/model_for_xy.pth"))
    model = model.cuda()
    xt = torch.randn(armdef.arm.spring_joint_count*2).cuda()
    first = True
    steps_ = steps
    for (x, y) in path:
        model.eval()
        Y = y
        X = x
        pos = torch.FloatTensor([[X, Y]]).cuda()
        arm_ = copy.deepcopy(armdef.arm)
        if not first:
            xt = normalize(xt)
            noise = torch.randn_like(xt).cuda()
            t = torch.Tensor([steps_-1]).long().cuda()
            at_ = extract(t, xt.shape)
            xt = torch.sqrt(at_)*xt+torch.sqrt(1-at_)*noise

        xt = model.denoise(xt, steps_, pos)

        xt = denormalize(xt)
        armdef.arm.calc(xt.tolist())
        display.fill((255, 255, 255))
        for (x, y) in path:
            display.set_at((int(x), int(y)), (0, 0, 0))
        armdef.arm.draw(display)
        pygame.display.update()
        time.sleep(while_sleep_time)

        if first:
            steps_ = 4
            first = False

    pygame.quit()

# 円の軌道を描かせる
def draw_circle():
    l = []
    y0 = armdef.height/2-100
    x0 = armdef.width/2-100
    r = 150
    for i in range(360*3):
        rad = np.deg2rad(i)
        l += [(x0+r*np.cos(rad), y0+r*np.sin(rad))]
    move(l,  0.001)

def xy_and_theta(x, y, theta):
    pygame.init()
    display = pygame.display.set_mode((armdef.width, armdef.height))

    model_xy = ModelForXY(steps)
    model_xy.load_state_dict(torch.load("data/model_for_xy.pth"))
    model_xy = model_xy.cuda()
    model_xy.eval()

    model_theta = ModelForTheta(steps)
    model_theta.load_state_dict(torch.load("data/model_for_theta.pth"))
    model_theta = model_theta.cuda()
    model_theta.eval()

    xt = torch.randn(armdef.arm.spring_joint_count*2).cuda()

    Y = y
    X = x
    pos = torch.FloatTensor([[X, Y]]).cuda()
    theta = torch.FloatTensor([[theta]]).cuda()

    for i in reversed(range(1, steps)):
        xt=model_xy.denoise_once(xt, i, pos)
        xt=model_theta.denoise_once(xt, i, theta).cuda()

    arm_ = copy.deepcopy(armdef.arm)

    xt = denormalize(xt)
    armdef.arm.calc(xt.tolist())
    display.fill((255, 255, 255))
    armdef.arm.draw(display)
    pygame.display.update()
    pygame.time.wait(1000)

    pygame.quit()

if __name__ == '__main__':
    for i in range(-30,30):
        xy_and_theta(100, 100, -i/20)
    #l = []
    #for i in range(360):
    #    rad = np.deg2rad(i)
    #    l += [(armdef.width/2+100*np.cos(rad), armdef.height/2+100*np.sin(rad))]
    #move(l, 0.001)