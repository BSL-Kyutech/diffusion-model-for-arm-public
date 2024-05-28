import pygame
import numpy as np
import copy


# シミュレーションで使うアームのパーツの実装

class Part():
    def __init__(self, pre, base):
        self.pre = pre
        self.base = base
        self.x = [[0.0, 0.0], 0.0]
        self.w = 0.0

    def init(self):
        self.x[1] = 0.0
        if self is not self.base.last:
            self.next.init()

    # 計算
    def calc(self, vals):
        vals = copy.copy(vals)
        self.x = self.calced_x(vals)
        if self is not self.base.last:
            self.next.calc(vals)

    # 描画
    def draw(self, display, color=(0, 0, 0)):
        self.draw_part(display, color)
        if self is not self.base.last:
            self.next.draw(display, color)

    # パーツの先に関節を生やす
    def add_spring_joint(self, c):
        self.base.spring_joint_count += 1
        self.next = SpringJoint(self, self.base, c)
        self.base.last = self.next
        return self.next

    # パーツの先に腕を生やす
    def add_bone(self, length):
        self.next = Bone(self, self.base, length)
        self.base.last = self.next
        return self.next


class Base(Part):
    def __init__(self, x_, y_):
        super().__init__(self, self)
        self.spring_joint_count = 0
        self.x = [[x_, y_], 0]
        self.last = self

    def calced_x(self, vals):
        return self.x

    def draw_part(self, display, color):
        pass


class SpringJoint(Part):
    # 半径
    r = 10

    def __init__(self, pre, base, c):
        super().__init__(pre, base)
        self.c = c

    def calced_x(self, vals):
        theta = self.calced_theta(vals[0], vals[1])
        vals.pop(0), vals.pop(0)
        return [self.pre.x[0], theta]

    def draw_part(self, display, color):
        pygame.draw.circle(
            display, color, (self.x[0][0], self.x[0][1]), self.r, 5)

    def calced_theta(self, u1, u2):
        return (-u1+u2)/(2*self.r)+self.pre.x[1]


class Bone(Part):
    def __init__(self, pre, base, length):
        self.length = length
        super().__init__(pre, base)

    def calced_x(self, vals):
        x = [0.0, 0.0]
        theta = self.pre.x[1]
        x[0] = self.pre.x[0][0]-self.length*np.sin(theta)
        x[1] = self.pre.x[0][1]-self.length*np.cos(theta)
        return [x, theta]

    def draw_part(self, display, color):
        pygame.draw.line(
            display, color, (self.pre.x[0][0], self.pre.x[0][1]), (self.x[0][0], self.x[0][1]), 5)
