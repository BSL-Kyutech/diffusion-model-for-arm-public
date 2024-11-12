import random


# ランダムな入力を生成
def gen_random_inputs(arm):
    res = []
    for i in range(arm.spring_joint_count):
        res.append(random.uniform(0, 40))
        res.append(random.uniform(0, 40))
    return res


# x, y: 目標座標
# c: 山登り法の試行回数
# 最もx,yに近かった時の入力、x座標、y座標、角度を返す
def yamanobori(arm, x, y, c):
    arm.init()
    min_inputs = []
    min_d = 10000000
    min_x = 0
    min_y = 0
    min_theta = 0
    for i in range(c):
        inputs = gen_random_inputs(arm)
        arm.calc(inputs.copy())
        x_ = arm.last.x[0][0]
        y_ = arm.last.x[0][1]
        d = (x-x_)**2+(y-y_)**2
        if min_d > d:
            min_d = d
            min_inputs = inputs
            min_x = x_
            min_y = y_
            min_theta = arm.last.x[1]
    return min_inputs, min_x, min_y, min_theta
