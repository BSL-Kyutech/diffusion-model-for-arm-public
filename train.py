from tqdm import tqdm
import torch
import torch.nn as nn
from diffusion_model import ControlNet, ModelForXY, ModelForTheta, Dataset, steps, gen_xt,  normalize
import os

# 学習をおこない, パラメータを保存する
def train_xy(model):
    device = "cuda"
    dataset = Dataset("data/train.csv")
    batch_size = 100
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=14,
        drop_last=True,
        pin_memory=True
    )

    epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=0.1, total_iters=epochs)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch, (x, pos, theta) in tqdm(enumerate(dataloader)):
            x, pos, theta = x.to(device), pos.to(device), theta.to(device)
            x = normalize(x)
            t = torch.randint(1, steps, (batch_size,),
                              device=device).long()
            y = torch.randn_like(x).to(device)
            x = gen_xt(x, t, y)
            x, y = x.to(device), y.to(device)
            pred = model(x, t, pos)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'epoch:{epoch} {total_loss/len(dataloader)}')
        scheduler.step()

    if not os.path.exists('data'):
        os.mkdir('data')
    torch.save(model.state_dict(), "data/model_for_xy.pth")

# 学習をおこない, パラメータを保存する
def train_theta(model):
    device = "cuda"
    dataset = Dataset("data/train.csv")
    batch_size = 100
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=14,
        drop_last=True,
        pin_memory=True
    )

    epochs = 20
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=0.1, total_iters=epochs)
    criterion = nn.MSELoss()

    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch, (x, pos, theta) in tqdm(enumerate(dataloader)):
            x, pos, theta = x.to(device), pos.to(device), theta.to(device)
            x = normalize(x)
            t = torch.randint(1, steps, (batch_size,),
                              device=device).long()
            y = torch.randn_like(x).to(device)
            x = gen_xt(x, t, y)
            x, y = x.to(device), y.to(device)
            pred = model(x, t, theta)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'epoch:{epoch} {total_loss/len(dataloader)}')
        scheduler.step()

    if not os.path.exists('data'):
        os.mkdir('data')
    torch.save(model.state_dict(), "data/model_for_theta.pth")

def train_controlnet(finetune=False):
    model = ControlNet(steps).cuda()
    device = "cuda"
    dataset = Dataset("data/train.csv")
    batch_size = 100
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=14,
        drop_last=True,
        pin_memory=True
    )

    epochs = 500
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=0.1, total_iters=epochs)
    criterion = nn.MSELoss()

    if finetune==False:
        print("train xy")
        #pbar
        with tqdm(range(epochs)) as pbar:
            for epoch in pbar:
                total_loss = 0
                for batch, (x, pos, theta) in tqdm(enumerate(dataloader)):
                    x, pos, theta = x.to(device), pos.to(device), theta.to(device)
                    x = normalize(x)
                    t = torch.randint(1, steps, (batch_size,),
                                    device=device).long()
                    y = torch.randn_like(x).to(device)
                    x = gen_xt(x, t, y)
                    x, y = x.to(device), y.to(device)
                    pred = model(x, t, pos)
                    loss = criterion(pred, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    pbar.set_description(f'epoch:{epoch} {loss.item()} ')
                print(f'epoch:{epoch} {total_loss/len(dataloader)}')
                scheduler.step()

        if not os.path.exists('data'):
            os.mkdir('data')
        torch.save(model.state_dict(), "data/controlnet_xy.pth")
    
    else:
        print("skip train xy")
        model.load_state_dict(torch.load("data/controlnet_xy.pth"))

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1, end_factor=0.1, total_iters=epochs)

    print("train theta")
    for epoch in tqdm(range(epochs)):
        total_loss = 0
        for batch, (x, pos, theta) in tqdm(enumerate(dataloader)):
            x, pos, theta = x.to(device), pos.to(device), theta.to(device)
            x = normalize(x)
            t = torch.randint(1, steps, (batch_size,),
                              device=device).long()
            y = torch.randn_like(x).to(device)
            x = gen_xt(x, t, y)
            x, y = x.to(device), y.to(device)
            pred = model(x, t, pos, theta)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'epoch:{epoch} {total_loss/len(dataloader)}')
        scheduler.step()

    if not os.path.exists('data'):
        os.mkdir('data')
    torch.save(model.state_dict(), "data/controlnet_xy_and_theta.pth")



if __name__ == '__main__':
    #model = ModelForXY(steps).cuda()
    #train_xy(model)
    #model = ModelForTheta(steps).cuda()
    #train_theta(mode)
    train_controlnet()
    
#0.74
#0.623