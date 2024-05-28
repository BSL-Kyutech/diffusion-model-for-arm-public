from tqdm import tqdm
import torch
import torch.nn as nn
from diffusion_model import Model, Dataset, steps, gen_xt,  normalize
import os


# 学習をおこない, パラメータをdata/model.pthに保存する
if __name__ == '__main__':
    device = "cuda"
    dataset = Dataset("data/train.csv")
    model = Model(steps).to(device)
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
    torch.save(model.state_dict(), "data/model.pth")
