from mamba import build_mamba_model
import torch
from torch import optim

model1 = build_mamba_model(mamba_version="./mamba/vmambav2_tiny_224.yaml", mamba_patch=3, ds_scale=[2,2,1], topconv_scale=1, inchans=1)
model2 = build_mamba_model(mamba_version="./mamba/vmambav2_tiny_224.yaml", mamba_patch=2, ds_scale=[2,2,1], topconv_scale=3, inchans=3)

model1 = model1.cuda()
model2 = model2.cuda()

optimizer1 = optim.Adam(
        model1.parameters(),
        lr=1,
        betas=(0.9, 0.99),
        weight_decay=1e-06,
    )

optimizer2 = optim.Adam(
        model2.parameters(),
        lr=1,
        betas=(0.9, 0.99),
        weight_decay=1e-06,
    )

model1.train()
optimizer1.zero_grad()

model2.train()
optimizer2.zero_grad()


while(True):
    torch.cuda.reset_peak_memory_stats()
    input1 = torch.ones(32, 1, 48, 900).cuda()
    input2 = torch.ones(32, 3, 120, 600).cuda()

    with torch.cuda.amp.autocast(enabled=True):
        output1 = model1(input1)
        output2 = model2(input2)
    loss = output1.sum() + output2.sum()
    loss.backward()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    print("-------------------------------------------------------------------------------------------------")
