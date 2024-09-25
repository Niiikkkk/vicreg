import torch

model = torch.load('/home/nberardo/vicreg/experiment/resnet50.pth', map_location="cpu")
new_state_dict = {}
"""for name,weights in model["state_dict"].items():
    if name.startswith("module.encoder."):
        name = name.replace("module.encoder.", "")
        new_state_dict[name] = weights
torch.save(new_state_dict, 'vicreg_resnet50.pth')"""
for x in model:
    print(x)