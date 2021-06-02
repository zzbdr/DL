import torch
a = torch.as_tensor([-1, 2])
b = torch.as_tensor([2, 1])
mask = torch.as_tensor(a > 0)
print(torch.masked_select(a, mask))
print(torch.mean(a.float()))


