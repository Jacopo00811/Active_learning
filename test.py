import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torchvision import datasets, transforms, models
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

not_last_model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.eval()
not_last_model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize RGB channels
])

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

for idx, (images, _) in enumerate(dataloader):
    images = images.to(device)
    outputs = model(images)
    softmax_out = torch.nn.functional.softmax(model(images),dim=1)
    penultimate_features = not_last_model(images).squeeze()
    hypothetical_labels = torch.argmax(softmax_out, dim=1)
    one_hot = torch.zeros_like(softmax_out)
    one_hot.scatter_(1, hypothetical_labels.unsqueeze(1), 1)
    penultimate_flat = penultimate_features.view(10, -1)  # [batch_size, feature_dim]
    term_1 = softmax_out - one_hot  # [batch_size, num_classes]
    print(term_1.shape)
    print(penultimate_features.shape)
    break
a = torch.tensor([[1, 2],[3,4]])
b = torch.tensor([[5, 6,7],[8,9,10]])
result = a.unsqueeze(-1)@b.unsqueeze(1)
c = torch.tensor([[ 5,  6,  7, 10, 12, 14],
        [24, 27, 30, 32, 36, 40]])
print(result)
print(result.reshape(2, -1))
aappened =[]
extened = []
aappened.append(result.reshape(2, -1))
extened.extend(result.reshape(2, -1))
aappened.append(c)
extened.extend(c)
print("||||||||||||")
print(aappened)
print("||||||||||||")
print(extened)
print("||||||||||||")

tensorL = torch.tensor([[0,0],[1,1]], dtype=torch.float)
tensorU = torch.tensor([[2,2],[3,3]], dtype=torch.float)
dis = torch.cdist(tensorL, tensorU,p=2)
print(dis)
min_distances = torch.min(dis, dim=0)
print(min_distances)
max_pos = torch.argmax(min_distances.values)
print(max_pos)
print("||||||||||||")
all_indices = []

# indices = torch.cat(all_indices)
# print(indices)
# print(len(dataloader.dataset[0]))

ls = [10,1,2,3,4,5,6]
_,tls = torch.topk(torch.tensor(ls), k=3)
tls = tls.tolist()
print([ls[i] for i in tls])

print("||||||||||||")
print(torch.argmax(torch.tensor([10,11,11,4,5,6,7,8,9,10]), dim=0))


all_indices = []
i =0
print("=================")
for idx, (images, _) in enumerate(dataloader):
    all_indices.append(torch.full((images.size(0),), idx, device=device))
    i+=1
    if i == 2:
        break
indices = torch.cat(all_indices)
print(indices)