# import code; code.interact(local=dict(globals(), **locals()))  # debugger
from __future__ import print_function, division
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
from sklearn.model_selection import StratifiedKFold
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

plt.ion()   # interactive mode



# Data augmentation and normalization for training
# Just normalization for validation
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
data_transforms = transforms.Compose([
    transforms.Scale(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

data_dir = 'data'

image_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms)

# dataset_sizes = { x: len(image_datasets[x]) for x in ['training', 'validation']}
# print(dataset_sizes)
class_names = image_dataset.classes
print(len(class_names))
images, labels = zip(*image_dataset)
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(images, labels)
folds = []
for train_index, validate_index in skf.split(images, labels):
    print("TRAIN:", train_index.shape, "TEST:", validate_index.shape)
    images_train, images_validate = np.array(images)[train_index], np.array(images)[validate_index]
    labels_train, labels_validate = np.array(labels)[train_index], np.array(labels)[validate_index]
    folds.append({
        'training': list(zip(images_train, labels_train)),
        'validation': list(zip(images_validate, labels_validate))
    })

print(len(folds))

use_gpu = torch.cuda.is_available()


def train_model(folds, model, criterion, optimizer, scheduler=None, num_epochs=5):
    since = time.time()
    use_gpu = torch.cuda.is_available()
    best_losses = np.full(len(folds), 1000)
    best_model_states = [None] * len(folds)
    initial_state = copy.deepcopy(model.state_dict())

    for index, fold in enumerate(folds):
        model.load_state_dict(initial_state)
        dataloaders = {
            x: torch.utils.data.DataLoader(
                fold[x], batch_size=128, shuffle=True, num_workers=12
            )
            for x in ['training', 'validation']
        }
        dataset_sizes = {'training': len(dataloaders['training'].dataset),
                         'validation': len(dataloaders['validation'].dataset)}

        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            for phase in ['training', 'validation']:
                if phase == 'training':
                    # scheduler.step()
                    model.train(True)
                else:
                    model.train(False)

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    if use_gpu:
                        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)

                    optimizer.zero_grad()

                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'training':
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.data[0]
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'training':
                    train_fold_loss = running_loss / dataset_sizes[phase]
                    train_fold_acc = running_corrects / dataset_sizes[phase]
                else:
                    valid_fold_loss = running_loss / dataset_sizes[phase]
                    valid_fold_acc = running_corrects / dataset_sizes[phase]

                    if valid_fold_loss < best_losses[index]:
                        best_losses[index] = valid_fold_loss
                        best_model_states[index] = model.state_dict()

            print('Epoch [{}/{}] Fold {} train loss: {:.4f} acc: {:.4f} '
                  'valid loss: {:.4f} acc: {:.4f} time: {:.4f} seconds'.format(
                    epoch, num_epochs - 1, index,
                    train_fold_loss, train_fold_acc,
                    valid_fold_loss, valid_fold_acc,
                    (time.time() - epoch_start_time)))

    for index, state_dict in enumerate(best_model_states):
        torch.save(state_dict, 'fold_{}.pth'.format(index))

    return best_model_states

class TestImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        identifier = path.split('/')[-1].split('.')[0]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, identifier




resnet = models.resnet152(pretrained=True)
# freeze all model parameters
for param in resnet.parameters():
    param.requires_grad = False

num_features = resnet.fc.in_features
print('num_features: ', num_features)

fc_layers = nn.Sequential(
    nn.Linear(num_features, 12),
    nn.Softmax()
)
resnet.fc = torch.nn.Linear(num_features, 12)
print(resnet)
if use_gpu:
    resnet = resnet.cuda()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=0.003)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

start_time = time.time()
# model = torch.load('resnet_model.pth')
models = train_model(folds, resnet, criterion, optimizer, num_epochs=15)


print('Training time: {:10f} minutes'.format((time.time()-start_time)/60))

# visualize_model(dataloaders, resnet)
