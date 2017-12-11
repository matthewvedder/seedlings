import torch
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np

labels = np.array([
    "Black-grass",
    "Charlock",
    "Cleavers",
    "Common Chickweed",
    "Common wheat",
    "Fat Hen",
    "Loose Silky-bent",
    "Maize",
    "Scentless Mayweed",
    "Shepherds Purse",
    "Small-flowered Cranesbill",
    "Sugar beet"
])

resnet = models.resnet152(pretrained=True)
# freeze all model parameters

num_features = resnet.fc.in_features
resnet.fc = torch.nn.Linear(num_features, 12)
for param in resnet.parameters():
    param.requires_grad = False

resnet = resnet.cuda()

class TestImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        identifier = path.split('/')[-1]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, identifier

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)

data_transforms = transforms.Compose([
    transforms.Scale(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

def predict(num_folds, directory):
    test_set = TestImageFolder(directory, data_transforms)
    test_dataloder = torch.utils.data.DataLoader(
        test_set, batch_size=32, shuffle=False, num_workers=4
    )

    index = 0
    # predictions = pd.DataFrame(columns=get_labels())
    csv = pd.DataFrame(columns=['file', 'species'])
    for data in test_dataloder:
        images, ids = data

        if torch.cuda.is_available():
            inputs = Variable(images.cuda())
        else:
            inputs = Variable(inputs)

        outputs = [None] * num_folds
        for i in range (num_folds):
            resnet.load_state_dict(torch.load('./fold_{}.pth'.format(i)))
            output = resnet(inputs).data.cpu().numpy()
            outputs[i] = output
        average_outputs = np.mean(np.array(outputs), axis=0)
        predictions = labels[np.argmax(average_outputs, axis=1)]
        ids_and_outputs = pd.DataFrame(np.c_[np.asarray(ids), predictions], columns=['file', 'species'])

        csv = pd.concat([csv, ids_and_outputs], ignore_index=True)

        if index % 50 == 0:
            print('{} batches complete'.format(index))
        index += 1

    with open('predictions.csv', 'w') as f:
        f.write(csv.to_csv(index=False))

predict(5, './data/test/')
