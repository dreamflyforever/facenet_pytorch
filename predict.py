import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

import os
import glob

img_embedding_list = []

# get embedding
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# get face detect 
mtcnn = MTCNN(image_size= 360, margin = 0)

# face database
WSI_MASK_PATH = 'photo_dataset'
paths = glob.glob(os.path.join(WSI_MASK_PATH, '*'))
paths.sort()
tmp = []
for i in paths:
    img2 = Image.open(i)
    img2 = img2.convert('RGB')
    file_name = os.path.basename('{}'.format(i))
    img_cropped2 = mtcnn(img2, save_path="mtcnn_crop/{}".format(file_name))
    img_embedding2 = resnet(img_cropped2.unsqueeze(0))
    name = os.path.splitext(i)[0]
    img_embedding_list.append([img_embedding2, name])

print('=============> database face name <=================')
for index in range(len(img_embedding_list)):
	print("{}".format(img_embedding_list[index][1]))
print('====================================================')

# check the face recognize
img = Image.open("photo_dataset/jim.png")
#print(img.mode)
img = img.convert('RGB')
# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img, save_path="checkface.jpg")

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))
for index in range(len(img_embedding_list)):
    dis = np.linalg.norm(img_embedding.detach().numpy() - img_embedding_list[index][0].detach().numpy())
    print("{} : distense:{} ".format(img_embedding_list[index][1], dis))
