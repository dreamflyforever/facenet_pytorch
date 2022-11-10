import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

import os
import glob

img_embedding_list = []
resnet = InceptionResnetV1(pretrained='vggface2').eval()
# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size= 360, margin = 0)
WSI_MASK_PATH = 'photo_dataset'#存放图片的文件夹路径
paths = glob.glob(os.path.join(WSI_MASK_PATH, '*.png'))
paths.sort()
tmp = []
for i in paths:
    img2 = Image.open(i)
    img2 = img2.convert('RGB')
    img_cropped2 = mtcnn(img2, save_path="{}.jpg".format(i))

    img_embedding2 = resnet(img_cropped2.unsqueeze(0))
    name = os.path.splitext(i)[0]
    img_embedding_list.append([img_embedding2, name])

print("{}".format(img_embedding_list[0][1]))
# Create an inception resnet (in eval mode):

img = Image.open("photo_dataset/jim.png")
print(img.mode)
img = img.convert('RGB')
# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img, save_path="photo_dataset/360.jpg")

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))
for index in range(len(img_embedding_list)):
    dis = np.linalg.norm(img_embedding.detach().numpy() - img_embedding_list[index][0].detach().numpy())
    print("distense:{} : {} ".format(dis, img_embedding_list[index][1]))