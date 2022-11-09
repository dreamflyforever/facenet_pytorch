import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# If required, create a face detection pipeline using MTCNN:
mtcnn = MTCNN(image_size= 360, margin = 0)

# Create an inception resnet (in eval mode):
resnet = InceptionResnetV1(pretrained='vggface2').eval()
from PIL import Image

img = Image.open("photo_dataset/jim.png")
img2 = Image.open("data/test_images/angelina_jolie/1.jpg")
print(img.mode)
img = img.convert('RGB')

# Get cropped and prewhitened image tensor
img_cropped = mtcnn(img, save_path="photo_dataset/360.jpg")
img_cropped2 = mtcnn(img2, save_path="photo_dataset/test.jpg")

# Calculate embedding (unsqueeze to add batch dimension)
img_embedding = resnet(img_cropped.unsqueeze(0))
img_embedding2 = resnet(img_cropped2.unsqueeze(0))
print(img_embedding)
print(img_embedding.shape)
dis = np.linalg.norm(img_embedding.detach().numpy() - img_embedding2.detach().numpy())
print("distense: ", dis)