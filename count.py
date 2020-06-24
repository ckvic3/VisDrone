from os import listdir
from os.path import join
root = "./training_dataset/VisDrone/crop511/VisDrone2019-SOT-train"
count =0
for path in listdir(root):
    count += len(listdir(join(root,path)))

print(count)