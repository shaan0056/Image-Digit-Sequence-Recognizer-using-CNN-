import os,sys
import shutil

root_path = "../Data/processed/"
dest_path = "../Data/Final/"

directory = os.listdir(root_path)

if __name__ == "__main__":
    class_count = {}
    for dir in directory:
            for classes in (os.listdir(root_path+dir)):
                if dir == 'train':
                    class_count[classes] = len(os.listdir(root_path+dir+'/'+classes))

    min_count = min(class_count.items(),key=lambda x: x[1])[1]

    for dir in directory:
            for classes in (os.listdir(root_path+dir)):
                if dir == 'train':
                    for count,files in enumerate(os.listdir(root_path+dir+'/'+classes)):
                        if count < min_count:
                            shutil.copy2(root_path +dir+'/'+classes+'/'+files,
                                         dest_path +dir+'/'+classes)

