from shutil import copy2
import os
import glob

def copy_dataset(source, destination_folder):
    count = 1
    for src_file_png in glob.iglob(source + '/**/*.png', recursive=True):
        #folder_path =  src_file_png.split('/')[0:-1]
        file_name_png = src_file_png.split('/')[-1]
        folder_path = src_file_png[0:-len(file_name_png)]

        file_name_json = file_name_png[0:-4] + '.json'
        #print(file_name_png)
        #print(file_name_json)
        #print(folder_path)
        
        file_json = os.path.join(folder_path, file_name_json)
        file_png = os.path.join(folder_path, file_name_png)
        
        if os.path.exists(file_json) == False:
            print('!')
            print(file_json)
        else:
            print(file_png)
            print(file_json)
            copy2(file_png, os.path.join(destination_folder, str(count)+'.png' ))
            copy2(file_json, os.path.join(destination_folder, str(count)+'.json' ))
            count += 1

    
if __name__ == '__main__':
    src = '../crossval/run1'
    dst = '../dataset_for_comb'
    copy_dataset(src, dst)
