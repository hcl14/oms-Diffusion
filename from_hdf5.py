import h5py
import numpy as np
from PIL import Image
import io

save_path = 'dresscode.hdf5'
data = []  # list all images files full path 'group/subgroup/b.png' for e.g. ./A/a/b.png. These are basically keys to access our image data.
group = [] # list all groups and subgroups in hdf5 file

def func(name, obj):
    if isinstance(obj, h5py.Dataset):
        data.append(name)
    elif isinstance(obj, h5py.Group):
        group.append(name)

hf = h5py.File(save_path, 'r')
hf.visititems(func)

# Now lets read the image files in their proper format to use it for our training.

for i in group:
    print('group:', i)
    for j in hf[i]:
        print('subgroup:', j)
        if i == 'annots':
            kk = np.array(hf[i][j])
            lines = io.BytesIO(kk).read().decode('UTF-8').split('\n')
            for line in lines:
                print(line)
            continue
        for k in hf[i][j]:
            print('image:', k)
            kk = np.array(hf[i][j][k])

            print(lines)
            img = Image.open(io.BytesIO(kk)) # our image file
            print('image size:', img.size)
            break
        break
    break
