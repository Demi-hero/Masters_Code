# imports
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from _utils_CNN import *

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

###################################################################################################

colour_channels = 3  # int(sys.argv[1])
script_name = sys.argv[0]
CWD = "../Data/"
images_root = "__IMAGE__"
csv_root    = "__CSV__"

###################################################################################################
trial_name = 'Conv128_GZ1_Validation_BW-64x'

if colour_channels == 3: trial_name = 'Conv128_GZ1_Validation_RGB-64x'

image_tuple   = (64, 64, colour_channels)
images_folder = "Blended_Image_Catalouge"
images_csv = "GZ1_Full_Expert.csv"
###################################################################################################
# ==================================================================================================
print_header(script_name)
# ==================================================================================================
output_folder = directory_check(CWD, 'CNN_' + trial_name, preserve=False)

# create the data generators
generator = ImageDataGenerator(width_shift_range=0.3, height_shift_range=0.3,
                               horizontal_flip=True, vertical_flip=True, rotation_range=180)


# Images and Labels loading:
images_folder = os.path.join(CWD, images_root, images_folder)
images_csv    = os.path.join(CWD, csv_root, images_csv)
images_labels = pd.read_csv(images_csv)

images_IDs    = np.array(images_labels['OBJID'], dtype=str)

merger_subset = (images_labels[images_labels.Label == "M"])
merger_subset = list(merger_subset.OBJID)

images_array  = read_images_tensor(images_IDs, images_folder, image_tuple, file_extension=".tiff")



for i in range(len(merger_subset)):
    id_index = np.where(images_array == merger_subset[i])
    id_index = id_index[0][0]
    it = generator.flow(images_array[id_index], batch_size=1)
    for i in range(9):
        batch = it.next()
        images_array = np.append(images_array, batch[0])
        images_IDs = np.append(images_IDs, [merger_subset[i]])
        plt.subplot(330 + 1 + i)
        image = batch[0].astype('uint8')
        plt.imshow(image)
    plt.show()