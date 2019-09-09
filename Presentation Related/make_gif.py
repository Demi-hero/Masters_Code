import os
import glob
from PIL import Image, ImageDraw


def make_gif_object(image_folder,image_range, image_spacing):
    frames = []
    for value in range(0,image_range,image_spacing):
        pth = os.path.join(image_folder,"{}.png".format(value))
        frame = Image.open(pth)
        frames.append(frame)
    return frames


if 0:

    # Create the frames
    frames = []
    x, y = 0, 0
    for i in range(10):
        new_frame = create_image_with_ball(400, 400, x, y, 40)
        frames.append(new_frame)
        x += 40
        y += 40

    # Save into a GIF file that loops forever
    frames[0].save('moving_ball.gif', format='GIF', append_images=frames[1:], save_all=True, duration=100, loop=1)

# Style GAN
if 0:
    img_file1 = "Results\*.jpg"
    img_file2 = "Results2\*.jpg"
    img_file3 = "Results3\*.jpg"
    frames = []
    size = (512, 256)

    pics1 = glob.glob(img_file1)
    pics2 = glob.glob(img_file2)
    pics3 = glob.glob(img_file3)

    imgs = [pics1, pics2, pics3]
    for pics in imgs:
        for path in pics:
            #path = os.path.join(img_file,"generated_{}.png".format(value))
            frame = Image.open(path)
            frames.append(frame)
    frames[0].save("Style_merger.gif", format="GIF", append_images=frames[1:], save_all=True, duration=100, loop=1)

if True:
    path =  "D:\Documents\Comp Sci Masters\Project_Data\Masters_Code\GANs\Gan\images"
    frames = make_gif_object(path, 30000, 200)
    frames[0].save("og_gan_merger.gif", format="GIF", append_images=frames[1:], save_all=True, duration=100, loop=1)
