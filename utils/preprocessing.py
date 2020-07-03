from tqdm import tqdm
from preprocessing_methods import trim_image, rotate_image
import os
from skimage import io, img_as_ubyte
from skimage.transform import resize
import numpy as np
import argparse

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    """
    Preprocessing Steps:
    Trim the black margins out of the image.

    Subsequently, the augmentation step follows:
    Resize to denoted size 
    Flip images
    Rotate those images whose retinas are complete circles.
    """

    parser = argparse.ArgumentParser(
        description="""Preprocessing""")
    parser.add_argument('imdir', type=str, default=None, metavar='image_dir',
                        help="""The path to the directory which contains the image folder. """)
    parser.add_argument('outdir', type=str, default=None, metavar='out_image_dir',
                        help="""The path to the directory which should contain processed augmented images.""")
    parser.add_argument('-na', '--n_augmentation', type=int, default=0,
                        help="""Number of Augmented images per image""")
    parser.add_argument('-mra', '--max_rotation', type=int, default=0,
                        help="""Max rotation degree +- for the images, e.g. if you pass 10 to this argument then 
                        the function will pick 10 random values from the range -10 to 10""")
    parser.add_argument('-r', '--resize', type=int, default=[256, 256], nargs=2,
                        help="""Enter wished Size. Example use: -r 256 256""")
    parser.add_argument('-ex', '--min_max_width_height_ratio', type=float, nargs=2, default=None,
                        help="""Enter minimial and maximal Ratio. Example use: -ex 0.85 1.15""")
    parser.add_argument('-f', '--flip', type=bool, default=False,
                        help="""Augmentation by flipping images""")
    args = parser.parse_args()


    def add_slash(path):
        if path[-1] != '/':
            return path + "/"
        else:
            return (path)


    dir = add_slash(args.imdir)
    outdir = add_slash(args.outdir)

    os.makedirs(outdir, exist_ok=True)

    print("Start cropping...")
    for f in tqdm(os.listdir(dir)):
        # Crop image
        trim_image(f, dir, outdir)
    print("Finished cropping...")

    print("Start resizing and data augmentation...")
    for f in tqdm(os.listdir(outdir)):
        if f[:-4] == ".jpg":
            fname = f.replace(".jpg", "")
            suffix = ".jpg"
        else:
            fname = f.replace(".jpeg", "")
            suffix = ".jpeg"
        image = io.imread(outdir + f)

        if args.min_max_width_height_ratio:
            if not args.min_max_width_height_ratio[0] < image.shape[0] / image.shape[1] \
                   < args.min_max_width_height_ratio[1]:
                os.system(f'rm {outdir}/{fname}{suffix}')
                continue

        # Resize image
        image = resize(image, output_shape=(args.resize[0], args.resize[1]))

        # save image under processed data
        io.imsave(outdir + f, img_as_ubyte(image))

        if args.n_augmentation > 0:
            #  rotate image and save it
            rotate_image(image, outdir, fname, args.n_augmentation, args.max_rotation, suffix)

            if args.flip:
                # flip image
                image_flipped = np.fliplr(image)
                io.imsave(outdir + fname + f"_flipped{suffix}", img_as_ubyte(image_flipped))

                # rotate image
                rotate_image(image_flipped, outdir, fname + "_flipped", args.n_augmentation, args.max_rotation, suffix)

    print("Finished resizing and data augmentation...")

