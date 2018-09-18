"""Core functions."""

import os
import nibabel as nb
import numpy as np
from enum import Enum
from matplotlib.cm import get_cmap
from imageio import mimwrite
from skimage.transform import resize

class Orient(Enum):
    SAGITTAL = 0
    CORONAL = 1
    TRANSVERSE = 2


def parse_filename(filepath):
    """Parse input file path into directory, basename and extension.

    Parameters
    ----------
    filepath: string
        Input name that will be parsed into directory, basename and extension.

    Returns
    -------
    dirname: str
        File directory.[
    basename: str
        File name without directory and extension.
    ext: str
        File extension.

    """
    path = os.path.normpath(filepath)
    dirname = os.path.dirname(path)
    filename = path.split(os.sep)[-1]
    basename, ext = filename.split(os.extsep, 1)
    return dirname, basename, ext


def load_and_prepare_image(filename, size=1, slice_orient=None):
    """Load and prepare image data.

    Parameters
    ----------
    filename1: str
        Input file (eg. /john/home/image.nii.gz)
    size: float
        Image resizing factor.
    slice_orient: str
        Output slice orientation (must be valid value of Orient enum)

    Returns
    -------
    out_img: numpy array

    """
    # Load image file
    data = nb.load(filename).get_data()

    if slice_orient and slice_orient not in Orient.__members__:
        raise ValueError('Slice orientation must be one of: {}'.format(', '.join(Orient.__members__)))
    if data.ndim == 4:
        # if data.shape[3] != 1:
        if not slice_orient:
            raise ValueError('Slice orientation must be specified when creating gifs of 4D images')

        orientation = Orient[slice_orient.upper()].value
        center_slice = data.shape[orientation] // 2
        out_img = np.take(data, center_slice, axis=orientation)
        maximum = np.max(out_img.shape)
        resize_shape = [ dim * size for dim in out_img.shape[:2] ] + [ out_img.shape[2] ] # don't resize number of frames
        # handle data formats that store 3D images as a single frame 4D image
        # else:
        #     data = np.reshape(data, data.shape[:3]) # remove 4th dimension

    if data.ndim == 3:
        maximum = np.max(data.shape)
        resize_shape = [ int(maximum * size) ] * 3

        # Pad data array with zeros to make the shape isometric
        out_img = np.zeros([maximum] * 3)

        a, b, c = data.shape
        x, y, z = (list(data.shape) - maximum) / -2

        out_img[int(x):a + int(x),
                int(y):b + int(y),
                int(z):c + int(z)] = data


    out_img = np.array(out_img) / out_img.max()  # scale image values between 0-1

    # Resize image by the following factor
    if size != 1:
        out_img = resize(out_img, resize_shape)

    maximum = int(maximum * size)

    out_img = (255 * out_img).astype(np.uint8)
    return out_img, maximum


def get_orient_slice(out_img, maximum, i, orient=Orient.TRANSVERSE):
    if orient == Orient.SAGITTAL:
        return np.flip(out_img[i, :, :], 1).T
    elif orient == Orient.CORONAL:
        return np.flip(out_img[:, maximum - i - 1, :], 1).T
    elif orient == Orient.TRANSVERSE:
        return np.flip(out_img[:, :, maximum - i - 1], 1).T


def create_mosaic_normal(out_img, maximum, slice_orient):
    """Create grayscale image.

    Parameters
    ----------
    out_img: numpy array
    maximum: int
    slice_orient: string

    Returns
    -------
    new_img: numpy array

    """
    if not slice_orient:
        new_img = \
            [np.hstack((
                np.hstack((
                    get_orient_slice(out_img, maximum, i, Orient.SAGITTAL),
                    get_orient_slice(out_img, maximum, i, Orient.CORONAL))),
                   get_orient_slice(out_img, maximum, i, Orient.TRANSVERSE)))
             for i in range(maximum)]
    else:
        orientation = Orient[slice_orient.upper()]
        orient_max = out_img.shape[orientation.value]

        # if image is isometric, we'll assume it is not a time series and get all slices for the specified orientation
        # otherwise, we will get one slice per timepoint --
        #     the image has already been filtered to the central slice of the specified orientation, so we'll treat
        #     it as a transverse image and simply iterate over the time points
        new_img = [ get_orient_slice(out_img, maximum, i, orientation) for i in range(maximum) ] if all(item == maximum for item in out_img.shape) \
            else [ get_orient_slice(out_img, orient_max, i) for i in range(out_img.shape[-1]) ]

    return np.array(new_img)


def create_mosaic_depth(out_img, maximum):
    """Create an image with concurrent slices represented with colors.

    The image shows you in color what the value of the next slice will be. If
    the color is slightly red or blue it means that the value on the next slide
    is brighter or darker, respectifely. It therefore encodes a certain kind of
    depth into the gif.

    Parameters
    ----------
    out_img: numpy array
    maximum: int

    Returns
    -------
    new_img: numpy array

    """
    # Load normal mosaic image
    new_img = create_mosaic_normal(out_img, maximum)

    # Create RGB image (where red and blue mean a positive or negative shift in
    # the direction of the depicted axis)
    rgb_img = [new_img[i:i + 3, ...] for i in range(maximum - 3)]

    # Make sure to have correct data shape
    out_img = np.rollaxis(np.array(rgb_img), 1, 4)

    # Add the 3 lost images at the end
    out_img = np.vstack(
        (out_img, np.zeros([3] + [o for o in out_img[-1].shape])))

    return out_img


def create_mosaic_RGB(out_img1, out_img2, out_img3, maximum):
    """Create RGB image.

    Parameters
    ----------
    out_img: numpy array
    maximum: int

    Returns
    -------
    new_img: numpy array

    """
    # Load normal mosaic image
    new_img1 = create_mosaic_normal(out_img1, maximum)
    new_img2 = create_mosaic_normal(out_img2, maximum)
    new_img3 = create_mosaic_normal(out_img3, maximum)

    # Create RGB image (where red and blue mean a positive or negative shift
    # in the direction of the depicted axis)
    rgb_img = [[new_img1[i, ...], new_img2[i, ...], new_img3[i, ...]]
               for i in range(maximum)]

    # Make sure to have correct data shape
    out_img = np.rollaxis(np.array(rgb_img), 1, 4)

    # Add the 3 lost images at the end
    out_img = np.vstack(
        (out_img, np.zeros([3] + [o for o in out_img[-1].shape])))

    return out_img



def write_gif_normal(filename, size=1, fps=18, slice_orient=None):
    """Procedure for writing grayscale image.

    Parameters
    ----------
    filename: str
        Input file (eg. /john/home/image.nii.gz)
    size: float
        Between 0 and 1.
    fps: int
        Frames per second

    """
    # Load NIfTI and put it in right shape
    out_img, maximum = load_and_prepare_image(filename, size, slice_orient)


    # Create output mosaic
    new_img = create_mosaic_normal(out_img, maximum, slice_orient)

    # Figure out extension
    ext = '.{}'.format(parse_filename(filename)[2])

    # Write gif file
    mimwrite(filename.replace(ext, '.gif'), new_img,
             format='gif', fps=int(fps * size))


def write_gif_depth(filename, size=1, fps=18):
    """Procedure for writing depth image.

    The image shows you in color what the value of the next slice will be. If
    the color is slightly red or blue it means that the value on the next slide
    is brighter or darker, respectifely. It therefore encodes a certain kind of
    depth into the gif.

    Parameters
    ----------
    filename: str
        Input file (eg. /john/home/image.nii.gz)
    size: float
        Between 0 and 1.
    fps: int
        Frames per second

    """
    # Load NIfTI and put it in right shape
    out_img, maximum = load_and_prepare_image(filename, size)

    # Create output mosaic
    new_img = create_mosaic_depth(out_img, maximum)

    # Figure out extension
    ext = '.{}'.format(parse_filename(filename)[2])

    # Write gif file
    mimwrite(filename.replace(ext, '_depth.gif'), new_img,
             format='gif', fps=int(fps * size))


def write_gif_rgb(filename1, filename2, filename3, size=1, fps=18):
    """Procedure for writing RGB image.

    Parameters
    ----------
    filename1: str
        Input file for red channel.
    filename2: str
        Input file for green channel.
    filename3: str
        Input file for blue channel.
    size: float
        Between 0 and 1.
    fps: int
        Frames per second

    """
    # Load NIfTI and put it in right shape
    out_img1, maximum1 = load_and_prepare_image(filename1, size)
    out_img2, maximum2 = load_and_prepare_image(filename2, size)
    out_img3, maximum3 = load_and_prepare_image(filename3, size)

    if maximum1 == maximum2 and maximum1 == maximum3:
        maximum = maximum1

    # Create output mosaic
    new_img = create_mosaic_RGB(out_img1, out_img2, out_img3, maximum)

    # Generate output path
    out_filename = '{}_{}_{}_rgb.gif'.format(parse_filename(filename1)[1],
                                             parse_filename(filename2)[1],
                                             parse_filename(filename3)[1])
    out_path = os.path.join(parse_filename(filename1)[0], out_filename)

    # Write gif file
    mimwrite(out_path, new_img, format='gif', fps=int(fps * size))


def write_gif_pseudocolor(filename, size=1, fps=18, colormap='hot', slice_orient=None):
    """Procedure for writing pseudo color image.

    The colormap can be any colormap from matplotlib.

    Parameters
    ----------
    filename1: str
        Input file (eg. /john/home/image.nii.gz)
    size: float
        Between 0 and 1.
    fps: int
        Frames per second
    colormap: str
        Name of the colormap that will be used.

    """
    # Load NIfTI and put it in right shape
    out_img, maximum = load_and_prepare_image(filename, size, slice_orient)

    # Create output mosaic
    new_img = create_mosaic_normal(out_img, maximum, slice_orient)

    # Transform values according to the color map
    cmap = get_cmap(colormap)
    color_transformed = [cmap(new_img[i, ...]) for i in range(maximum)]
    cmap_img = np.delete(color_transformed, 3, 3)

    # Figure out extension
    ext = '.{}'.format(parse_filename(filename)[2])
    # Write gif file
    mimwrite(filename.replace(ext, '_{}.gif'.format(colormap)),
             cmap_img, format='gif', fps=int(fps * size))
