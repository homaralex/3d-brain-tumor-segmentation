"""Contains inference script for generating segmentation mask."""
import os
import glob
import tensorflow as tf
import numpy as np
import nibabel as nib
from tqdm import tqdm

from utils.arg_parser import test_parser
from utils.constants import *
from model.volumetric_cnn import VolumetricCNN


def get_npy_image(subject_folder, name):
    """Returns np.array from .nii files."""
    try:
        file_card = glob.glob(os.path.join(subject_folder, '*' + name + '.nii' + '*'))[0]
    except:
        file_card = glob.glob(os.path.join(subject_folder, '*' + name + '_proc.nii'))[0]
    return np.array(nib.load(file_card).dataobj).astype(np.float32)


def convert_to_nii(save_folder, mask):
    affine = np.array([[-0.977, 0., 0., 91.1309967],
                       [0., -0.977, 0., 149.34700012],
                       [0., 0., 1., -74.67500305],
                       [0., 0., 0., 1.]])
    img = nib.Nifti1Image(mask, affine)
    nib.save(img, os.path.join(save_folder, 'mask.nii'))


def prepare_image(X, voxel_mean, voxel_std, data_format):
    """Normalize image and sample crops to represent whole input."""
    # Expand batch dimension and normalize.
    X = (X - voxel_mean) / voxel_std

    # Crop to size.
    if data_format == 'channels_last':
        crop1 = X[:H, :W, :D, :]
        crop2 = X[-H:, :W, :D, :]
        crop3 = X[:H, -W:, :D, :]
        crop4 = X[-H:, -W:, :D, :]
        crop5 = X[:H, :W, -D:, :]
        crop6 = X[-H:, :W, -D:, :]
        crop7 = X[:H, -W:, -D:, :]
        crop8 = X[-H:, -W:, -D:, :]
    elif data_format == 'channels_first':
        crop1 = X[:, :H, :W, :D]
        crop2 = X[:, -H:, :W, :D]
        crop3 = X[:, :H, -W:, :D]
        crop4 = X[:, -H:, -W:, :D]
        crop5 = X[:, :H, :W, -D:]
        crop6 = X[:, -H:, :W, -D:]
        crop7 = X[:, :H, -W:, -D:]
        crop8 = X[:, -H:, -W:, -D:]
        
    return np.stack([crop1, crop2, crop3, crop4, crop5, crop6, crop7, crop8], axis=0)


def create_mask(y, data_format):
    def merge_h(h1, h2):
        if data_format == 'channels_last':
            return tf.concat([h1[:-h_overlap, :, :, :],
                              tf.minimum(h1[-h_overlap:, :, :, :], h2[:h_overlap, :, :, :]),
                              h2[h_overlap:, :, :, :]], axis=0)
        elif data_format == 'channels_first':
            return tf.concat([h1[:, :-h_overlap, :, :],
                              tf.minimum(h1[:, -h_overlap:, :, :], h2[:, :h_overlap, :, :]),
                              h2[:, h_overlap:, :, :]], axis=1)

    def merge_w(w1, w2):
        if data_format == 'channels_last':
            return tf.concat([w1[:, :-w_overlap, :, :],
                              tf.minimum(w1[:, -w_overlap:, :, :], w2[:, :w_overlap, :, :]),
                              w2[:, w_overlap:, :, :]], axis=1)
        elif data_format == 'channels_first':
            return tf.concat([w1[:, :, :-w_overlap, :],
                              tf.minimum(w1[:, :, -w_overlap:, :], w2[:, :, :w_overlap, :]),
                              w2[:, :, w_overlap:, :]], axis=2)

    def merge_d(d1, d2):
        if data_format == 'channels_last':
            return tf.concat([d1[:, :, :-d_overlap :],
                              tf.minimum(d1[:, :, -d_overlap:, :], d2[:, :, :d_overlap, :]),
                              d2[:, :, d_overlap:, :]], axis=2)
        elif data_format == 'channels_first':
            return tf.concat([d1[:, :, :-d_overlap :],
                              tf.minimum(d1[:, :, :,  -d_overlap:], d2[:, :, :, :d_overlap]),
                              d2[:, :, :, d_overlap:]], axis=-1)

    crop1, crop2, crop3, crop4, crop5, crop6, crop7, crop8 = [y[i, ...] for i in range(8)]

    # Calculate region of overlap.
    h_overlap = 2 * H - RAW_H
    w_overlap = 2 * W - RAW_W
    d_overlap = 2 * D - RAW_D

    h1 = merge_h(crop1, crop2)
    h2 = merge_h(crop3, crop4)
    h3 = merge_h(crop5, crop6)
    h4 = merge_h(crop7, crop8)

    w1 = merge_w(h1, h2)
    w2 = merge_w(h3, h4)

    d1 = merge_d(w1, w2)

    axis = -1 if data_format == 'channels_last' else 0

    # Mask out values that correspond to values < 0.5.
    mask = tf.reduce_max(d1, axis=axis)
    mask = tf.cast(mask > 0.5, tf.int32)

    # Take the argmax to determine label.
    seg_mask = tf.argmax(d1, axis=axis, output_type=tf.int32)

    # Convert [0, 1, 2] to [1, 2, 3] for consistency.
    seg_mask += 1

    # Mask out values that correspond to <0.5 probability.
    seg_mask *= mask

    return seg_mask


def main(args):
    # Initialize.
    data = np.load(args.prepro_file)
    voxel_mean = data.item()['mean'].astype(np.float32)
    voxel_std = data.item()['std'].astype(np.float32)
    
    # Initialize model.
    model = VolumetricCNN(
                        data_format=args.data_format,
                        kernel_size=args.conv_kernel_size,
                        groups=args.gn_groups,
                        reduction=args.se_reduction,
                        use_se=args.use_se,
                        kernel_regularizer=tf.keras.regularizers.l2(l=args.l2_scale),
                        kernel_initializer=args.kernel_init,
                        downsampling=args.downsamp,
                        upsampling=args.upsamp,
                        normalization=args.norm)

    # Build model with initial forward pass.
    _ = model(tf.zeros(shape=(1, H, W, D, IN_CH) if args.data_format == 'channels_last' \
                                    else (1, IN_CH, H, W, D)))
    # Load weights.
    model.load_weights(args.chkpt_file)

    # Loop through each patient.
    for subject_folder in tqdm(glob.glob(os.path.join(args.test_folder, '*'))):

        # Extract raw images from each.
        if args.data_format == 'channels_last':
            X = np.stack(
                [get_npy_image(subject_folder, name) for name in BRATS_MODALITIES], axis=-1)
        elif args.data_format == 'channels_first':
            X = np.stack(
                [get_npy_image(subject_folder, name) for name in BRATS_MODALITIES], axis=0)

        # Prepare input image.
        X = prepare_image(X, voxel_mean, voxel_std, args.data_format)

        # Forward pass.
        with tf.device(args.device):
            y, *_ = model(X, training=False)

            # Create mask.
            mask = create_mask(y, args.data_format).numpy()

            # Replace label `3` with `4` for consistency with data.
            np.place(mask, mask >= 3, [4])

            # Save as Numpy.
            np.save(os.path.join(args.test_folder, subject_folder, 'mask.npy'), mask)

            # Save as Nifti.
            convert_to_nii(os.path.join(args.test_folder, subject_folder), mask)


if __name__ == '__main__':
    args = test_parser()
    print(args)
    main(args)
