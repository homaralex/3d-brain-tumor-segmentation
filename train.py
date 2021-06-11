"""Contains training loops."""
import os
import tensorflow as tf
import numpy as np
import tensorflow.python.framework.ops

import wandb
from tqdm import tqdm

from args import TrainArgParser
from util import DiceVAELoss, DiceCoefficient, ScheduledOptim
from model import Model


def prepare_dataset(loc, batch_size, prepro_size, crop_size, out_ch, shuffle=True, data_format='channels_last'):
    """Returns a BatchDataset object containing loaded data."""

    def parse_example(example_proto):
        # Parse examples.
        parsed = tf.io.parse_single_example(example_proto, example_desc)
        x = tf.reshape(parsed['x'], (h, w, d, c))
        y = tf.reshape(parsed['y'], (h, w, d, 1))

        # Apply random intensity shift.
        _, var = tf.nn.moments(x, axes=(0, 1, 2), keepdims=True)
        shift = tf.random.uniform((1, 1, 1, c), -0.1, 0.1)
        scale = tf.random.uniform((1, 1, 1, c), 0.9, 1.1)
        x += shift * tf.sqrt(var)
        x *= scale

        # Create random crop.
        xy = tf.concat([x, y], axis=-1)
        xy = tf.image.random_crop(xy, crop_size + [c + 1])

        # Randomly flip across each axis.
        for axis in (0, 1, 2):
            xy = tf.cond(
                tf.random.uniform(()) > 0.5,
                lambda: tf.reverse(xy, axis=[axis]),
                lambda: xy)

        x, y = tf.split(xy, [c, 1], axis=-1)

        # Convert labels into one-hot encodings.
        y = tf.squeeze(tf.cast(y, tf.int32), axis=-1)
        y = tf.one_hot(y, out_ch + 1, axis=-1, dtype=tf.float32)
        _, y = tf.split(y, [1, out_ch], axis=-1)

        if data_format == 'channels_first':
            x = tf.transpose(x, (3, 0, 1, 2))
            y = tf.transpose(y, (3, 0, 1, 2))

        return (x, y)

    h, w, d, c = prepro_size

    example_desc = {
        'x': tf.io.FixedLenFeature([h * w * d * c], tf.float32),
        'y': tf.io.FixedLenFeature([h * w * d * 1], tf.float32)}

    files = [os.path.join(loc, f) for f in os.listdir(loc)]
    dataset = tf.data.TFRecordDataset(files)

    # Shuffle for training set, else no shuffle for validation set.
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(files))

    dataset = (dataset.map(map_func=parse_example, num_parallel_calls=tf.data.experimental.AUTOTUNE)
               .batch(batch_size=batch_size)
               .prefetch(buffer_size=tf.data.experimental.AUTOTUNE))

    return dataset, len(files)


def train(args):
    # Load data.
    train_data, n_train = prepare_dataset(
        args.train_loc,
        args.batch_size,
        args.prepro_size,
        args.crop_size,
        args.model_args['out_ch'],
        shuffle=True,
        data_format=args.data_format)
    val_data, n_val = prepare_dataset(
        args.val_loc,
        args.batch_size,
        args.prepro_size,
        args.crop_size,
        args.model_args['out_ch'],
        shuffle=False,
        data_format=args.data_format)
    print('{} training examples.'.format(n_train), flush=True)
    print('{} validation examples.'.format(n_val), flush=True)

    # Initialize model.
    in_ch = args.prepro_size[-1]
    model = Model(**args.model_args)
    _ = model(tf.zeros(shape=[1] + args.crop_size + [in_ch] if args.data_format == 'channels_last' \
        else [1, in_ch] + args.crop_size))
    model.summary()

    # Load weights if specified.
    if args.load_folder:
        model.load_weights(os.path.join(args.load_folder, 'chkpt.hdf5'))

    # Initialize loss and metrics.
    optimizer = ScheduledOptim(learning_rate=args.lr)
    loss_fn = DiceVAELoss(data_format=args.data_format)
    dice_fn = DiceCoefficient(data_format=args.data_format)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_macro_dice = tf.keras.metrics.Mean(name='train_macro_dice')
    train_micro_dice = tf.keras.metrics.Mean(name='train_micro_dice')
    train_kld = tf.keras.metrics.Mean(name='train_kld')
    train_rec = tf.keras.metrics.Mean(name='train_rec')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_macro_dice = tf.keras.metrics.Mean(name='val_macro_dice')
    val_micro_dice = tf.keras.metrics.Mean(name='val_micro_dice')
    val_kld = tf.keras.metrics.Mean(name='val_kld')
    val_rec = tf.keras.metrics.Mean(name='val_rec')

    # Set up logging.
    if args.save_folder:
        with open(os.path.join(args.save_folder, 'train.log'), 'w') as f:
            header = ','.join(['epoch',
                               'lr',
                               'train_loss',
                               'train_macro_dice',
                               'train_micro_dice',
                               'val_loss',
                               'val_macro_dice',
                               'val_micro_dice'])
            f.write(header + '\n')

    wandb_project = args.wandb_project
    if wandb_project is not None:
        wandb.init(project=wandb_project, config=vars(args))

        wandb_data, _ = prepare_dataset(
            args.val_loc,
            args.batch_size,
            args.prepro_size,
            args.crop_size,
            args.model_args['out_ch'],
            shuffle=False,
            data_format=args.data_format)

    best_val_dice = 0.0
    patience = 0

    # Train.
    for epoch in range(model.epoch.value().numpy(), args.n_epochs):
        print('Epoch {}.'.format(epoch))
        model.epoch.assign(epoch)
        optimizer(epoch=epoch)

        with tf.device(args.device):
            # Training epoch.
            for x, y in tqdm(train_data, total=n_train, desc='Training      '):
                # Forward and loss.
                with tf.GradientTape() as tape:
                    y_pred, y_vae, z_mean, z_logvar = model(x, training=True, inference=False)

                    loss = loss_fn(x, y, y_pred, y_vae, z_mean, z_logvar)
                    loss += tf.reduce_sum(model.losses)

                macro_dice, micro_dice = dice_fn(y, y_pred)
                l2_loss = tf.reduce_mean((x - y_vae) ** 2)
                kld_loss = tf.reduce_mean(z_mean ** 2 + tf.math.exp(z_logvar) - z_logvar - 1.0)

                # Gradients and backward.
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                # Update logs.
                train_loss.update_state(loss)
                train_kld.update_state(kld_loss)
                train_rec.update_state(l2_loss)
                train_macro_dice.update_state(macro_dice)
                train_micro_dice.update_state(micro_dice)

            print('Training. Loss: {l: .4f}, Macro Dice: {d1: 1.4f}, Micro Dice: {d2: 1.4f}.'
                  .format(l=train_loss.result(),
                          d1=train_macro_dice.result(),
                          d2=train_micro_dice.result()), flush=True)

            # Validation epoch.
            for x, y in tqdm(val_data, total=n_val, desc='Validation    '):
                # Forward and loss.
                y_pred, y_vae, z_mean, z_logvar = model(x, training=False, inference=False)

                loss = loss_fn(x, y, y_pred, y_vae, z_mean, z_logvar)
                loss += tf.reduce_sum(model.losses)

                macro_dice, micro_dice = dice_fn(y, y_pred)
                l2_loss = tf.reduce_mean((x - y_vae) ** 2)
                kld_loss = tf.reduce_mean(z_mean ** 2 + tf.math.exp(z_logvar) - z_logvar - 1.0)

                val_loss.update_state(loss)
                val_rec.update_state(l2_loss)
                val_kld.update_state(kld_loss)
                val_macro_dice.update_state(macro_dice)
                val_micro_dice.update_state(micro_dice)

            print('Validation. Loss: {l: .4f}, Macro Dice: {d1: 1.4f}, Micro Dice: {d2: 1.4f}.'
                  .format(l=val_loss.result(),
                          d1=val_macro_dice.result(),
                          d2=val_micro_dice.result()), flush=True)

        # Write logs.
        if args.save_folder:
            with open(os.path.join(args.save_folder, 'train.log'), 'a') as f:
                entry = ','.join([str(epoch),
                                  str(optimizer.learning_rate.numpy()),
                                  str(train_loss.result().numpy()),
                                  str(train_macro_dice.result().numpy()),
                                  str(train_micro_dice.result().numpy()),
                                  str(val_loss.result().numpy()),
                                  str(val_macro_dice.result().numpy()),
                                  str(val_micro_dice.result().numpy())])
                f.write(entry + '\n')

        if wandb_project is not None:
            wandb.log({
                'epoch': epoch,
                'lr': optimizer.learning_rate.numpy(),
                'train_loss': train_loss.result().numpy(),
                'train_kld': train_kld.result().numpy(),
                'train_rec': train_rec.result().numpy(),
                'train_macro_dice': train_macro_dice.result().numpy(),
                'train_micro_dice': train_micro_dice.result().numpy(),
                'val_loss': val_loss.result().numpy(),
                'val_kld': val_kld.result().numpy(),
                'val_rec': val_rec.result().numpy(),
                'val_macro_dice': val_macro_dice.result().numpy(),
                'val_micro_dice': val_micro_dice.result().numpy(),
            })

            segs, recs = [], []
            for x, y in wandb_data:
                if len(segs) >= 4:
                    break

                # take the middle slice
                slice_idx = x[0].shape[1] // 2
                if args.data_format == 'channels_first':
                    seg_orig = y[0][0][slice_idx].numpy()
                    if seg_orig.sum() == 0:
                        continue

                    y_pred, y_vae, _, _ = model(x, training=False, inference=False)
                    rec = y_vae[0][0][slice_idx].numpy()
                    orig = x[0][0][slice_idx].numpy()
                    seg = y_pred[0][0][slice_idx].numpy()
                else:
                    seg_orig = y[0][:, :, :, 0][slice_idx].numpy()
                    if seg_orig.sum() == 0:
                        continue

                    y_pred, y_vae, z_mean, z_logvar = model(x, training=False, inference=False)
                    rec = y_vae[0][:, :, :, 0][slice_idx].numpy()
                    orig = x[0][:, :, :, 0][slice_idx].numpy()
                    seg = y_pred[0][:, :, :, 0][slice_idx].numpy()

                recs.append(wandb.Image(rec))
                mask_img = wandb.Image(orig, masks={
                    "predictions": {
                        "mask_data": seg,
                    },
                    "ground_truth": {
                        "mask_data": seg_orig,
                    },
                })
                segs.append(mask_img)

            wandb.log({"reconstructions": recs})
            wandb.log({"segmentations": segs})

        # Checkpoint and patience.
        if val_macro_dice.result() > best_val_dice:
            best_val_dice = val_macro_dice.result()
            patience = 0
            if args.save_folder:
                model.save_weights(os.path.join(args.save_folder, 'chkpt.hdf5'))
            print('Saved model weights.', flush=True)
        elif patience == args.patience:
            print('Validation dice has not improved in {} epochs. Stopped training.'
                  .format(args.patience), flush=True)
            return
        else:
            patience += 1

        # Reset statistics.
        train_loss.reset_states()
        train_kld.reset_states()
        train_rec.reset_states()
        train_macro_dice.reset_states()
        train_micro_dice.reset_states()
        val_loss.reset_states()
        val_kld.reset_states()
        val_rec.reset_states()
        val_macro_dice.reset_states()
        val_micro_dice.reset_states()


if __name__ == '__main__':
    parser = TrainArgParser()
    args = parser.parse_args()
    print('Train args: {}'.format(args), flush=True)
    # tensorflow.python.framework.ops.disable_eager_execution()
    train(args)
