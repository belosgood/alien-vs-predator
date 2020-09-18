import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import os

train_alien_path = pathlib.Path('./data/train/alien')
train_pred_path = pathlib.Path('./data/train/predator')
test_alien_path = pathlib.Path('./data/validation/alien')
test_pred_path = pathlib.Path('./data/validation/predator')

# I think it might be easiest to load everything in at once the split later
alien_train_list = sorted([str(path) for path in train_alien_path.glob('*.jpg')])
pred_train_list = sorted([str(path) for path in train_pred_path.glob('*.jpg')])
alien_test_list = sorted([str(path) for path in test_alien_path.glob('*.jpg')])
pred_test_list = sorted([str(path) for path in test_pred_path.glob('*.jpg')])

# only use training data (for now?)
file_list = alien_train_list + pred_train_list  # + alien_test_list + pred_test_list

# test data needs to be read in

fig = plt.figure(figsize=(10, 5))
for i, file in enumerate(file_list[:6]):  # Just show first 6 aliens
    img_raw = tf.io.read_file(file)
    img = tf.image.decode_image(img_raw)
    print('Image shape: ', img.shape)
    ax = fig.add_subplot(2, 3, i + 1)
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)

plt.tight_layout()
plt.show()

labels = [1 if 'alien' in file else 0 for file in file_list]

ds_files_labels = tf.data.Dataset.from_tensor_slices(
    (file_list, labels))

# for item in ds_files_labels:
#     print(item[0].numpy(), item[1].numpy())

def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image /= 255.0

    return image, label

img_width, img_height = 120, 80

ds_train = ds_files_labels.map(load_and_preprocess)

# seperate train and validate

tf.random.set_seed(1)

ds = ds_train.shuffle(buffer_size=150, reshuffle_each_iteration=False)

ds_valid = ds_train.take(150).batch(32)
ds_train = ds_train.skip(150).batch(32)

features, labels = next(iter(ds_train))

print(features)
print(labels)

# ds_valid = ds_train.take(150)
# ds_train = ds_train.skip(150)

# fig = plt.figure(figsize=(10, 5))
# for i,example in enumerate(ds):
#     if i < 6: # Just display first 6
#         print(example[0].shape, example[1].numpy())
#         ax = fig.add_subplot(2, 3, i+1)
#         ax.set_xticks([]); ax.set_yticks([])
#         ax.imshow(example[0])
#         ax.set_title('{}'.format(example[1].numpy()),
#                      size=15)

# plt.tight_layout()
# plt.show()