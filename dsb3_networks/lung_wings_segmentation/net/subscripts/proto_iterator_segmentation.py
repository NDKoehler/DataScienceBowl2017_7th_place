'''
Proto Iterator class
'''
from hellsicht.tensorflow.io.iterator import Iterator
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import array_ops


class Proto_Iterator(Iterator):

    def __init__(self, H, **conf):
        super(Proto_Iterator, self).__init__()

        self.record_file = conf['record_file']
        self.img_shape = conf['img_shape']
        self.label_shape = conf['label_shape']

        self.in_img_shape = H['in_image_shape']
        self.in_lab_shape = H['in_label_shape']

        self.batch_size = conf['batch_size']
        self.num_preprocess_threads = conf['num_preprocess_threads']
        self.shuffle = conf['shuffle']
        self.is_training = conf['is_training']

        self.crop_min = H['crop_min']
        self.crop_max = H['crop_max']
        self.rand_crop = H['rand_crop']
        self.rand_rot = H['rand_rot']
        self.rand_flip_ud = H['rand_flip_ud']
        self.rand_flip_lr = H['rand_flip_lr']

        self.rand_brightness = H['brightness']
        self.contrast_lower = H['contrast_lower']
        self.contrast_upper = H['contrast_upper']

        self.images = None
        self.labels = None


        #set seed
        tf.set_random_seed(1234)



    def read_data_(self):
        '''
        Returns Tensors for images and labels
        '''
        if not tf.gfile.Exists(self.record_file):
            raise IOError('record_file not found: {}'.format(self.record_file))

        with tf.variable_scope('InputProducer') as scope:
            filename_queue = tf.train.string_input_producer(
                [self.record_file], num_epochs=None)
            reader = tf.TFRecordReader()
            key, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image_shape': tf.FixedLenFeature([3], tf.int64),
                    'label_shape': tf.FixedLenFeature([3], tf.int64),
                    'label': tf.FixedLenFeature([], tf.string),
                    'image_raw': tf.FixedLenFeature([], tf.string),
                })

            label = features['label']
            label = tf.parse_tensor(label, tf.float32, name=None)
            #label = tf.reverse(label,[False,False,True])	
            label = tf.reshape(label, self.in_lab_shape) #tf.cast(features['label_shape'], tf.int32)

            image = features['image_raw']
            image = tf.parse_tensor(image, tf.float32, name=None)
            #image = tf.reverse(image,[False,False,True])
            image = tf.reshape(image, self.in_img_shape) #tf.cast(features['image_shape'], tf.int32)

            return image, label

    def NormalizeImage(self, image, sub, norm):
        image = tf.subtract(image, sub)
        image = tf.div(image, norm)
        return image

    def rot90_custom(self, image, k=1, name=None):
        """Rotate an image counter-clockwise by 90 degrees.
        Args:
            image: A 3-D tensor of shape `[height, width, channels]`.
            k: A scalar integer. The number of times the image is rotated by 90 degrees.
            name: A name for this operation (optional).
        Returns:
            A rotated 3-D tensor of the same type and shape as `image`.
        """

        def _rot90():
            return array_ops.transpose(array_ops.reverse_v2(image, [1]),
                                    [1, 0, 2])
        def _rot180():
            return array_ops.reverse_v2(image, [0, 1])
        def _rot270():
            return array_ops.reverse_v2(array_ops.transpose(image, [1, 0, 2]),
                                    [1])
        cases = [(math_ops.equal(k, 1), _rot90),
                (math_ops.equal(k, 2), _rot180),
                (math_ops.equal(k, 3), _rot270)]

        ret = control_flow_ops.case(cases, default=lambda: image, exclusive=True)
        ret.set_shape([image.get_shape()[0], image.get_shape()[1], image.get_shape()[2]])
        return ret

    def augment_image_label_pair_(self, image, label):

        if self.is_training:

            if (self.crop_min and self.crop_max):

                print("cropping!", self.in_img_shape[0])

                rand_target_size = tf.random_uniform([1],int(self.crop_min),int(self.crop_max), dtype = tf.int32)[0]
                img_size_y = tf.constant(self.in_img_shape[0], dtype = tf.int32)
                img_size_x = tf.constant(self.in_img_shape[1], dtype = tf.int32)

                rand_crop_start_y = tf.random_uniform([1],0,img_size_y - rand_target_size, dtype = tf.int32)[0]
                rand_crop_start_x = tf.random_uniform([1],0,img_size_x - rand_target_size, dtype = tf.int32)[0]

                image = tf.image.crop_to_bounding_box(image, rand_crop_start_y, rand_crop_start_x, rand_target_size, rand_target_size)
                label = tf.image.crop_to_bounding_box(label, rand_crop_start_y, rand_crop_start_x, rand_target_size, rand_target_size)

                # resize to correct size
                image = tf.image.resize_images(image, [self.img_shape[0], self.img_shape[1]])
                label = tf.image.resize_images(label, [self.label_shape[0], self.label_shape[1]])

            # random rotate:
            if self.rand_rot:
                print("rotating!")

                rand_rot = tf.random_uniform([1],0,4, dtype = tf.int32)[0]
                image = self.rot90_custom(image, rand_rot)
                label = self.rot90_custom(label, rand_rot)

            # randomly flip
            if self.rand_flip_lr:
                print("flipping lr !")

                tmp = tf.constant(1, dtype = tf.int32)
                rand_var = tf.random_uniform([1],0,2, dtype = tf.int32)[0]
                rand_lr_flip = tf.less(rand_var, tmp)

                label = tf.where(rand_lr_flip, label, tf.image.flip_left_right(label))
                # label = tf.cond(rand_lr_flip > 1.0, lambda: label, lambda: tf.image.flip_left_right(label))
                image = tf.where(rand_lr_flip, image, tf.image.flip_left_right(image))
            if self.rand_flip_ud:
                print("flipping ud !")
                tmp = tf.constant(1, dtype = tf.int32)

                rand_var2 = tf.random_uniform([1],0,2, dtype = tf.int32)[0]
                rand_ud_flip = tf.less(rand_var2, tmp)

                label = tf.where(rand_ud_flip, label, tf.image.flip_up_down(label))
                image = tf.where(rand_ud_flip, image, tf.image.flip_up_down(image))

            if self.contrast_lower or self.contrast_upper or self.rand_brightness:
                image = image / 255
                image = tf.image.random_contrast(image, self.contrast_lower, self.contrast_upper)
                image = tf.clip_by_value(image, 0, 1, name=None)
                image = tf.image.random_brightness(image, self.rand_brightness)
                image = tf.clip_by_value(image, 0, 1, name=None)
                image = image * 255

        else:

            if (self.crop_min and self.crop_max):

                print("cropping!", self.in_img_shape[0])
                rand_target_size = int(np.ceil((self.crop_max + self.crop_min) / 2.0))
                print (rand_target_size)
                print (self.in_img_shape)
                rand_crop_start_x = 0
                rand_crop_start_y = 0
                if (self.in_img_shape[0] - rand_target_size) > 0:
                    rand_crop_start_y = tf.random_uniform([1],0,self.in_img_shape[0] - rand_target_size, dtype = tf.int32)[0]
                if (self.in_img_shape[1] - rand_target_size) > 0:
                    rand_crop_start_x = tf.random_uniform([1],0,self.in_img_shape[1] - rand_target_size, dtype = tf.int32)[0]

                image = tf.image.crop_to_bounding_box(image, rand_crop_start_y, rand_crop_start_x, rand_target_size, rand_target_size)
                label = tf.image.crop_to_bounding_box(label, rand_crop_start_y, rand_crop_start_x, rand_target_size, rand_target_size)

                # resize to correct size
                image = tf.image.resize_images(image, [self.img_shape[0], self.img_shape[1]])
                label = tf.image.resize_images(label, [self.label_shape[0], self.label_shape[1]])

        image = image - 128
        image = image / 128.
        # image = image / 127.5
        # image -= 1.
        return image,label


        

    def batch_input_(self, image, label):
        # Batch input
        if self.shuffle:
            image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                              batch_size=self.batch_size,
                                                              num_threads=self.num_preprocess_threads,
                                                              capacity=10 * self.batch_size,
                                                              min_after_dequeue= 9 * self.batch_size)
        else:
            image_batch, label_batch = tf.train.batch([image, label],
                                                      batch_size=self.batch_size,
                                                      num_threads=self.num_preprocess_threads,
                                                      capacity = 10 * self.batch_size)

        #if len(label_batch.get_shape().as_list()) > 1:
        #    label_batch = tf.reshape(
        #        label_batch, [self.batch_size]) if label_batch.get_shape().as_list()[1] == 1 else label_batch
        return image_batch, label_batch

    def need_queue_runners(self):
        return True

    def initialize(self):
        image, label = self.read_data_()

        '''
        #draw random parameters
        self.target_height = np.random.random_integers(int(self.crop_min), int(self.crop_max))
        self.target_width = self.target_height
        self.random_crop_start_x = np.random.randint(0, self.in_img_shape[0] - self.target_height)
        self.random_crop_start_y = np.random.randint(0, self.in_lab_shape[1] - self.target_width)   

        self.contrast_factor = np.random.randint(0.5, 1.5)
        self.brightness_delta = np.random.randint(-30, 30)

        if self.rand_flip:
            self.lr_flip = bool(np.random.randint(0,2))
            self.ud_flip = bool(np.random.randint(0,2))

        self.rot = np.random.randint(0,4)

        image = self.PreprocessImage_(image)
        label = self.PreprocessImage_(label)
        '''

        image, label = self.augment_image_label_pair_(image, label)

        #image = self.NormalizeImage(image, 128, 128)
        label = self.NormalizeImage(label, 0, 256)

        self.images, self.labels = self.batch_input_(image, label)

    def data_batch(self):
        return {'images': self.images}

    def label_batch(self):
        return {'labels': self.labels}
