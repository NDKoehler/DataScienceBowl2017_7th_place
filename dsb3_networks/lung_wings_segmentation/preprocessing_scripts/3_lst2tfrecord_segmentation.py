""" Convert img-list to tensorflow TFRecord format """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
import cv2


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(img_lst, force_grayscale, record_file, img_shape, interpolation, img_paths_lst, lab_paths_lst):
    writer = tf.python_io.TFRecordWriter(record_file)
    with open(img_lst) as lst:
        # with tf.Session() as sess:
        for cnt, line in enumerate(lst):
            line = line.strip().split('\t')
            key = line[0]

            img_paths = line[2].split(',')
            lab_paths = line[1].split(',')

            images = []

            for img_cnt, img_path in enumerate(img_paths):

                if img_paths_lst and not img_cnt in img_paths_lst:
                    continue

                if force_grayscale:
                    img = cv2.imread(img_path, 0)
                else:
                    img = cv2.imread(img_path, -1)
                if img_shape:
                    img = cv2.resize(
                        img, img_shape, interpolation=interpolation)
                if len(img.shape) == 2:
                    img = np.expand_dims(img, 2)
                if img.shape[2] == 1:
                    images.append(img[:, :, 0])
                else:
                    for channel in range(3):
                        images.append(img[:, :, channel])

            img = np.array(images)
            img = img.transpose(1, 2, 0)

            labels = []

            for lab_cnt, lab_path in enumerate(lab_paths):

                if lab_paths_lst and not lab_cnt in lab_paths_lst:
                    continue

                lab = cv2.imread(lab_path, 0)
                if img_shape:
                    lab = cv2.resize(
                        lab, img_shape, interpolation=interpolation)
                if len(lab.shape) == 2:
                    lab = np.expand_dims(lab, 2)
                if lab.shape[2] == 1:
                    labels.append(lab[:, :, 0])
                else:
                    for channel in range(3):
                        labels.append(lab[:, :, channel])

            lab = np.array(labels)
            lab = lab.transpose(1, 2, 0)

            height = img.shape[0]
            width = img.shape[1]
            image_paths = img.shape[2]
            label_paths = lab.shape[2]

            #img = img.astype(np.float)/128.-1.0 # normalize and zerocenter
            #lab = lab.astype(np.float)/255.

            image_proto = tf.contrib.util.make_tensor_proto(
                img, dtype=tf.float32, shape=img.shape)
            label_proto = tf.contrib.util.make_tensor_proto(
                lab, dtype=tf.float32, shape=lab.shape)

            image_shape_value = [height, width, image_paths]
            label_shape_value = [height, width, label_paths]

            example = tf.train.Example(features=tf.train.Features(feature={

                                                                  'image_shape': _int64_feature([height, width, image_paths]),
                                                                  'label_shape': _int64_feature([height, width, label_paths]),

                                                                  'label': _bytes_feature(label_proto.SerializeToString()),
                                                                  'image_raw': _bytes_feature(image_proto.SerializeToString())}))
            if cnt % 100 == 0:
                print('Counter: {}, key : {}, label : {}, path : {}, img-shape : {}'.format(
                    cnt, key, lab.shape, img_paths, img.shape))
            writer.write(example.SerializeToString())

    writer.close()
    lst.close()


def read_records(record_file):
    filename_queue = tf.train.string_input_producer(
        [record_file], num_epochs=None)
    reader = tf.TFRecordReader()
    key, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            #'height': tf.FixedLenFeature([], tf.int64),
            #'width': tf.FixedLenFeature([], tf.int64),

            'image_shape': tf.FixedLenFeature([3], tf.int64),
            'label_shape': tf.FixedLenFeature([3], tf.int64),

            'label': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string),
        })

    print(features['image_shape'])

    label = features['label']
    label = tf.parse_tensor(label, tf.float32, name=None)
    # label = tf.reverse(label,[False,False,True])
    label = tf.reshape(label, tf.cast(features['label_shape'], tf.int32))

    image = features['image_raw']
    image = tf.parse_tensor(image, tf.float32, name=None)
    # image = tf.reverse(image,[False,False,True])
    image = tf.reshape(image,  tf.cast(features['image_shape'], tf.int32))

    return image, label


def main(_):
    run_training()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    img_paths_lst = [0]

    parser = argparse.ArgumentParser(
        description="Convert img-lists format to tensorflow TFRecord format")
    parser.add_argument(
        '--class_id', type=int, required=True, help="class_id")
    parser.add_argument(
        '--img_lst', type=str, required=True, help="img-list file")
    parser.add_argument(
        '--force_grayscale', type=bool, required=None, help="force grayscale")
    parser.add_argument(
        '--record_file', type=str, default='tfrecord.tfr', help="record file")
    parser.add_argument(
        '--resize', type=str, default=None, help="resize all images to WxH")
    parser.add_argument('--interpolation', type=str, default='cv2.INTER_CUBIC',
                        help="interpolation for resize function: cv2.INTER_CUBIC or cv2.INTER_AREA")

    args = parser.parse_args()

    lab_paths_lst = [args.class_id]
    print(lab_paths_lst)

    print(args.img_lst)
    print(args.record_file)
    if args.resize:
        print('resize images to: {}'.format(args.resize))
        img_shape = tuple(map(int, (args.resize.split('x'))))
    else:
        img_shape = None

    if args.interpolation == 'cv2.INTER_CUBIC':
        interpolation = cv2.INTER_CUBIC
    elif args.interpolation == 'cv2.INTER_AREA':
        interpolation = cv2.INTER_AREA
    else:
        print ('ERROR with interpolation')
        sys.exit()
    print('interpolation function: {}'.format(args.interpolation))

    if args.force_grayscale:
        force_grayscale = True
    else:
        force_grayscale = False

    convert(args.img_lst,
            force_grayscale, args.record_file, img_shape, interpolation, img_paths_lst, lab_paths_lst)

    image, label = read_records(args.record_file)

    sess = tf.Session()
    init_op = tf.initialize_all_variables()

    sess.run(init_op)

    tf.train.start_queue_runners(sess=sess)

    for i in range(4):

        img1, lab1 = sess.run([image, label])

        img1 /= 255
        img1 = img1.transpose(2, 0, 1)
        lab1 /= 255
        lab1 = lab1.transpose(2, 0, 1)

        for img in img1:
            cv2.imshow('img', img)
            for lab in lab1:
                cv2.imshow('lab', lab)
                cv2.waitKey(0)
