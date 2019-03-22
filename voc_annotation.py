import xml.etree.ElementTree as ET
import tensorflow as tf
from os import path
import numpy as np
classes = ["FlowManagement", "Heapmaterial", "Illegal_parking", "laji", "outManagement"]
tfrecords_size = 1000

def convert_to_tfrecord(xml, record_writer):
    name, _ = xml.split('/')[-1].split('.')
    root = ET.parse(xml.encode('utf-8')).getroot()
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []
    labels = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        xmins.append(float(xmlbox.find('xmin').text))
        ymins.append(float(xmlbox.find('ymin').text))
        xmaxs.append(float(xmlbox.find('xmax').text))
        ymaxs.append(float(xmlbox.find('ymax').text))
        labels.append(int(cls_id))
    with tf.gfile.Open(tf.gfile.Glob('%s/%s/**/%s.jp*g' % (clazz, file, name))[0], 'rb') as image_file:
        image_data=image_file.read()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
            'image/object/bbox/label': tf.train.Feature(float_list=tf.train.FloatList(value=labels))
        }))
        record_writer.write(example.SerializeToString())


for clazz in classes:
    index_records = 1
    num = 1
    record_writer = tf.io.TFRecordWriter(
        path.join('./', 'cci_%d_%s.tfrecords' % (index_records, clazz)))

    for file in tf.gfile.ListDirectory(clazz):
        if tf.gfile.IsDirectory('%s/%s' % (clazz, file)):
            xmls = tf.gfile.Glob('%s/%s/**/*.xml' % (clazz, file))
            np.random.shuffle(xmls)
            for xml in xmls:
                if num >= tfrecords_size:
                    tf.gfile.Rename('cci_%d_%s.tfrecords' % (index_records, clazz),
                                    'cci_%d_%s_%d.tfrecords' % (index_records, clazz, num))
                    index_records += 1
                    num = 1
                    record_writer.close()
                    record_writer = tf.io.TFRecordWriter(
                        path.join('./', 'cci_%d_%s.tfrecords' % (index_records, clazz)))
                convert_to_tfrecord(xml, record_writer)
                num += 1
            tf.gfile.Rename('cci_%d_%s.tfrecords' % (index_records, clazz),
                            'cci_%d_%s_%d.tfrecords' % (index_records, clazz, num))

            record_writer.close()
