import tensorflow as tf
import argparse
import xml.etree.ElementTree as ET
import os
import random

if hasattr(tf, 'enable_eager_execution'):
    tf.enable_eager_execution()

parser=argparse.ArgumentParser(argument_default=argparse.SUPPRESS)

parser.add_argument('--folder',type=str,help='parse folder path',default='./')

FLAGS = parser.parse_args()
xmls=tf.io.gfile.glob(os.path.join(FLAGS.folder,'**/*.xml'))
#classes = ["FlowManagement", "Heapmaterial", "Illegal_parking", "laji", "outManagement"]
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
train_file = open('train.txt', 'w')
val_file = open('val.txt', 'w')
test_file = open('test.txt', 'w')
train_num=0
val_num=0
test_num=0
splits=[0.8,0.9,1]
for xml_path in xmls:
    image_path=tf.io.gfile.glob(os.path.join('/'.join(xml_path.split('/')[:-2]),'**',xml_path.split('/')[-1].split('.')[0]+'.jp*g'))[0]
    rand = random.random()
    if rand < splits[0]:
        file=train_file
        num=train_num
    elif rand >= splits[0] and rand < splits[1]:
        file=val_file
        num = val_num
    else:
        file=test_file
        num = test_num
    xml_root=ET.parse(xml_path.encode('utf-8')).getroot()
    file.write(image_path)
    for obj in xml_root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        file.write(' '+' '.join([str(a) for a in b])+' '+ str(cls_id))
    file.write('\n')
    num+=1
train_file.close()
val_file.close()
test_file.close()
os.rename('train.txt','voc_train_'+str(train_num)+'.txt')
os.rename('val.txt','voc_val_'+str(val_num)+'.txt')
os.rename('test.txt','voc_test_'+str(test_num)+'.txt')
