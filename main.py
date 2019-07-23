from absl import app,flags
import tensorflow as tf
if tf.version.VERSION.startswith('1.'):
    tf.enable_eager_execution()
if tf.version.VERSION.startswith('1.'):
    tf.enable_v2_behavior()
if tf.version.VERSION.startswith('1.'):
    tf.enable_v2_tensorshape()
from yolo3.enum import BACKBONE,MODE,OPT
from train import train
from train_backbone import train as train_backbone
from yolo import YOLO, detect_video, detect_img,export_tflite_model,export_serving_model,calculate_map,export_tfjs_model
import os


FLAGS=flags.FLAGS

flags.DEFINE_enum_class('backbone',default=BACKBONE.MOBILENETV2,enum_class=BACKBONE,help="Select network backbone, One of {'MOBILENETV2','DARKNET53','EFFICIENTNET'}")
flags.DEFINE_integer('batch_size',default=8,lower_bound=0,help="Train batch size")
flags.DEFINE_string('config',default=None,help="Config path")
flags.DEFINE_multi_integer('epochs',default=[10,10],lower_bound=0,help="Frozen train epochs and Full train epochs")
flags.DEFINE_string('export',default='export_model/8',help="Export path")
flags.DEFINE_string('input',default=None,help="Input data for various mode")
flags.DEFINE_multi_integer('input_size',default=(380,380),lower_bound=0,help="Input size")
flags.DEFINE_string('log_directory',default=None,help="Log directory")
flags.DEFINE_string('model',default='../download/mobilenetv2_trained_weights_final (1).h5',help="Model path")
flags.DEFINE_enum_class('mode',default=MODE.SERVING,enum_class=MODE,help="Select exec mode, One of {'TRAIN','TRAIN_BACKBONE','IMAGE','VIDEO','TFLITE','SERVING','MAP','PRUNE'}")
flags.DEFINE_string('gpus',default='0',help="Specific gpu indexes to run")
flags.DEFINE_string('train_dataset',default='../pascal/VOCdevkit/train/*.tfrecords',help="Dataset glob for train")
flags.DEFINE_string('val_dataset',default='../pascal/VOCdevkit/val/*.tfrecords',help="Dataset glob for validate")
flags.DEFINE_string('test_dataset',default='../pascal/VOCdevkit/test/*.tfrecords',help="Dataset glob for test")
flags.DEFINE_string('anchors_path',default='model_data/yolo_anchors.txt',help="Anchors path")
flags.DEFINE_string('classes_path',default='model_data/voc_classes.txt',help="Classes Path")
flags.DEFINE_multi_float('learning_rate',default=[1e-3,1e-4],lower_bound=0,help="Learning rate")
flags.DEFINE_enum_class('opt',default=None,enum_class=OPT,help="Select optimization, One of {'XLA','DEBUG','MKL'}")
flags.DEFINE_bool('use_tpu',default=False,help="Whether use tpu")
flags.DEFINE_bool('prune',default=False,help="Whether prune model")

def parse_tuple(val):
    if isinstance(val,str):
        return tuple([int(num) for num in val[1:-1].split(',')])
    return tuple(val)

def log(msg):
    if tf.version.VERSION.startswith('1.'):
        tf.logging.info(msg)
    else:
        print(msg)
        
def main(_):
    flags_dict=FLAGS.flag_values_dict()
    if FLAGS.config is not None:
        import yaml
        with open(FLAGS.config) as stream:
            config=yaml.safe_load(stream)
            if 'backbone' in config:
                config['backbone']=BACKBONE[config['backbone']]
            if 'opt' in config:
                config['opt'] = OPT[config['opt']]
            if 'input_size' in config:
                if isinstance(config['input_size'],str):
                    config['input_size']=parse_tuple(config['input_size'])
                elif isinstance(config['input_size'],list):
                    config['input_size']=[parse_tuple(size) for size in config['input_size']]
                else:
                    raise ValueError('Please use array or tuple to define input_size')
            if 'learning_rate' in config:
                config['learning_rate']=[float(lr) for lr in config['learning_rate']]
            flags_dict.update(config)

    os.environ["CUDA_VISIBLE_DEVICES"] = flags_dict['gpus']
    if flags_dict['backbone'] is None:
        raise ValueError("Please select your model's backbone")
    if FLAGS.mode==MODE.TRAIN:
        log('Train mode')
        train(flags_dict)
    elif FLAGS.mode==MODE.TRAIN_BACKBONE:
        log('Train backbone mode')
        train_backbone(flags_dict)
    elif FLAGS.mode==MODE.IMAGE:
        if flags_dict['model'] is None:
            raise ValueError('Please enter your model path')
        log('Image detection mode')
        detect_img(YOLO(flags_dict))
    elif FLAGS.mode==MODE.VIDEO:
        if flags_dict['model'] is None:
            raise ValueError('Please enter your model path')
        log('Video detection mode')
        detect_video(YOLO(flags_dict), FLAGS.input, FLAGS.output)
    elif FLAGS.mode==MODE.MAP:
        if flags_dict['model'] is None:
            raise ValueError('Please enter your model path')
        log('Calculate test dataset map')
        calculate_map(YOLO(flags_dict),FLAGS.test_dataset)
    elif FLAGS.mode==MODE.SERVING:
        tf.disable_eager_execution()
        log('Export hdf5 model to serving model')
        export_serving_model(YOLO(flags_dict),FLAGS.export)
    elif FLAGS.mode==MODE.TFLITE:
        log('Export hdf5 model to tflite model')
        export_tflite_model(YOLO(flags_dict),FLAGS.export)
    elif FLAGS.mode==MODE.TFJS:
        log('Export hdf5 model to tensorflow.js model')
        export_tfjs_model(YOLO(flags_dict),FLAGS.export)


if __name__=='__main__':
    if tf.version.VERSION.startswith('1.'):
        tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
