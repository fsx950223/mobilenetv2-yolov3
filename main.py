from absl import app,flags
import tensorflow as tf
if hasattr(tf, 'enable_eager_execution'):
    tf.disable_eager_execution()
#tf.enable_v2_behavior()
#tf.enable_v2_tensorshape()
from yolo3.enum import BACKBONE,MODE,OPT
from train import train
from train_backbone import train as train_backbone
from yolo import YOLO, detect_video, detect_img,export_tflite_model,export_serving_model,calculate_map,export_tfjs_model
import os


FLAGS=flags.FLAGS

flags.DEFINE_enum_class('backbone',default=BACKBONE.EFFICIENTNET,enum_class=BACKBONE,help="Select network backbone, One of {'MOBILENETV2','DARKNET53','EFFICIENTNET'}")
flags.DEFINE_integer('batch_size',default=8,lower_bound=0,help="Train batch size")
flags.DEFINE_string('config',default=None,help="Config path")
flags.DEFINE_multi_integer('epochs',default=[10,10],lower_bound=0,help="Frozen train epochs and Full train epochs")
flags.DEFINE_string('export',default='export_model/7',help="Export path")
flags.DEFINE_string('input',default=None,help="Input data for various mode")
flags.DEFINE_multi_integer('input_size',default=(380,380),lower_bound=0,help="Input size")
flags.DEFINE_string('log_directory',default=None,help="Log directory")
flags.DEFINE_string('model',default='../download/mobilenetv2_trained_weights_final (5).h5',help="Model path")
flags.DEFINE_enum_class('mode',default=MODE.IMAGE,enum_class=MODE,help="Select exec mode, One of {'TRAIN','TRAIN_BACKBONE','IMAGE','VIDEO','TFLITE','SERVING','MAP','PRUNE'}")
flags.DEFINE_string('gpus',default='',help="Specific gpu indexes to run")
flags.DEFINE_string('train_dataset',default=None,help="Dataset glob for train")
flags.DEFINE_string('val_dataset',default=None,help="Dataset glob for validate")
flags.DEFINE_string('test_dataset',default='../pascal/VOCdevkit/test/*.tfrecords',help="Dataset glob for test")
flags.DEFINE_enum_class('opt',default=None,enum_class=OPT,help="Select optimization, One of {'XLA','DEBUG','MKL'}")
flags.DEFINE_bool('use_tpu',default=False,help="Whether use tpu")
flags.DEFINE_bool('prune',default=False,help="Whether prune model")

def main(_):
    flags_dict=FLAGS.flag_values_dict()
    if FLAGS.config is not None:
        import yaml
        with open(FLAGS.config) as stream:
            config=yaml.safe_load(stream)
            config['backbone']=BACKBONE[config['backbone']]
            flags_dict.update(config)

    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
    if FLAGS.mode==MODE.TRAIN:
        tf.logging.info('Train mode')
        train(flags_dict)
    elif FLAGS.mode==MODE.TRAIN_BACKBONE:
        tf.logging.info('Train backbone mode')
        train_backbone(flags_dict)
    elif FLAGS.mode==MODE.IMAGE:
        tf.logging.info('Image detection mode')
        detect_img(YOLO(flags_dict))
    elif FLAGS.mode==MODE.VIDEO:
        tf.logging.info('Video detection mode')
        detect_video(YOLO(flags_dict), FLAGS.input, FLAGS.output)
    elif FLAGS.mode==MODE.MAP:
        tf.logging.info('Calculate test dataset map')
        calculate_map(YOLO(flags_dict),FLAGS.test_dataset)
    elif FLAGS.mode==MODE.SERVING:
        tf.logging.info('Export hdf5 model to saved model')
        export_serving_model(YOLO(flags_dict),FLAGS.export)
    elif FLAGS.mode==MODE.TFLITE:
        tf.logging.info('Export hdf5 model to tflite model')
        export_tflite_model(YOLO(flags_dict),FLAGS.export)
    elif FLAGS.mode==MODE.TFJS:
        tf.logging.info('Export hdf5 model to tflite model')
        export_tfjs_model(YOLO(flags_dict),FLAGS.export)


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
