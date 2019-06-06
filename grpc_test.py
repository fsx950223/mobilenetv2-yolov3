from PIL import Image, ImageFont, ImageDraw
import numpy as np
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import threading
import os
import time
from grpc._cython.cygrpc import CompressionAlgorithm
from grpc._cython.cygrpc import CompressionLevel
import colorsys
# The image URL is the location of the image we should send to the server

SIZE=224
def main(_):
    names = os.listdir(
        '/media/fangsixie/data/keras-yolo3/logs/VOCdevkit/Illegal_parking/标注前/'
    )[:10]
    classes_path = os.path.expanduser('model_data/cci.names')
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    hsv_tuples = [
        (x / len(class_names), 1., 1.) for x in range(len(class_names))
    ]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    results = []

    def make_predict(request):
        result = stub.Predict(request, 100000.0)  # 10 secs timeout
        #end = int(round(time.time() * 1000))
        #print(end - start)
        #results.append(result)
        out_classes = np.array(result.outputs['yolo/classes:0'].int_val)
        out_scores = np.array(result.outputs['yolo/scores:0'].float_val)
        out_boxes = np.array(result.outputs['yolo/boxes:0'].int_val).reshape(-1,4)
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = class_names[c]
            box = out_boxes[i]
            score = out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            top, left, bottom, right = box
            print(label, (left, top), (right, bottom))
        # font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
        #                           size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # thickness = (image.size[1] + image.size[0]) // 300
        #
        # for i, c in reversed(list(enumerate(out_classes))):
        #     predicted_class = class_names[c]
        #     box = out_boxes[i]
        #     score = out_scores[i]
        #
        #     label = '{} {:.2f}'.format(predicted_class, score)
        #
        #     draw = ImageDraw.Draw(image)
        #     label_size = draw.textsize(label, font)
        #
        #     top, left, bottom, right = box
        #     print(label, (left, top), (right, bottom))
        #
        #     if top - label_size[1] >= 0:
        #         text_origin = np.array([left, top - label_size[1]])
        #     else:
        #         text_origin = np.array([left, top + 1])
        #
        #     # My kingdom for a good redistributable image drawing library.
        #     for i in range(thickness):
        #         draw.rectangle(
        #             [left + i, top + i, right - i, bottom - i],
        #             outline=colors[c])
        #     draw.rectangle(
        #         [tuple(text_origin), tuple(text_origin + label_size)],
        #         fill=colors[c])
        #     draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        #     del draw
        #
        # image.show()

    threads = []
    start = int(round(time.time() * 1000))
    options = [
        ('grpc.max_receive_message_length', -1),
        ('grpc.default_compression_algorithm', CompressionAlgorithm.gzip),
        ('grpc.default_compression_level', CompressionLevel.high)
    ]
    #channel = grpc.insecure_channel('10.12.102.32:8000', options)
    channel = grpc.insecure_channel('localhost:8500',options)

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    for i in range(len(names)):
        image = Image.open(
            '/media/fangsixie/data/keras-yolo3/logs/VOCdevkit/Illegal_parking/标注前/'
            + names[i])
        scale = SIZE / max(image.size)
        if scale < 1:
            image = image.resize((int(line * scale) for line in image.size),
                                 Image.BILINEAR)
        image_data = np.array(image, dtype='uint8')
        image_data = np.expand_dims(image_data, 0)


        # Send request
        # See prediction_service.proto for gRPC request/response details.
        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'detection'
        #request.model_spec.name = 'default'
        request.model_spec.signature_name = 'serving_default'
        request.inputs['predict_image:0'].CopyFrom(
            tf.make_tensor_proto(image_data))
        thread = threading.Thread(target=make_predict, args=[request])
        threads.append(thread)
    val1=(int(round(time.time() * 1000)) - start)
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    end = int(round(time.time() * 1000))
    val2 =end - start
    print(val2-val1)


if __name__ == '__main__':
    main(None)

