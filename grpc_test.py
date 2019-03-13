# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Send JPEG image to tensorflow_model_server loaded with ResNet model.

"""

#from __future__ import print_function

# This is a placeholder for a Google-internal import.

import grpc
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from yolo3.utils import letterbox_image
# The image URL is the location of the image we should send to the server
IMAGE_URL = 'http://10.10.67.225:8090/files/newPics/Illegal_parking/2017-01/2017-01_%E2%94%A4%D0%B6%E2%94%94%D1%8D%E2%95%9F%E2%96%91/image3869.jpeg'


def main(_):

    image = Image.open('/media/fangsixie/data/keras-yolo3/logs/VOCdevkit/Illegal_parking/标注前/image396.jpeg')
    boxed_image=letterbox_image(image,(224,224))
    data = np.array(boxed_image, dtype='float32')
    data /= 255.
    data = np.expand_dims(data, 0)
    channel = grpc.insecure_channel('localhost:8500')
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    # Send request
    # See prediction_service.proto for gRPC request/response details.
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'test'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['input_1:0'].CopyFrom(
        tf.make_tensor_proto(data))
    request.inputs['Placeholder_366:0'].CopyFrom(
        tf.make_tensor_proto([image.size[1], image.size[0]],dtype=tf.float32))
    request.inputs['bn_Conv1/keras_learning_phase:0'].CopyFrom(
        tf.make_tensor_proto(0))

    result = stub.Predict(request, 10.0)  # 10 secs timeout
    print(result)


if __name__ == '__main__':
    tf.app.run()
