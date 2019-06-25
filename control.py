from tensorflow_serving.apis import get_model_status_pb2, model_service_pb2_grpc, model_management_pb2
import grpc
from absl import app,flags
from enum import unique,Enum

@unique
class MODE(Enum):
    STATUS=1
    CONFIG=2

FLAGS=flags.FLAGS
flags.DEFINE_enum_class("mode",default=MODE.STATUS,enum_class=MODE,help='exec mode')
flags.DEFINE_string("address",default="10.12.102.39:8500",help='grpc server address')

def main(_):
    channel = grpc.insecure_channel(FLAGS.address)

    stub = model_service_pb2_grpc.ModelServiceStub(channel)
    if MODE.STATUS==FLAGS.mode:
        request = get_model_status_pb2.GetModelStatusRequest()
        request.model_spec.name = 'detection'
        request.model_spec.signature_name = 'serving_default'
        result = stub.GetModelStatus(request)
    elif MODE.CONFIG==FLAGS.mode:
        request = model_management_pb2.ReloadConfigRequest()
        config = request.config.model_config_list.config.add()
        config.name = 'detection'
        config.base_path = '/models/detection/detection'
        config.model_platform = 'tensorflow'
        config2 = request.config.model_config_list.config.add()
        config2.name = 'pascal'
        config2.base_path = '/models/detection/pascal'
        config2.model_platform = 'tensorflow'
        result = stub.HandleReloadConfigRequest(request)

    print(result)


if __name__ == '__main__':
    app.run(main)
