import argparse
import tensorflow as tf
assert tf.__version__.startswith('2')
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader
from tensorflowjs.converters import convert_tf_saved_model

if __name__ == "__main__":
    str2bool = lambda str : False if str.lower() == 'false' else True
    parser = argparse.ArgumentParser(description = 'model_maker')
    parser.add_argument('--input', type = str, default = "./", help = "input dataset path. default: ./")
    parser.add_argument('--output', type = str, default = './', help = 'output model path. default: ./')
    parser.add_argument('--epochs', type = int, default = 30, help = 'More epochs could achieve better accuracy until it converges but training for too many epochs may lead to overfitting. 30 by default')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'Number of samples to use in one training step. 32 by default')
    parser.add_argument('--lr', type = float, default = 0.0005, help = 'Base learning rate. 0.0005 by default')
    parser.add_argument('--dropout', type = float, default = 0.3, help = 'The rate for dropout, avoid overfitting. 0.3 by default')
    parser.add_argument('--train_whole', type = str2bool, default = 'True', help = 'Boolean, if true, the Hub module is trained together with the classification layer on top. Otherwise, only train the top classification layer. True by default')
    parser.add_argument('--use_augmentation', type = str2bool, default = 'True', help = 'Boolean, use data augmentation for preprocessing. True by default.')
    parser.add_argument('--split_train', type = int, default = 0.8, help = 'Split train dataset ratio. 0.8 by default')
    parser.add_argument('--split_valid', type = int, default = 0.5, help = 'Split valid dataset ratio. 0.5 by default')
    parser.add_argument('--shuffle', type = str2bool, default = 'True', help = 'Boolean, whether the data should be shuffled. True by default.')
    # 导入数据集
    data = DataLoader.from_folder(parser.parse_args().input)
    # 数据集划分 训练集80% 测试集10% 验证集10%
    train_data, rest_data = data.split(parser.parse_args().split_train)
    validation_data, test_data = rest_data.split(parser.parse_args().split_valid)
    # 创建模型
    model = image_classifier.create(
        train_data, 
        model_spec = image_classifier.ModelSpec(
            uri = "https://hub.tensorflow.google.cn/google/imagenet/mobilenet_v2_050_224/feature_vector/5",
            input_image_shape = [224, 224]
        ), 
        validation_data = validation_data,
        epochs = parser.parse_args().epochs,
        batch_size = parser.parse_args().batch_size,
        learning_rate = parser.parse_args().lr,
        dropout_rate = parser.parse_args().dropout,
        train_whole_model = parser.parse_args().train_whole,
        use_augmentation = parser.parse_args().use_augmentation,
        shuffle = parser.parse_args().shuffle,
    )
    # 模型评估
    _, accuracy = model.evaluate(test_data)
    print("测试集精度: {}".format(accuracy))
    # 模型量化导出
    model.export(
        with_metadata = False,
        export_dir = parser.parse_args().output, 
        export_format = [ExportFormat.LABEL, ExportFormat.SAVED_MODEL, ExportFormat.TFLITE],
        quantization_config = QuantizationConfig(
            optimizations = tf.lite.Optimize.DEFAULT,
            representative_data = validation_data,
            supported_ops = tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            inference_input_type = tf.int8, 
            inference_output_type = tf.int8)
    )
    # 模型量化精度
    print("测试集量化精度: {}".format(model.evaluate_tflite('model.tflite', data = test_data)['accuracy']))
    # 模型转换tfjs格式
    convert_tf_saved_model("saved_model", "tfjs_model")