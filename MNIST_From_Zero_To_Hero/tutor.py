from hashlib import md5
import tensorflow as tf
import numpy as np


class Tutor():

    def __init__(self, name):
        self.name = name
        print(f"Hi, I'm {name}. Good to see you!")

    def __speak(self, message):
        print(f"\033[34m[{self.name}]\033[0m " + message)

    def check_files(self, training_image_file, training_label_file,
                    test_image_file, test_label_file):

        self.training_image_file = training_image_file
        self.training_label_file = training_label_file
        self.test_image_file = test_image_file
        self.test_label_file = test_label_file

        train_image_hash = '6bbc9ace898e44ae57da46a324031adb'
        train_label_hash = 'a25bea736e30d166cdddb491f175f624'
        test_image_hash = '2646ac647ad5339dbf082846283269ea'
        test_label_hash = '27ae3e4e09519cfbb04c329615203637'

        def _checksum(filepath):
            with open(filepath, 'rb') as f:
                return md5(f.read()).hexdigest()

        if _checksum(training_image_file) == train_image_hash:
            self.__speak("训练用图像数据集文件OK!")
        else:
            self.__speak("训练用图像数据集文件异常，请核查。")

        if _checksum(training_label_file) == train_label_hash:
            self.__speak("训练用标签数据集文件OK!")
        else:
            self.__speak(("训练用标签数据集文件异常，请核查。"))

        if _checksum(test_image_file) == test_image_hash:
            self.__speak("测试用图像数据集文件OK!")
        else:
            self.__speak("测试用图像数据集文件异常，请核查！")

        if _checksum(test_label_file) == test_label_hash:
            self.__speak("测试用标签数据集文件OK!")
        else:
            self.__speak("测试用标签数据集文件异常，请核查！")

    def check_reading_functions(self, pixel_reading_func=None, label_reading_func=None):

        if pixel_reading_func:
            self.__pixel_reading_func = pixel_reading_func
            self.__speak("正在检查图像数据集读取函数..")

            magic_number, num_images, num_rows, num_cols, pixels = pixel_reading_func(
                self.test_image_file)

            results = []
            if magic_number != 2051:
                self.__speak("Magic number异常，请检查。")
            else:
                results.append(True)

            if num_images != 10000:
                self.__speak("图像数量异常，请检查。")
            else:
                results.append(True)

            if num_rows != 28:
                self.__speak("图像高度异常，请检查。")
            else:
                results.append(True)

            if num_cols != 28:
                self.__speak("图像宽度异常，请检查。")
            else:
                results.append(True)

            if len(pixels) != (num_images * num_cols * num_rows):
                self.__speak("像素数量与预期不符，请检查。")
            else:
                results.append(True)

            if all(results):
                self.__speak("图像数据集读取方法OK！")
        else:
            self.__speak("没有提供图像数据集读取函数，跳过。")

        if label_reading_func:
            self.__label_reading_func = label_reading_func
            self.__speak("正在检查标签数据集读取函数..")

            magic_number, num_items, labels = label_reading_func(
                self.test_label_file)

            results = []
            if magic_number != 2049:
                self.__speak("Magic number异常，请检查。")
            else:
                results.append(True)

            if num_items != 10000:
                self.__speak("标签数量异常，请检查。")
            else:
                results.append(True)

            if len(labels) != num_items:
                self.__speak("标签数量与预期不符，请检查。")
            else:
                results.append(True)

            if all(results):
                self.__speak("标签数据集读取方法OK！")
        else:
            self.__speak("没有提供标签数据集读取函数，跳过。")

    def check_model_function(self, build_model_func):

        model = build_model_func(128, 0.2)
        outputs = model(tf.expand_dims(tf.constant(tf.range(28*28)), axis=0))

        if outputs.shape.as_list() != [1, 10]:
            self.__speak("看起来模型的输出形状与预期不符，请检查。")
        else:
            self.__speak("看上去不错，给过！")

    def check_dataset_function(self, build_dataset_func):

        self.build_dataset_func = build_dataset_func

        _, _, _, _, pixels = self.__pixel_reading_func(self.test_image_file)
        _, _, labels = self.__label_reading_func(self.test_label_file)

        dataset = build_dataset_func(pixels, labels)

        p, l = iter(dataset).next()

        if p.shape.as_list() != [784] or l.shape.as_list() != []:
            self.__speak("数据集形状与预期不符呢！请检查。")
        else:
            self.__speak("看上去不错，加油！")

    def check_loss_func(self, loss_func):

        y = tf.constant(2)
        y_ = tf.constant([0.1, 0.2, 0.9, 0.15])

        l = tf.keras.losses.sparse_categorical_crossentropy(y, y_, True)
        l_ = loss_func(y, y_)

        if l != l_:
            self.__speak("损失函数看上去有点不正常。你确定吗？")
        else:
            self.__speak("损失函数看上去很棒，请继续！")

    def check_optimizer(self, optimizer):

        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            self.__speak("看起来你构建的并非是优化器，请检查。")
        else:
            self.__speak("优化器看上去正常。请继续。")

    def check_metrics(self, metrics):

        for metric in metrics:
            if not isinstance(metric, tf.keras.metrics.Metric):
                self.__speak("看起来你构建的并非是有效指标，请检查。")
            else:
                self.__speak("评价指标选择看上去不错！请继续。")

    def demo(self, model):
        _, _, _, _, pixels = self.__pixel_reading_func(self.test_image_file)
        _, _, labels = self.__label_reading_func(self.test_label_file)
        dataset = self.build_dataset_func(pixels, labels)
        p, l = iter(dataset).next()
        prob = model(tf.expand_dims(p, axis=0)).numpy().squeeze()
        result = np.argmax(prob)
        self.__speak("预测结果为：{:1d}".format(int(result)))

        return (p.numpy().reshape((28, 28)) * 255).astype(np.uint8)
