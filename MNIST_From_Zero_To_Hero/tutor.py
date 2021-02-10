from hashlib import md5


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
            self.__speak("正在检查图像数据集读取函数..")
            magic_number, num_images, num_rows, num_cols, pixels = pixel_reading_func(
                self.training_image_file)

            results = []
            if magic_number != 2051:
                self.__speak("Magic number异常，请检查。")
            else:
                results.append(True)

            if num_images != 60000:
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
            self.__speak("正在检查标签数据集读取函数..")
            magic_number, num_items, labels = label_reading_func(
                self.training_label_file)

            results = []
            if magic_number != 2049:
                self.__speak("Magic number异常，请检查。")
            else:
                results.append(True)

            if num_items != 60000:
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
