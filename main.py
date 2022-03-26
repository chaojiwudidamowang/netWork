import numpy
import scipy.special
import matplotlib.pyplot

class neuralNetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 输入层节点
        self.innodes = inputnodes
        # 隐藏层节点
        self.hnodes = hiddennodes
        # 输出层节点
        self.onodes = outputnodes
        # 输入层和隐藏层之间的链接权重矩阵 大小为 隐藏层节点乘以输入层节点
        # 正态分布中心0.0，节点数目的-0.5次方 数组形状的大小
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.innodes))
        # 隐藏层和输出层之间的权重 大小为输出层节点乘以隐藏层节点
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # 学习率
        self.lr = learningrate
        # 激活函数 用匿名表达式创建函数
        self.activation_function = lambda x: scipy.special.expit(x)
        pass
    # 训练函数
    def train(self, input_list, targets_list):
        # 将input_list变为数组 ，最小维度为2，T表示矩阵转置
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # 数组积或矩阵积
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 激活函数
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets-final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)), numpy.transpose(inputs))
        pass

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs




if __name__ == '__main__':
    print("hello")
    # 读文件 将文件的数据分行读出来 一行是一个图片文件的数据
    data_file = open("C:\\Users\\Administrator\\Desktop\\newcode\\network\\测试数据\\mnist_train.csv", 'r')
    # 将所有图片信息读到一个链表里面
    data_list = data_file.readlines()
    data_file.close()
    print(len(data_list))
    print(data_list[0])

    # 将数据绘成图
    all_values = data_list[9].split(',')
    image_array = numpy.asfarray(all_values[1:]).reshape((28,28))
    matplotlib.pyplot.imshow(image_array, cmap='Greys',interpolation='None')

    # 将数据缩放到0-1
    scaled_inpus = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
    print(scaled_inpus)

    # 构建目标矩阵
    onodes = 10
    targets = numpy.zeros(onodes)+0.01
    targets[int(all_values[0])] = 0.99

    # 开始准备数据测试
    input_nodes = 784  # 输入层784个节点
    hidden_nodes = 100  # 隐藏层100个节点
    output_nodes = 10  # 输出层10个节点
    learning_rate = 0.1

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # 将文件逐个带进模型训练
    for record in data_list:
        train_all_values = record.split(',')
        inputs = (numpy.asfarray(train_all_values[1:])/255.0*0.99)+0.01
        targets = numpy.zeros(output_nodes)+0.01
        targets[int(train_all_values[0])] = 0.99
        n.train(inputs, targets)
        pass

    ceshi_values = data_list[9].split(',')
    print(n.query((numpy.asfarray(ceshi_values[1:])/255.0*0.99)+0.01))

    scorecard = []

    for record in data_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        print(correct_label, "correct label")
        inputs = (numpy.asfarray(all_values[1:])/255.0*0.99)+0.01
        outputs = n.query(inputs)
        label = numpy.argmax(outputs)
        print(label, "network answer")
        if(label == correct_label):
            scorecard.append(1)
        else:
            scorecard.append(0)
            pass
        pass

    scorecard_array = numpy.asarray(scorecard)
    print("performance =", scorecard_array.sum()/scorecard_array.size)

    #显示图片 显示的内容为其中一张图片 数字4或者9
    matplotlib.pyplot.show()