import numpy
import scipy.special


class NeuralNetwork:
    def __init__(self, in_nodes, hidden_nodes, out_nodes, learning_grate, activation_function=scipy.special.expit):
        '''
        :param in_nodes: Количество входных вершин
        :param hidden_nodes: Количество скрытых вершин
        :param out_nodes: Количество выходных вершин
        :param learning_grate: Коэффициент обучения
        :param activation_function: Функция активации нейрона ( по умолчанию - сигмоида )
        '''
        self.inodes = in_nodes
        self.hnodes = hidden_nodes
        self.onodes = out_nodes
        self.lgrate = learning_grate
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.af = activation_function

    def train(self, inputs_list, targets_list):
        # Прогоняем входные данные через сеть
        inputs = numpy.array(inputs_list, dtype=float, ndmin=2).T
        # Считаем вход для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Применяем функцию активации
        hidden_outputs = self.af(hidden_inputs)
        # Аналогично для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.af(final_inputs)

        # Транспонируем лист с ответами
        targets = numpy.array(targets_list, dtype=float, ndim=2).T

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lgrate * numpy.dot(
            (output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lgrate * numpy.dot(
            (hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

    def query(self, inputs_list):
        # Транспонируем исходную матрицу
        inputs = numpy.array(inputs_list, dtype=float, ndmin=2).T
        # Считаем вход для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Применяем функцию активации
        hidden_outputs = self.af(hidden_inputs)
        # Аналогично для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.af(final_inputs)
        return final_outputs


if __name__ == '__main__':
    n = NeuralNetwork(3, 3, 3, 0.5)
    print(numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ndmin=2).T)
