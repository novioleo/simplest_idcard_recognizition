from __future__ import print_function
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
from mxnet.gluon.data import Dataset
import Augmentor


mx.random.seed(1)
ctx = mx.gpu()
batch_size = 64
num_outputs = 3819


class chinese_character(Dataset):
    def __init__(self, pic_path, size):
        self.p = Augmentor.Pipeline(pic_path, '../chinese_character_pics_output')
        self.p.rotate(probability=0.1, max_left_rotation=8, max_right_rotation=8)
        self.p.random_distortion(probability=0.1, grid_width=2, grid_height=2, magnitude=3)
        # self.p.skew(0.3)
        # self.p.skew_top_bottom(0.3)
        # self.p.skew_tilt(0.3)
        # self.p.skew_corner(0.3)
        # self.p.skew_left_right(0.3)
        self.ig = self.p.image_generator()
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        m_img, m_label = next(self.ig)
        return nd.array(m_img.convert('L')).reshape((1, 28, 28)) / 255, np.array([m_label, ], np.float32)

    def close(self):
        self.ig.close()


def get_net():
    net = gluon.nn.Sequential()
    with net.name_scope():
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=0, strides=1))
        net.add(gluon.nn.LeakyReLU(0.3))
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=0, strides=1))
        net.add(gluon.nn.LeakyReLU(0.3))

        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=0, strides=1))
        net.add(gluon.nn.LeakyReLU(0.3))

        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dropout(0.3))
        net.add(gluon.nn.Dense(num_outputs))
    return net


train_gen = chinese_character('../../chinese_character_pics', 10000)
test_gen = chinese_character('../../chinese_character_pics', 100)
# generate chinese.txt for predict
# with open('../chinese.txt','w') as to_write:
#     for m_label in train_gen.p.class_labels:
#         to_write.write(m_label[0]+'\n')

train_data = mx.gluon.data.DataLoader(train_gen, batch_size)
test_data = mx.gluon.data.DataLoader(test_gen, batch_size)

net = get_net()

net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001})


def evaluate_accuracy(data_iterator, net):
    acc = mx.metric.Accuracy()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=label)
    return acc.get()[1]


epochs = 20
smoothing_constant = .01
# net.load_params('./train_model/chinese_19.para', ctx)
for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        ##########################
        #  Keep a moving average of the losses
        ##########################
        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (e == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, test_accuracy))
    net.save_params('./train_model/chinese_%d.para' % e)

train_gen.close()
test_gen.close()
