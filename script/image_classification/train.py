import gzip
import argparse

import paddle.v2.dataset.flowers as flowers
import paddle.v2 as paddle
import reader
import vgg
import resnet
import alexnet
import googlenet
import inception_v4
import inception_resnet_v2
import xception
from visualdl import LogWriter
import time

DATA_DIM = 3 * 224 * 224  # Use 3 * 331 * 331 or 3 * 299 * 299 for Inception-ResNet-v2.
CLASS_DIM = 3
BATCH_SIZE = 96
logdir = "./tmp"
logwriter = LogWriter(logdir, sync_cycle=10)
finetuning_layers = ['___fc_layer_0__']
with logwriter.mode("train") as writer:
    loss_scalar = writer.scalar("loss")

with logwriter.mode("train") as writer:
    acc_scalar = writer.scalar("acc")

num_samples = 4
with logwriter.mode("train") as writer:
    conv_image = writer.image("conv_image", num_samples, 1)
    input_image = writer.image("input_image", num_samples, 1)

with logwriter.mode("train") as writer:
    param1_histgram = writer.histogram("param1", 100)
step = 0
start = time.time()
def main():
    # parse the argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m','--model',
        help='The model for image classification',
        choices=[
            'alexnet', 'vgg13', 'vgg16', 'vgg19', 'resnet', 'googlenet',
            'inception-resnet-v2', 'inception_v4', 'xception'
        ])
    parser.add_argument('-r','--retrain_file', type=str,default='', help="The model file to retrain, none is for train from scratch")
    args = parser.parse_args()

    # PaddlePaddle init
    paddle.init(use_gpu=True, trainer_count=1)

    image = paddle.layer.data(
        name="image", type=paddle.data_type.dense_vector(DATA_DIM))
    lbl = paddle.layer.data(
        name="label", type=paddle.data_type.integer_value(CLASS_DIM))

    extra_layers = None
    learning_rate = 0.0001
    if args.model == 'alexnet':
        out = alexnet.alexnet(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg13':
        out = vgg.vgg13(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg16':
        out = vgg.vgg16(image, class_dim=CLASS_DIM)
    elif args.model == 'vgg19':
        out = vgg.vgg19(image, class_dim=CLASS_DIM)
    elif args.model == 'resnet':
        conv, pool, out = resnet.resnet_imagenet(image, class_dim=CLASS_DIM)
        learning_rate = 0.1
    elif args.model == 'googlenet':
        out, out1, out2 = googlenet.googlenet(image, class_dim=CLASS_DIM)
        loss1 = paddle.layer.cross_entropy_cost(
            input=out1, label=lbl, coeff=0.3)
        paddle.evaluator.classification_error(input=out1, label=lbl)
        loss2 = paddle.layer.cross_entropy_cost(
            input=out2, label=lbl, coeff=0.3)
        paddle.evaluator.classification_error(input=out2, label=lbl)
        extra_layers = [loss1, loss2]
    elif args.model == 'inception-resnet-v2':
        assert DATA_DIM == 3 * 331 * 331 or DATA_DIM == 3 * 299 * 299
        out = inception_resnet_v2.inception_resnet_v2(
            image, class_dim=CLASS_DIM, dropout_rate=0.5, data_dim=DATA_DIM)
    elif args.model == 'inception_v4':
        conv, pool, out = inception_v4.inception_v4(image, class_dim=CLASS_DIM)
    elif args.model == 'xception':
        out = xception.xception(image, class_dim=CLASS_DIM)

    cost = paddle.layer.classification_cost(input=out, label=lbl)

    # Create parameters
    parameters = paddle.parameters.create(cost)
    for k,v in parameters.__param_conf__.items():
        print(" config key {0}\t\t\tval{1}".format(k,v))
    print("-"*50)
    #print(parameters.__param_conf__[0])
   
    if args.retrain_file is not None and ''!=args.retrain_file:
        print("restore parameters from {0}".format(args.retrain_file))
        exclude_params = [param for param in parameters.names() if param.startswith('___fc_layer_0__')]
        parameters.init_from_tar(gzip.open(args.retrain_file), exclude_params)

    # Create optimizer
    optimizer = paddle.optimizer.Momentum(
        momentum=0.9,
        regularization=paddle.optimizer.L2Regularization(rate=0.0005 *
                                                         BATCH_SIZE),
        learning_rate=learning_rate / BATCH_SIZE,
        learning_rate_decay_a=0.1,
        learning_rate_decay_b=128000 * 35,
        learning_rate_schedule="discexp", )

    train_reader = paddle.batch(
        paddle.reader.shuffle(
            # flowers.train(),
            # To use other data, replace the above line with:
            reader.train_reader('valid_train0.lst'),
            buf_size=2048),
        batch_size=BATCH_SIZE)
    test_reader = paddle.batch(
        # flowers.valid(),
        # To use other data, replace the above line with:
        reader.test_reader('valid_val.lst'),
        batch_size=BATCH_SIZE)

    # Create trainer
    trainer = paddle.trainer.SGD(cost=cost,
                                 parameters=parameters,
                                 update_equation=optimizer,
                                 extra_layers=extra_layers)

    # End batch and end pass event handler
    def event_handler(event):
        global step
        global start
        if isinstance(event, paddle.event.EndIteration):
            if event.batch_id % 10== 0:
                print "\nPass %d, Batch %d, Cost %f, %s, %s" % (
                    event.pass_id, event.batch_id, event.cost, event.metrics, time.time()-start)
                start = time.time()
                loss_scalar.add_record(step, event.cost)
                acc_scalar.add_record(step, 1-event.metrics['classification_error_evaluator'])
                start = time.time()
                step+=1
            if event.batch_id % 100 == 0:
                with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                    trainer.save_parameter_to_tar(f)
                
        if isinstance(event, paddle.event.EndPass):
            with gzip.open('params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                trainer.save_parameter_to_tar(f)
            result = trainer.test(reader=test_reader)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)
    trainer.train(
        reader=train_reader, num_passes=200, event_handler=event_handler)


if __name__ == '__main__':
    main()
