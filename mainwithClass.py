import argparse
import pickle
import time
from utils import build_graph, Data, split_validation
from model import *

class DefaultParams:
    def __init__(self):
        self.dataset = 'diginetica'
        self.batchSize = 100
        self.hiddenSize = 100
        self.epoch = 30
        self.lr = 0.001
        self.lr_dc = 0.1
        self.lr_dc_step = 3
        self.l2 = 0.00001
        self.step = 1
        self.patience = 10
        self.nonhybrid = True
        self.validation = True
        self.valid_portion = 0.1

opt = DefaultParams()
def main():
    train_data = pickle.load(open(opt.dataset+'/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open(opt.dataset + '/test.txt', 'rb'))
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    n_node = 43098
    

    model = trans_to_cuda(SessionGraph(opt, n_node))

    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr = train_test(model, train_data, test_data)
        flag = 0
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
        print('Best Result:')
        print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f\tEpoch:\t%d,\t%d'% (best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
