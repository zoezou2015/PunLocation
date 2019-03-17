from __future__ import division
import torch

class Config():
    def __init__(self, task, batch_size, epoch):

        self.debug = False
        self.BPA = True
        # Data parameters
        self.task = task  # tagging task, to choose column in CoNLL 2003 dataset
        self.path1_test = "datasets/subtask1-{0}graphic-test.xml".format(self.task)  # pun detection
        self.path1_gold = "datasets/subtask1-{0}graphic-test.gold".format(self.task)  # gold label for pun detection
        self.path2_test = "datasets/subtask2-{0}graphic-test.xml".format(self.task)  # pun location
        self.path2_gold = "datasets/subtask2-{0}graphic-test.gold".format(self.task)  # gold label for pun location
        self.emb_file = './embeddings/glove.6B.100d.txt' if not self.debug else './embeddings/glove.6B.100d.txt.orig'  # path to pre-trained word embeddings
        self.min_word_freq = 1  # 5  # threshold for word frequency
        self.min_char_freq = 1  # threshold for character frequency
        self.caseless = True  # lowercase everything?
        self.expand_vocab = True if not self.debug else False  # expand model's input vocabulary to the pre-trained embeddings' vocabulary?
        self.fold_num = 10  # the number of n-fold cross validation

        self.use_pos_mask = True  # True: include position binary feature
        self.use_all_instances = True  # True: all instances; False: only pun instances

        # Model parameters
        self.char_emb_dim = 30  # character embedding size
        with open(self.emb_file, 'r') as f:
            self.word_emb_dim = len(f.readline().split(' ')) - 1  # word embedding size
        self.word_rnn_dim = 300  # word RNN size
        self.char_rnn_dim = 300  # character RNN size
        self.char_rnn_layers = 1  # number of layers in character RNN
        self.word_rnn_layers = 1  # number of layers in word RNN
        self.highway_layers = 1  # number of layers in highway network
        self.dropout = 0.5  # dropout
        self.fine_tune_word_embeddings = False  # fine-tune pre-trained word embeddings?

        # Training parameters
        self.start_epoch = 0  # start at this epoch
        self.batch_size = batch_size
        self.lr = 0.015  # learning rate
        self.lr_decay = 0.05  # decay learning rate by this amount
        self.momentum = 0.9  # momentum
        self.workers = 1  # number of workers for loading data in the DataLoader
        self.epochs = epoch
        self.grad_clip = 5.  # clip gradients at this value
        self.print_freq = 100  # print training or validation status every __ batches
        self.checkpoint = None  # path to model checkpoint, None if none
        self.best_f1 = -0.1  # F1 score to start with
        self.tag_ind = 1 if self.task == 'pos' else 3  # choose column in CoNLL 2003 dataset
        self.b_factor = 0.0  # balance score between classification and location
        self.p_factor = 0.0  #

        self.opt = 'SGD'

        self.is_cuda = torch.cuda.is_available()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def __repr__(self):
        return str(vars(self))


# config = Config()
