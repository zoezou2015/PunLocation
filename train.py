import time
import torch
import torch.optim as optim
import os
import sys
from models import LM_LSTM_CRF, ViterbiLoss
from utils import *
from torch.nn.utils.rnn import pack_padded_sequence
from datasets import WCDataset
from inference import ViterbiDecoder
from sklearn.metrics import f1_score
from config import Config
import copy
import argparse

# fix random seed
seed = 1234
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed=seed)
else:
    torch.manual_seed(seed=seed)
np.random.seed(seed=seed)

def main(fold, task, batch_size, epoch):
    """
    Training and validation.
    """

    config = Config(task, batch_size, epoch)
    print(config)

    global best_f1, epochs_since_improvement, checkpoint, start_epoch, word_map, char_map, tag_map
    global sentences, tags, pos_mask
    sentences, tags, pos_mask = load_sentences(config.path1_test, config.path1_gold, config.path2_test, config.path2_gold,
                                               config.use_all_instances, isDebug=config.debug)

    # Read training and validation data
    train_words, train_tags, train_pos_mask, \
    val_words, val_tags, val_pos_mask,\
    test_words, test_tags, test_pos_mask\
        = get_n_fold_splitting(sentences, tags, pos_mask, fold, fold_num)
    best_f1 = -0.1  # F1 score to start with

    # Initialize model or load checkpoint
    if config.checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        word_map = checkpoint['word_map']
        lm_vocab_size = checkpoint['lm_vocab_size']
        tag_map = checkpoint['tag_map']
        char_map = checkpoint['char_map']
        start_epoch = checkpoint['epoch'] + 1
        best_f1 = checkpoint['f1']
    else:
        word_map, char_map, tag_map, mask_map = create_maps(train_words + val_words, train_tags + val_tags, train_pos_mask+val_pos_mask,
                                                  config.min_word_freq, config.min_char_freq)  # create word, char, tag maps
        embeddings, word_map, lm_vocab_size = load_embeddings(config.emb_file, word_map,
                                                              config.expand_vocab)  # load pre-trained embeddings

        model = LM_LSTM_CRF(tagset_size=len(tag_map),
                            charset_size=len(char_map),
                            char_emb_dim=config.char_emb_dim,
                            char_rnn_dim=config.char_rnn_dim,
                            char_rnn_layers=config.char_rnn_layers,
                            vocab_size=len(word_map),
                            lm_vocab_size=lm_vocab_size,
                            word_emb_dim=config.word_emb_dim,
                            word_rnn_dim=config.word_rnn_dim,
                            word_rnn_layers=config.word_rnn_layers,
                            dropout=config.dropout,
                            config=config,
                            highway_layers=config.highway_layers).to(config.device)
        model.init_word_embeddings(embeddings.to(config.device))  # initialize embedding layer with pre-trained embeddings
        model.fine_tune_word_embeddings(config.fine_tune_word_embeddings)  # fine-tune
        if config.opt == "SGD":
            optimizer = optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr,
                                  momentum=config.momentum)
        elif config.opt == "Adam":
            optimizer = optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
        elif config.opt == "Adadelta":
            optimizer = optim.Adadelta(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
        elif config.opt == "Adagrad":
            optimizer = optim.Adagrad(params=filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

    # Loss functions
    lm_criterion = nn.CrossEntropyLoss().to(config.device)
    crf_criterion = ViterbiLoss(tag_map, config).to(config.device)
    # penalty_criterion = UniquePenalty(tag_map).to(config.device)

    # Since the language model's vocab is restricted to in-corpus indices, encode training/val with only these!
    # word_map might have been expanded, and in-corpus words eliminated due to low frequency might still be added because
    # they were in the pre-trained embeddings
    temp_word_map = {k: v for k, v in word_map.items() if v <= word_map['<unk>']}
    train_inputs = create_input_tensors(train_words, train_tags, temp_word_map, char_map,
                                        tag_map, train_pos_mask, mask_map)
    val_inputs = create_input_tensors(val_words, val_tags, temp_word_map, char_map, tag_map, val_pos_mask, mask_map)
    test_inputs = create_input_tensors(test_words, test_tags, temp_word_map, char_map, tag_map, test_pos_mask, mask_map)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(WCDataset(*train_inputs), batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.workers, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(WCDataset(*val_inputs), batch_size=config.batch_size, shuffle=True,
                                             num_workers=config.workers, pin_memory=False)
    test_loader = torch.utils.data.DataLoader(WCDataset(*test_inputs), batch_size=config.batch_size, shuffle=True,
                                             num_workers=config.workers, pin_memory=False)

    # Viterbi decoder (to find accuracy during validation)
    vb_decoder = ViterbiDecoder(tag_map)

    # Epochs
    for epoch in range(config.start_epoch, config.epochs):

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              lm_criterion=lm_criterion,
              crf_criterion=crf_criterion,
              optimizer=optimizer,
              epoch=epoch,
              vb_decoder=vb_decoder,
              tag_map=tag_map,
              config=config)

        # One epoch's validation
        # val_loader
        val_f1 = validate(val_loader=val_loader,
                                 model=model,
                                 crf_criterion=crf_criterion,
                                 vb_decoder=vb_decoder,
                                 tag_map=tag_map,
                                 config=config)

        # Did validation F1 score improve?
        is_best = val_f1 > best_f1
        best_f1 = max(val_f1, best_f1)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0
            best_model = copy.deepcopy(model)
        # Save checkpoint
        save_checkpoint(task, epoch, model, optimizer, val_f1, word_map, char_map, tag_map, lm_vocab_size, is_best, fold)

        # Decay learning rate every epoch
        adjust_learning_rate(optimizer, config.lr / (1 + (epoch + 1) * config.lr_decay))
    b_true_pos, b_false_pos, b_true_neg, b_false_neg, \
    b_loc_true_pos, b_loc_false_pos, b_loc_true_neg, \
    b_loc_false_neg, b_loc_multi_pos = eval(test_loader=test_loader,
                           model=best_model,
                           crf_criterion=crf_criterion,
                           vb_decoder=vb_decoder,
                           tag_map=tag_map, config=config)

    assert (b_true_neg+b_true_pos+b_false_pos+b_false_neg) == len(test_words)
    return b_true_pos, b_false_pos, b_true_neg, b_false_neg, b_loc_true_pos, b_loc_false_pos, b_loc_true_neg, b_loc_false_neg, b_loc_multi_pos


def train(train_loader, model, lm_criterion, crf_criterion, optimizer, epoch, vb_decoder, tag_map, config):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param lm_criterion: cross entropy loss layer
    :param crf_criterion: viterbi loss layer
    :param optimizer: optimizer
    :param epoch: epoch number
    :param vb_decoder: viterbi decoder (to decode and find F1 score)
    """

    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time per batch
    data_time = AverageMeter()  # data loading time per batch
    ce_losses = AverageMeter()  # cross entropy loss
    vb_losses = AverageMeter()  # viterbi loss
    p_losses = AverageMeter()  # penalty losses
    f1s = AverageMeter()  # f1 score
    loc_f1s = AverageMeter()

    start = time.time()

    # Batches
    for i, (wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths, pos_mask) in enumerate(
            train_loader):

        data_time.update(time.time() - start)

        max_word_len = max(wmap_lengths.tolist())
        max_char_len = max(cmap_lengths.tolist())

        # Reduce batch's padded length to maximum in-batch sequence
        # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
        wmaps = wmaps[:, :max_word_len].to(config.device)
        cmaps_f = cmaps_f[:, :max_char_len].to(config.device)
        cmaps_b = cmaps_b[:, :max_char_len].to(config.device)
        cmarkers_f = cmarkers_f[:, :max_word_len].to(config.device)
        cmarkers_b = cmarkers_b[:, :max_word_len].to(config.device)
        tmaps = tmaps[:, :max_word_len].to(config.device)
        wmap_lengths = wmap_lengths.to(config.device)
        cmap_lengths = cmap_lengths.to(config.device)
        pos_mask = pos_mask[:, :max_word_len].to(config.device)

        # Forward prop.
        crf_scores, lm_f_scores, lm_b_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, _, __ = model(cmaps_f,
                                                                                                             cmaps_b,
                                                                                                             cmarkers_f,
                                                                                                             cmarkers_b,
                                                                                                             wmaps,
                                                                                                             tmaps,
                                                                                                             wmap_lengths,
                                                                                                             cmap_lengths,
                                                                                                             pos_mask)

        # LM loss

        # We don't predict the next word at the pads or <end> tokens
        # We will only predict at [dunston, checks, in] among [dunston, checks, in, <end>, <pad>, <pad>, ...]
        # So, prediction lengths are word sequence lengths - 1
        lm_lengths = wmap_lengths_sorted - 1
        lm_lengths = lm_lengths.tolist()

        # Remove scores at timesteps we won't predict at
        # pack_padded_sequence is a good trick to do this (see dynamic_rnn.py, where we explore this)
        lm_f_scores, _ = pack_padded_sequence(lm_f_scores, lm_lengths, batch_first=True)
        lm_b_scores, _ = pack_padded_sequence(lm_b_scores, lm_lengths, batch_first=True)

        # For the forward sequence, targets are from the second word onwards, up to <end>
        # (timestep -> target) ...dunston -> checks, ...checks -> in, ...in -> <end>
        lm_f_targets = wmaps_sorted[:, 1:]
        lm_f_targets, _ = pack_padded_sequence(lm_f_targets, lm_lengths, batch_first=True)

        # For the backward sequence, targets are <end> followed by all words except the last word
        # ...notsnud -> <end>, ...skcehc -> dunston, ...ni -> checks
        lm_b_targets = torch.cat(
            [torch.LongTensor([word_map['<end>']] * wmaps_sorted.size(0)).unsqueeze(1).to(config.device), wmaps_sorted], dim=1)
        lm_b_targets, _ = pack_padded_sequence(lm_b_targets, lm_lengths, batch_first=True)

        # Calculate loss
        ce_loss = lm_criterion(lm_f_scores, lm_f_targets) + lm_criterion(lm_b_scores, lm_b_targets)
        vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted)

       loss = ce_loss + vb_loss

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        if config.grad_clip is not None:
            clip_gradient(optimizer, config.grad_clip)

        optimizer.step()

        # Viterbi decode to find accuracy / f1
        decoded = vb_decoder.decode(crf_scores.to("cpu"), wmap_lengths_sorted.to("cpu"))

        tmaps_sorted = tmaps_sorted % vb_decoder.tagset_size  # actual target indices (see create_input_tensors())

        f1, loc_f1, true_pos, false_pos, true_neg, false_neg, \
        loc_true_pos, loc_false_pos, loc_true_neg, \
        loc_false_neg, loc_multi_pos = calculate_f1_score(tmaps_sorted, decoded, tag_map)

        # Keep track of metrics
        ce_losses.update(ce_loss.item(), sum(lm_lengths))
        vb_losses.update(vb_loss.item(), crf_scores.size(0))
        # p_losses.update(p_loss.item(), crf_scores.size(0))
        batch_time.update(time.time() - start)
        f1s.update(f1, true_neg+true_pos+false_pos+false_neg)
        # loc_f1s.update(loc_f1, loc_true_neg+loc_true_pos+loc_false_pos+loc_false_neg)

        start = time.time()

        # Print training status
        if i % config.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'CE Loss {ce_loss.val:.4f} ({ce_loss.avg:.4f})\t'
                  'VB Loss {vb_loss.val:.4f} ({vb_loss.avg:.4f})\t'
                  'PE Loss {p_loss.val:.4f} ({p_loss.avg:.4f})\t'
                  'F1 {f1:.3f} ({f1:.3f})'.format(epoch, i, len(train_loader),
                                                          batch_time=batch_time,
                                                          data_time=data_time, ce_loss=ce_losses,
                                                          vb_loss=vb_losses, p_loss=p_losses,
                                                  f1=(1 - config.b_factor) * f1s.avg + config.b_factor * loc_f1s.avg))


def validate(val_loader, model, crf_criterion, vb_decoder, tag_map, config):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param crf_criterion: viterbi loss layer
    :param vb_decoder: viterbi decoder
    :return: validation F1 score
    """
    model.eval()

    batch_time = AverageMeter()
    vb_losses = AverageMeter()
    p_losses = AverageMeter()

    f1s = AverageMeter()
    loc_f1s = AverageMeter()

    # pun classification
    start = time.time()
    for i, (wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths, pos_mask) in enumerate(
            val_loader):

        max_word_len = max(wmap_lengths.tolist())
        max_char_len = max(cmap_lengths.tolist())

        # Reduce batch's padded length to maximum in-batch sequence
        # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
        wmaps = wmaps[:, :max_word_len].to(config.device)
        cmaps_f = cmaps_f[:, :max_char_len].to(config.device)
        cmaps_b = cmaps_b[:, :max_char_len].to(config.device)
        cmarkers_f = cmarkers_f[:, :max_word_len].to(config.device)
        cmarkers_b = cmarkers_b[:, :max_word_len].to(config.device)
        tmaps = tmaps[:, :max_word_len].to(config.device)
        pos_mask = pos_mask[:, :max_word_len].to(config.device)
        wmap_lengths = wmap_lengths.to(config.device)
        cmap_lengths = cmap_lengths.to(config.device)

        # Forward prop.
        crf_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, _, __ = model(cmaps_f,
                                                                                   cmaps_b,
                                                                                   cmarkers_f,
                                                                                   cmarkers_b,
                                                                                   wmaps,
                                                                                   tmaps,
                                                                                   wmap_lengths,
                                                                                   cmap_lengths,
                                                                                   pos_mask)

        # Viterbi / CRF layer loss
        vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted)

        # Viterbi decode to find accuracy / f1
        decoded = vb_decoder.decode(crf_scores.to("cpu"), wmap_lengths_sorted.to("cpu"))
        tmaps_sorted = tmaps_sorted % vb_decoder.tagset_size  # actual target indices (see create_input_tensors())

        f1, loc_f1, true_pos, false_pos, true_neg, \
        false_neg, loc_true_pos, loc_false_pos, \
        loc_true_neg, loc_false_neg, loc_multi_pos = calculate_f1_score(tmaps_sorted, decoded, tag_map)

        # Keep track of metrics
        vb_losses.update(vb_loss.item(), crf_scores.size(0))
        # p_losses.update(p_loss, crf_scores.size(0))
        f1s.update(f1, true_pos + false_pos + true_neg + false_neg)
        # loc_f1s.update(loc_f1, loc_true_pos + loc_true_neg + loc_false_neg + loc_false_pos)
        batch_time.update(time.time() - start)

        start = time.time()

        if i % config.print_freq == 0:
            print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'VB Loss {vb_loss.val:.4f} ({vb_loss.avg:.4f})\t'
                      'PE Loss {p_loss.val:.4f} ({p_loss.avg:.4f})\t'
                      'F1 Score {f1:.3f} ({f1:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                      vb_loss=vb_losses, p_loss=p_losses,
                                                              f1=(1 - config.b_factor) * f1s.avg + config.b_factor * loc_f1s.avg))

    print(
        '\n * LOSS - {vb_loss.avg:.3f}, F1 SCORE - {f1.avg:.3f}\n'.format(vb_loss=vb_losses,
                                                                          f1=f1s))
    # , true_pos, false_pos, true_neg, false_neg, loc_true_pos, loc_false_pos, loc_true_neg, loc_false_neg

    # if config.use_all_instances:
    #     return (1 - config.b_factor) * f1s.avg + config.b_factor * loc_f1s.avg
    # else:
    #     return loc_f1s.avg
    return (1 - config.b_factor) * f1s.avg + config.b_factor * loc_f1s.avg


def eval(test_loader, model, crf_criterion, vb_decoder, tag_map, config):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data
    :param model: model
    :param crf_criterion: viterbi loss layer
    :param vb_decoder: viterbi decoder
    :return: validation F1 score
    """
    model.eval()

    batch_time = AverageMeter()
    vb_losses = AverageMeter()
    f1s = AverageMeter()
    b_true_pos = 0
    b_false_pos = 0
    b_true_neg = 0
    b_false_neg = 0
    b_loc_true_pos = 0
    b_loc_false_pos = 0
    b_loc_true_neg = 0
    b_loc_false_neg = 0
    b_loc_multi_pos = 0

    # pun classification

    start = time.time()
    instances_len = 0
    instances_len = 0
    for i, (wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, wmap_lengths, cmap_lengths, pos_mask) in enumerate(
            test_loader):

        max_word_len = max(wmap_lengths.tolist())
        max_char_len = max(cmap_lengths.tolist())

        # Reduce batch's padded length to maximum in-batch sequence
        # This saves some compute on nn.Linear layers (RNNs are unaffected, since they don't compute over the pads)
        wmaps = wmaps[:, :max_word_len].to(config.device)
        cmaps_f = cmaps_f[:, :max_char_len].to(config.device)
        cmaps_b = cmaps_b[:, :max_char_len].to(config.device)
        cmarkers_f = cmarkers_f[:, :max_word_len].to(config.device)
        cmarkers_b = cmarkers_b[:, :max_word_len].to(config.device)
        tmaps = tmaps[:, :max_word_len].to(config.device)
        pos_mask = pos_mask[:, :max_word_len].to(config.device)
        wmap_lengths = wmap_lengths.to(config.device)
        cmap_lengths = cmap_lengths.to(config.device)

        # Forward prop.
        crf_scores, wmaps_sorted, tmaps_sorted, wmap_lengths_sorted, _, __ = model(cmaps_f,
                                                                                   cmaps_b,
                                                                                   cmarkers_f,
                                                                                   cmarkers_b,
                                                                                   wmaps,
                                                                                   tmaps,
                                                                                   wmap_lengths,
                                                                                   cmap_lengths,
                                                                                   pos_mask)

        # Viterbi / CRF layer loss
        vb_loss = crf_criterion(crf_scores, tmaps_sorted, wmap_lengths_sorted)

        # Viterbi decode to find accuracy / f1
        decoded = vb_decoder.decode(crf_scores.to("cpu"), wmap_lengths_sorted.to("cpu"))
        tmaps_sorted = tmaps_sorted % vb_decoder.tagset_size  # actual target indices (see create_input_tensors())
        print('pred\n', decoded)
        print('gold\n', tmaps_sorted)

        f1, _, true_pos, false_pos, true_neg, \
        false_neg, loc_true_pos, loc_false_pos, \
        loc_true_neg, loc_false_neg, loc_multi_pos = calculate_f1_score(tmaps_sorted, decoded, tag_map)

        # Keep track of metrics
        vb_losses.update(vb_loss.item(), crf_scores.size(0))
        f1s.update(f1, sum((wmap_lengths_sorted - 1).tolist()))
        batch_time.update(time.time() - start)
        b_true_pos += true_pos
        b_false_pos += false_pos
        b_true_neg += true_neg
        b_false_neg += false_neg
        b_loc_true_pos += loc_true_pos
        b_loc_false_pos += loc_false_pos
        b_loc_true_neg += loc_true_neg
        b_loc_false_neg += loc_false_neg
        b_loc_multi_pos += loc_multi_pos
        start = time.time()

        if i % config.print_freq == 0:
            print('Evaluation: [{0}/{1}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'VB Loss {vb_loss.val:.4f} ({vb_loss.avg:.4f})\t'
                  'F1 Score {f1.val:.3f} ({f1.avg:.3f})\t'.format(i, len(test_loader), batch_time=batch_time,
                                                                  vb_loss=vb_losses, f1=f1s))

    print(
        '\n * LOSS - {vb_loss.avg:.3f}, F1 SCORE - {f1.avg:.3f}\n'.format(vb_loss=vb_losses,
                                                                          f1=f1s))
    return b_true_pos, b_false_pos, b_true_neg, b_false_neg, b_loc_true_pos, b_loc_false_pos, b_loc_true_neg, b_loc_false_neg, b_loc_multi_pos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pun Location')
    # parser.add_argument('--lr', type=float, default=0.001,
    #                     help='initial learning rate [default: 0.001]')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs for train')
    parser.add_argument('--batch-size', type=int, default=15,
                        help='batch size for training [default: 16]')
    parser.add_argument('--task', type=str, default='homo',
                        help='specify dataset')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='the probability for dropout (0 = no dropout) [default: 0.5]')

    args = parser.parse_args()
    batch_size = args.batch_size
    epoch = args.epochs
    task = args.task

    fold_num = 10
    f1_score = 0.0
    b_true_pos = 0
    b_false_pos = 0
    b_true_neg = 0
    b_false_neg = 0
    b_loc_true_pos = 0
    b_loc_false_pos = 0
    b_loc_true_neg = 0
    b_loc_false_neg = 0
    b_loc_multi_pos = 0
    for f in range(fold_num):
        print('Working on {0} fold'.format(f))
        true_pos, false_pos, true_neg, false_neg, \
        loc_true_pos, loc_false_pos, loc_true_neg, \
        loc_false_neg, loc_multi_pos = main(f, task, batch_size, epoch)
        b_true_pos += true_pos
        b_false_pos += false_pos
        b_true_neg += true_neg
        b_false_neg += false_neg
        b_loc_true_pos += loc_true_pos
        b_loc_false_pos += loc_false_pos
        b_loc_true_neg += loc_true_neg
        b_loc_false_neg += loc_false_neg
        b_loc_multi_pos += loc_multi_pos

    total = b_true_pos + b_false_pos + b_true_neg + b_false_neg
    c_precision = b_true_pos * 1.0 / (b_true_pos + b_false_pos) * 100 if (b_true_pos + b_false_pos) > 0 else 0.0
    c_recall = b_true_pos * 1.0 / (b_true_pos + b_false_neg) * 100 if (b_true_pos + b_false_neg) > 0 else 0.0
    c_f1 = (c_precision * c_recall * 2) / (c_precision + c_recall) if (c_precision + c_recall) > 0 else 0.0
    c_accuracy = (b_true_pos + b_true_neg) * 1.0 / total * 100 if total > 0 else 0.0

    loc_total = b_loc_true_pos + b_loc_false_pos + b_loc_true_neg + b_loc_false_neg
    l_precision = b_loc_true_pos * 1.0 / (b_loc_true_pos + b_loc_false_pos) * 100 if (
                                                                                             b_loc_true_pos + b_loc_false_pos) > 0 else 0.0
    l_recall = b_loc_true_pos * 1.0 / loc_total * 100 if loc_total > 0 else 0.0
    l_f1 = l_precision * l_recall * 2.0 / (l_recall + l_precision) if (l_recall + l_precision) > 0 else 0.0
    l_accuracy = b_loc_true_pos * 1.0 / loc_total * 100 if loc_total > 0 else 0.0


    print('CV classification [{0}] instances\n'
          'Precision {prec:.3f}\t'
          'Recall {rec:.3f}\t'
          'F1 {f1:.3f}\t'
          'Acc {acc:.3f}\n'
          'CV location [{1}] pun instances\n'
          'Precision {l_prev:.3f}\t'
          'Recall {l_rec:.3f}\t'
          'F1 {l_f1:.3f}\t'
          'Acc {l_acc:.3f}\n'.format(total, loc_total, prec=c_precision, rec=c_recall, f1=c_f1, acc=c_accuracy,
                                     l_prev=l_precision,
                                     l_rec=l_recall, l_f1=l_f1, l_acc=l_accuracy))

    print('Multi position prediction: {0}'.format(b_loc_multi_pos))