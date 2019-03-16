from collections import Counter
import codecs
import itertools
from functools import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence
import xml.etree.ElementTree as ET
import math
import csv
import os
import errno


def load_sentences(path1_test, path1_gold, path2_test, path2_gold, use_all_instances=True, isDebug=False):
    """
    Load sentences.

    """
    all_instances = {}
    pun_instances = {}
    classes = {}
    locations = {}
    sentences = []
    tags = []
    tree1 = ET.parse(path1_test)
    root1 = tree1.getroot()
    for child in root1:
        idx = child.attrib["id"]
        line = []
        for kid in child:
            line.append(kid.text)
            all_instances[idx] = line

    with open(path1_gold) as gold1:
        lines = gold1.readlines()
        for line in lines:
            token = line.strip().split("\t")
            classes[token[0]] = token[1]

    tree2 = ET.parse(path2_test)
    root2 = tree2.getroot()
    pun_classes = {}
    for child in root2:
        line = []
        idx = child.attrib["id"]
        for kid in child:
            line.append(kid.text)
        pun_instances[idx] = line

    with open(path2_gold) as gold2:
        lines = gold2.readlines()
        for line in lines:
            token = line.strip().split("\t")
            sub_tokens = token[1].split("_")
            locations[token[0]] = sub_tokens[2]
            pun_classes[token[0]] = '1'

    if use_all_instances:
        sentences, tags, pos_mask = map_sentence_to_tag_sequence(all_instances, classes, locations)
    else:
        sentences, tags, pos_mask = map_sentence_to_tag_sequence(pun_instances, pun_classes, locations)
    if isDebug:
        return sentences[:10], tags[:10], pos_mask[:10]
    return sentences, tags, pos_mask


def map_sentence_to_tag_sequence(all_instances, classes, locations):
    sentences = []
    tags = []
    pos_mask = []

    for idx in all_instances.keys():
        sentence = all_instances[idx]
        tag = [0 for s in sentence]
        pos = [0 if s < len(sentence) // 2 else 1 for s in range(len(sentence))]
        # this is a pun
        if int(classes[idx]) == 1:
            pun_loc = int(locations[idx])
            for i in range(len(tag)):
                if i < pun_loc - 1:
                    tag[i] = 0
                elif i == pun_loc - 1:
                    tag[i] = 1
                else:
                    tag[i] = 2  # 2
        sentences.append(sentence)
        tags.append(tag)
        pos_mask.append(pos)
    return sentences, tags, pos_mask


def get_n_fold_splitting(sentences, tags, pos_mask, fold, fold_num=10):
    chunk_len = math.ceil(len(sentences)/fold_num)
    sent_chunks = []
    tag_chunks = []
    pos_mask_chunks = []
    for i in range(0, len(sentences), chunk_len):
        if i == fold_num-1:
            sent_chunks.append(sentences[i:])
            tag_chunks.append(tags[i:])
            pos_mask_chunks.append(pos_mask[i:])
        else:
            sent_chunks.append(sentences[i:i+chunk_len])
            tag_chunks.append(tags[i:i+chunk_len])
            pos_mask_chunks.append(pos_mask[i:i+chunk_len])

    dev_sents, dev_tags, dev_pos_mask = [], [], []
    for i in range(fold_num):
        if i != fold:
            dev_sents += sent_chunks[i]
            dev_tags += tag_chunks[i]
            dev_pos_mask += pos_mask_chunks[i]
    val_sents, val_tags, val_pos_mask = dev_sents[int(len(dev_sents)*0.8):], dev_tags[int(len(dev_tags)*0.8):], dev_pos_mask[int(len(dev_pos_mask)*0.8):]
    train_sents, train_tags, train_pos_mask = dev_sents[:int(len(dev_sents) * 0.8)], dev_tags[:int(len(dev_tags) * 0.8)], dev_pos_mask[:int(len(dev_pos_mask) * 0.8)]
    test_sents, test_tags, test_pos_mask = sent_chunks[fold], tag_chunks[fold], pos_mask_chunks[fold]
    # print(len(train_sents), len(val_sents), len(test_sents))
    assert len(train_sents) + len(val_sents) + len(test_sents) == len(sentences)
    assert len(train_tags) + len(val_tags) + len(test_tags) == len(tags)
    assert len(train_pos_mask) + len(val_pos_mask) + len(test_pos_mask) == len(pos_mask)

    return train_sents, train_tags, train_pos_mask, val_sents, val_tags, \
           val_pos_mask, test_sents, test_tags, test_pos_mask


def read_words_tags(file, tag_ind, caseless=True):
    """
    Reads raw data in the CoNLL 2003 format and returns word and tag sequences.

    :param file: file with raw data in the CoNLL 2003 format
    :param tag_ind: column index of tag
    :param caseless: lowercase words?
    :return: word, tag sequences
    """
    with codecs.open(file, 'r', 'utf-8') as f:
        lines = f.readlines()
    words = []
    tags = []
    temp_w = []
    temp_t = []
    pos_mask = []
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            feats = line.rstrip('\n').split()
            temp_w.append(feats[0].lower() if caseless else feats[0])
            temp_t.append(feats[tag_ind])
        elif len(temp_w) > 0:
            assert len(temp_w) == len(temp_t)
            words.append(temp_w)
            tags.append(temp_t)
            temp_w = []
            temp_t = []


    # last sentence
    if len(temp_w) > 0:
        assert len(temp_w) == len(temp_t)
        words.append(temp_w)
        tags.append(temp_t)

    # Sanity check
    assert len(words) == len(tags)

    return words, tags


def create_maps(words, tags, pos_masks, min_word_freq=5, min_char_freq=1):
    """
    Creates word, char, tag maps.

    :param words: word sequences
    :param tags: tag sequences
    :param min_word_freq: words that occur fewer times than this threshold are binned as <unk>s
    :param min_char_freq: characters that occur fewer times than this threshold are binned as <unk>s
    :return: word, char, tag maps
    """
    word_freq = Counter()
    char_freq = Counter()
    tag_map = set()
    mask_map = set()
    for w, t, p in zip(words, tags, pos_masks):
        word_freq.update(w)
        char_freq.update(list(reduce(lambda x, y: list(x) + [' '] + list(y), w)))
        tag_map.update(t)
        mask_map.update(p)

    word_map = {k: v + 1 for v, k in enumerate([w for w in word_freq.keys() if word_freq[w] > min_word_freq])}
    char_map = {k: v + 1 for v, k in enumerate([c for c in char_freq.keys() if char_freq[c] > min_char_freq])}
    tag_map = {k: v + 1 for v, k in enumerate(tag_map)}
    mask_map = {k: v + 1 for v, k in enumerate(mask_map)}

    word_map['<pad>'] = 0
    word_map['<end>'] = len(word_map)
    word_map['<unk>'] = len(word_map)
    char_map['<pad>'] = 0
    char_map['<end>'] = len(char_map)
    char_map['<unk>'] = len(char_map)
    tag_map['<pad>'] = 0
    tag_map['<start>'] = len(tag_map)
    tag_map['<end>'] = len(tag_map)
    mask_map['<pad>'] = 0
    mask_map['<end>'] = len(mask_map)

    print(mask_map)
    return word_map, char_map, tag_map, mask_map


def create_input_tensors(words, tags, word_map, char_map, tag_map, pos_mask, mask_map):
    """
    Creates input tensors that will be used to create a PyTorch Dataset.

    :param words: word sequences
    :param tags: tag sequences
    :param word_map: word map
    :param char_map: character map
    :param tag_map: tag map
    :return: padded encoded words, padded encoded forward chars, padded encoded backward chars,
            padded forward character markers, padded backward character markers, padded encoded tags,
            word sequence lengths, char sequence lengths
    """
    # Encode sentences into word maps with <end> at the end
    # [['dunston', 'checks', 'in', '<end>']] -> [[4670, 4670, 185, 4669]]
    wmaps = list(map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [word_map['<end>']], words))

    # Forward and backward character streams
    # [['d', 'u', 'n', 's', 't', 'o', 'n', ' ', 'c', 'h', 'e', 'c', 'k', 's', ' ', 'i', 'n', ' ']]
    chars_f = list(map(lambda s: list(reduce(lambda x, y: list(x) + [' '] + list(y), s)) + [' '], words))

    # [['n', 'i', ' ', 's', 'k', 'c', 'e', 'h', 'c', ' ', 'n', 'o', 't', 's', 'n', 'u', 'd', ' ']]
    chars_b = list(
        map(lambda s: list(reversed([' '] + list(reduce(lambda x, y: list(x) + [' '] + list(y), s)))), words))

    # Encode streams into forward and backward character maps with <end> at the end
    # [[29, 2, 12, 8, 7, 14, 12, 3, 6, 18, 1, 6, 21, 8, 3, 17, 12, 3, 60]]
    cmaps_f = list(
        map(lambda s: list(map(lambda c: char_map.get(c, char_map['<unk>']), s)) + [char_map['<end>']], chars_f))
    # [[12, 17, 3, 8, 21, 6, 1, 18, 6, 3, 12, 14, 7, 8, 12, 2, 29, 3, 60]]
    cmaps_b = list(
        map(lambda s: list(map(lambda c: char_map.get(c, char_map['<unk>']), s)) + [char_map['<end>']], chars_b))

    # Positions of spaces and <end> character
    # Words are predicted or encoded at these places in the language and tagging model respectively
    # [[7, 14, 17, 18]] are points after '...dunston', '...checks', '...in', '...<end>' respectively
    cmarkers_f = list(map(lambda s: [ind for ind in range(len(s)) if s[ind] == char_map[' ']] + [len(s) - 1], cmaps_f))
    # Reverse the markers for the backward stream before adding <end>, so the words of the f and b markers coincide
    # i.e., [[17, 9, 2, 18]] are points after '...notsnud', '...skcehc', '...ni', '...<end>' respectively
    cmarkers_b = list(
        map(lambda s: list(reversed([ind for ind in range(len(s)) if s[ind] == char_map[' ']])) + [len(s) - 1],
            cmaps_b))

    # Encode tags into tag maps with <end> at the end
    tmaps = list(map(lambda s: list(map(lambda t: tag_map[t], s)) + [tag_map['<end>']], tags))
    # Since we're using CRF scores of size (prev_tags, cur_tags), find indices of target sequence in the unrolled scores
    # This will be row_index (i.e. prev_tag) * n_columns (i.e. tagset_size) + column_index (i.e. cur_tag)
    tmaps = list(map(lambda s: [tag_map['<start>'] * len(tag_map) + s[0]] + [s[i - 1] * len(tag_map) + s[i] for i in
                                                                             range(1, len(s))], tmaps))
    # Note - the actual tag indices can be recovered with tmaps % len(tag_map)

    # Map position index to its id
    mask_maps = list(map(lambda s: list(map(lambda t: mask_map[t], s)) + [mask_map['<end>']], pos_mask))
    # Pad, because need fixed length to be passed around by DataLoaders and other layers
    word_pad_len = max(list(map(lambda s: len(s), wmaps)))
    char_pad_len = max(list(map(lambda s: len(s), cmaps_f)))

    # Sanity check
    assert word_pad_len == max(list(map(lambda s: len(s), tmaps)))

    padded_wmaps = []
    padded_cmaps_f = []
    padded_cmaps_b = []
    padded_cmarkers_f = []
    padded_cmarkers_b = []
    padded_tmaps = []
    padded_pos_maps = []
    wmap_lengths = []
    cmap_lengths = []

    for w, cf, cb, cmf, cmb, t, p in zip(wmaps, cmaps_f, cmaps_b, cmarkers_f, cmarkers_b, tmaps, mask_maps):
        # Sanity  checks
        assert len(w) == len(cmf) == len(cmb) == len(t) == len(p)
        assert len(cmaps_f) == len(cmaps_b)

        # Pad
        # A note -  it doesn't really matter what we pad with, as long as it's a valid index
        # i.e., we'll extract output at those pad points (to extract equal lengths), but never use them

        padded_wmaps.append(w + [word_map['<pad>']] * (word_pad_len - len(w)))
        padded_cmaps_f.append(cf + [char_map['<pad>']] * (char_pad_len - len(cf)))
        padded_cmaps_b.append(cb + [char_map['<pad>']] * (char_pad_len - len(cb)))

        # 0 is always a valid index to pad markers with (-1 is too but torch.gather has some issues with it)
        padded_cmarkers_f.append(cmf + [0] * (word_pad_len - len(w)))
        padded_cmarkers_b.append(cmb + [0] * (word_pad_len - len(w)))

        padded_tmaps.append(t + [tag_map['<pad>']] * (word_pad_len - len(t)))

        padded_pos_maps.append(p + [mask_map['<pad>']] * (word_pad_len - len(w)))

        wmap_lengths.append(len(w))
        cmap_lengths.append(len(cf))

        # Sanity check
        assert len(padded_wmaps[-1]) == len(padded_tmaps[-1]) == len(padded_cmarkers_f[-1]) == len(
            padded_cmarkers_b[-1]) == word_pad_len
        assert len(padded_cmaps_f[-1]) == len(padded_cmaps_b[-1]) == char_pad_len

    padded_wmaps = torch.LongTensor(padded_wmaps)
    padded_cmaps_f = torch.LongTensor(padded_cmaps_f)
    padded_cmaps_b = torch.LongTensor(padded_cmaps_b)
    padded_cmarkers_f = torch.LongTensor(padded_cmarkers_f)
    padded_cmarkers_b = torch.LongTensor(padded_cmarkers_b)
    padded_tmaps = torch.LongTensor(padded_tmaps)
    padded_pos_maps = torch.FloatTensor(padded_pos_maps)
    wmap_lengths = torch.LongTensor(wmap_lengths)
    cmap_lengths = torch.LongTensor(cmap_lengths)

    return padded_wmaps, padded_cmaps_f, padded_cmaps_b, padded_cmarkers_f, padded_cmarkers_b, padded_tmaps, \
           wmap_lengths, cmap_lengths, padded_pos_maps


def init_embedding(input_embedding):
    """
    Initialize embedding tensor with values from the uniform distribution.

    :param input_embedding: embedding tensor
    :return:
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def load_embeddings(emb_file, word_map, expand_vocab=True):
    """
    Load pre-trained embeddings for words in the word map.

    :param emb_file: file with pre-trained embeddings (in the GloVe format)
    :param word_map: word map
    :param expand_vocab: expand vocabulary of word map to vocabulary of pre-trained embeddings?
    :return: embeddings for words in word map, (possibly expanded) word map,
            number of words in word map that are in-corpus (subject to word frequency threshold)
    """
    with open(emb_file, 'r') as f:
        emb_len = len(f.readline().split(' ')) - 1

    print("Embedding length is %d." % emb_len)

    # Create tensor to hold embeddings for words that are in-corpus
    ic_embs = torch.FloatTensor(len(word_map), emb_len)
    init_embedding(ic_embs)

    if expand_vocab:
        print("You have elected to include embeddings that are out-of-corpus.")
        ooc_words = []
        ooc_embs = []
    else:
        print("You have elected NOT to include embeddings that are out-of-corpus.")

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        if not expand_vocab and emb_word not in word_map:
            continue

        # If word is in train_vocab, store at the correct index (as in the word_map)
        if emb_word in word_map:
            ic_embs[word_map[emb_word]] = torch.FloatTensor(embedding)

        # If word is in dev or test vocab, store it and its embedding into lists
        elif expand_vocab:
            ooc_words.append(emb_word)
            ooc_embs.append(embedding)

    lm_vocab_size = len(word_map)  # keep track of lang. model's output vocab size (no out-of-corpus words)

    if expand_vocab:
        print("'word_map' is being updated accordingly.")
        for word in ooc_words:
            word_map[word] = len(word_map)
        ooc_embs = torch.FloatTensor(np.asarray(ooc_embs))
        embeddings = torch.cat([ic_embs, ooc_embs], 0)

    else:
        embeddings = ic_embs

    # Sanity check
    assert embeddings.size(0) == len(word_map)

    print("\nDone.\n Embedding vocabulary: %d\n Language Model vocabulary: %d.\n" % (len(word_map), lm_vocab_size))

    return embeddings, word_map, lm_vocab_size


def clip_gradient(optimizer, grad_clip):
    """
    Clip gradients computed during backpropagation to prevent gradient explosion.

    :param optimizer: optimized with the gradients to be clipped
    :param grad_clip: gradient clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(task, epoch, model, optimizer, val_f1, word_map, char_map, tag_map, lm_vocab_size, is_best, fold):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimized
    :param val_f1: validation F1 score
    :param word_map: word map
    :param char_map: char map
    :param tag_map: tag map
    :param lm_vocab_size: number of words in-corpus, i.e. size of output vocabulary of linear model
    :param is_best: is this checkpoint the best so far?
    :return:
    """
    state = {'epoch': epoch,
             'f1': val_f1,
             'model': model,
             'optimizer': optimizer,
             'word_map': word_map,
             'tag_map': tag_map,
             'char_map': char_map,
             'lm_vocab_size': lm_vocab_size}
    # filename = 'fold_checkpoint_lm_lstm_crf.pth.tar'
    filename = '{0}_fold{1}_checkpoint_lm_lstm_crf.pth.tar'.format(task, fold)
    torch.save(state, 'model/'+filename)
    # torch.save(model.state_dict(), 'model/'+filename)

    # If checkpoint is the best so far, create a copy to avoid being overwritten by a subsequent worse checkpoint
    if is_best:
        torch.save(state, 'model/BEST_' + filename)
        # torch.save(model.state_dict(), 'model/BEST_' + filename)


def load_checkpoint(model, fold):

    filename = 'model/BEST_fold{0}_checkpoint_lm_lstm_crf.pth.tar'.format(fold)
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint)
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # word_map.load_state_dict(checkpoint['word_map'])
    # char_map.load_state_dict(checkpoint['char_map'])
    # tag_map.load_state_dict(checkpoint['tag_map'])
    # epoch = checkpoint['epoch']
    # lm_vocab_size = checkpoint['lm_vocab_size']

    # return epoch, model, optimizer, val_f1, word_map, char_map, tag_map, lm_vocab_size

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, new_lr):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param new_lr: new learning rate
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def log_sum_exp(tensor, dim):
    """
    Calculates the log-sum-exponent of a tensor's dimension in a numerically stable way.

    :param tensor: tensor
    :param dim: dimension to calculate log-sum-exp of
    :return: log-sum-exp
    """
    m, _ = torch.max(tensor, dim)
    m_expanded = m.unsqueeze(dim).expand_as(tensor)
    return m + torch.log(torch.sum(torch.exp(tensor - m_expanded), dim))


def calculate_f1_score(golds, preds, tag_map):
    if torch.cuda.is_available():
        pun_tag_id = tag_map.get(1)
    else:
        pun_tag_id = tag_map.get("1")

    # for k in tag_map.keys():
    #     print(type(k))
    true_pos = 0
    true_neg = 0
    false_neg = 0
    false_pos = 0
    total = 0
    loc_true_pos = 0
    loc_true_neg = 0
    loc_false_neg = 0
    loc_false_pos = 0
    loc_total = 0
    loc_multi_pos = 0

    for g, p in zip(golds, preds):
        total += 1
        is_pun = False
        gold = g.cpu().numpy()
        pred = p.cpu().numpy()
        if pun_tag_id in gold and pun_tag_id in pred:
            true_pos += 1
            is_pun = True
        elif pun_tag_id in gold and pun_tag_id not in pred:
            is_pun = True
            false_neg += 1
        elif pun_tag_id not in gold and pun_tag_id not in pred:
            true_neg += 1
        elif pun_tag_id not in gold and pun_tag_id in pred:
            false_pos += 1

        if is_pun:
            loc_total += 1
            idx = np.where(gold == pun_tag_id)
            # if any([(gold == d_).all() for d_ in pred]):
            if pred[idx[0]] == pun_tag_id:
                tag_occ_count = len(np.where(pred == pun_tag_id)[0])
                # for i in range(len(pred)):
                #     print(pred[i])
                #     if pred[i] == pun_tag_id:
                #         tag_occ_count += 1
                if tag_occ_count == 1:
                    loc_true_pos += 1
                else:
                    loc_false_pos += 1
                    loc_multi_pos += 1
            else:
                if pun_tag_id not in pred:
                    loc_false_neg += 1
                else:
                    loc_false_pos += 1

    c_precision = true_pos * 1.0 / (true_pos + false_pos) * 100 if (true_pos + false_pos) > 0 else 0.0
    c_recall = true_pos * 1.0 / (true_pos + false_neg) * 100 if (true_pos + false_neg) > 0 else 0.0
    c_f1 = (c_precision * c_recall * 2) / (c_precision + c_recall) if (c_precision + c_recall) > 0 else 0.0
    c_accuracy = (true_pos + true_neg) * 1.0 / total * 100 if total > 0 else 0.0

    l_precision = loc_true_pos * 1.0 / (loc_true_pos + loc_false_pos) * 100 if (loc_true_pos + loc_false_pos) > 0 else 0.0
    l_recall = loc_true_pos * 1.0 / (loc_true_pos + loc_false_neg) * 100 if (loc_true_pos + loc_false_neg) > 0 else 0.0
    l_f1 = l_precision * l_recall * 2.0 / (l_recall + l_precision) if (l_recall + l_precision) > 0 else 0.0
    l_accuracy = loc_true_pos * 1.0 / loc_total * 100 if loc_total > 0 else 0.0

    # for g, p in zip(golds, preds):
    #     gold = g.cpu().numpy()
    #     pred = p.cpu().numpy()
        # print(gold, pred)

    # print(true_pos, false_pos, true_neg, false_neg)
    # print(c_precision, c_recall, c_accuracy)
    # print(c_f1)

    return c_f1, l_f1, true_pos, false_pos, true_neg, false_neg, loc_true_pos, loc_false_pos, loc_true_neg, loc_false_neg, loc_multi_pos


def save_to_csv(save_path, data):
    if not os.path.exists(os.path.dirname(save_path)):
        try:
            os.makedirs(os.path.dirname(save_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    csv.register_dialect('myDialect', delimiter='\t', quoting=csv.QUOTE_NONE)
    with open(save_path, 'w') as save_file:
        writer = csv.writer(save_file, dialect='myDialect')
        writer.writerows(data)
    print("Save to file: {0}".format(save_path))


