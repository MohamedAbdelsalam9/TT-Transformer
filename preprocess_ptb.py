''' Handling the data io '''
import argparse
import torch
import transformer.Constants as Constants

def read_instances_from_file(inst_file, max_sent_len, keep_case):
    ''' Convert file into word seq lists and vocab '''

    word_insts = []
    trimmed_sent_count = 0
    with open(inst_file) as f:
        for sent in f:
            if not keep_case:
                sent = sent.lower()
            words = sent.split()
            if len(words) > max_sent_len:
                trimmed_sent_count += 1
            word_inst = words[:max_sent_len]

            if word_inst:
                word_insts += [[Constants.BOS_WORD] + word_inst + [Constants.EOS_WORD]]
            else:
                word_insts += [None]

    print('[Info] Get {} instances from {}'.format(len(word_insts), inst_file))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts

def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''

    full_vocab = set(w for sent in word_insts for w in sent)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        Constants.BOS_WORD: Constants.BOS,
        Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1

    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]

def main():
    ''' Main function '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-train', required=True)
    parser.add_argument('-valid', required=True)
    parser.add_argument('-test', required=True)
    parser.add_argument('-save_data', required=True)
    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=50)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')
    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)

    opt = parser.parse_args()
    opt.max_token_seq_len = opt.max_word_seq_len + 2 # include the <s> and </s>

    # Training set
    train_word_insts = read_instances_from_file(
        opt.train, opt.max_word_seq_len, opt.keep_case)

    #- Remove empty instances
    train_word_insts = [s for s in train_word_insts if s]

    # Validation set
    valid_word_insts = read_instances_from_file(
        opt.valid, opt.max_word_seq_len, opt.keep_case)

    #- Remove empty instances
    valid_word_insts = [s for s in valid_word_insts if s]

    # Validation set
    test_word_insts = read_instances_from_file(
        opt.test, opt.max_word_seq_len, opt.keep_case)

    # - Remove empty instances
    test_word_insts = [s for s in test_word_insts if s]

    # Build vocabulary
    if opt.vocab:
        predefined_data = torch.load(opt.vocab)
        assert 'dict' in predefined_data

        print('[Info] Pre-defined vocabulary found.')
        word2idx = predefined_data['dict']
    else:
        print('[Info] Build vocabulary.')
        word2idx = build_vocab_idx(train_word_insts, opt.min_word_count)

    # word to index
    print('[Info] Convert word instances into sequences of word index.')
    train_insts = convert_instance_to_idx_seq(train_word_insts, word2idx)
    valid_insts = convert_instance_to_idx_seq(valid_word_insts, word2idx)
    test_insts = convert_instance_to_idx_seq(test_word_insts, word2idx)

    data = {
        'settings': opt,
        'dict': word2idx,
        'train': train_insts,
        'valid': valid_insts,
        'test': test_insts}

    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')

if __name__ == '__main__':
    main()
