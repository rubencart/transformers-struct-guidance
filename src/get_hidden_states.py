import json
import random
import sys

import nltk
from tqdm import tqdm

from get_oracle import get_actions_and_terms, is_valid_action_sequence
from plm_gen import PLM, load_data
import numpy as np
import torch

import os



if __name__ == "__main__":
    seed = 4325
    random_init = True
    restore_from = '/cw/liir_code/NoCsBack/wolf/projects/MMMAli/models/plm/xplm-mask_bllip-lg_rand-init_1101_5.params'
    test_data = 'data/train_gen.oracle'
    test_data = '/cw/liir/NoCsBack/testliir/rubenc/vpcfg-dev/data_v2/absurd_trees_with_tags.json'
    #test_data = '/cw/liir/NoCsBack/testliir/rubenc/vpcfg-dev/data_v2/train2017_trees_with_tags.json'
    add_structured_mask = True



    log_softmax = torch.nn.LogSoftmax(-1)

    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Set random seed
    RANDOM_SEED = seed if seed is not None else int(np.random.random() * 10000)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    print('Random seed: {}'.format(RANDOM_SEED), file=sys.stderr)

    # specify non-GEN parsing actions
    NT_CATS = ['ADJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC', 'NP', 'NX', 'PP',
               'PRN', 'PRT', 'QP', 'RRC', 'S', 'SBAR', 'SBARQ', 'SINV', 'SQ', 'UCP', 'VP',
               'WHADJP', 'WHADVP', 'WHNP', 'WHPP', 'X']
    REDUCE = 'REDUCE()'
    ROOT = '[START]'

    plm = PLM(is_random_init=random_init, NT_CATS=NT_CATS, REDUCE=REDUCE, ROOT=ROOT, device=device)

    if restore_from is not None:
        print('Load parameters from {}'.format(restore_from), file=sys.stderr)
        checkpoint = torch.load(restore_from)
        plm.model.load_state_dict(checkpoint['model_state_dict'])


    if test_data is None:
        raise ValueError('Test data not specified')

    test_data_path = test_data
    test_lines = load_data(test_data_path)

    with open(test_data_path) as f:
        lines = f.readlines()
    lines = [json.loads(line)[2] for line in lines]
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']

    line_ctr = 0
    res = []
    for line in tqdm(lines[:16]):
        line_ctr += 1
        # assert that the parenthesis are balanced
        if line.count('(') != line.count(')'):
            raise NotImplementedError('Unbalanced number of parenthesis in line ' + str(line_ctr))
        output_actions, output_terms = get_actions_and_terms(line, is_generative=True)

        if not is_valid_action_sequence(output_actions):
            print('ivalid action seq in line ' + str(line_ctr))
            continue

        if len(output_actions) > 500:
            print('too long sentence in line ' + str(line_ctr))
            continue

        for i, a in enumerate(output_actions):
            if a == 'NT(NML)':
                output_actions[i] = 'NT(NP)'

        res.append(' '.join(output_actions))

    '''RANDOM_SEED = 43
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)'''
    #res = ['NT(NP) NT(NP) A bicycle replica REDUCE() NT(PP) with NT(NP) NT(NP) a clock REDUCE() NT(PP) as NT(NP) the front wheel REDUCE() REDUCE() REDUCE() REDUCE() REDUCE()']
    hs, mask = plm.get_hidden_states(res, add_structured_mask=add_structured_mask)
    print('hs shape: {}'.format(hs.shape))

