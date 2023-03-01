from tg_gen import get_attention_mask_from_actions


def pretty_print_mask(mask):
    for sample_mask in mask:
        sample_mask = sample_mask[0]
        for token_mask in sample_mask:
            for i in token_mask:
                if i == 0:
                    print('-', end='')
                else:
                    print('X', end='')
            print()
        print('\n\n\n')


actions = ['[START]', 'NT(S)', 'NT(NP)', 'The', 'blue', 'bird', 'REDUCE(NP)', 'REDUCE(NP)', 'NT(VP)', 'sings', 'REDUCE(VP)', 'REDUCE(VP)', 'REDUCE(S)', 'REDUCE(S)']
pretty_print_mask(get_attention_mask_from_actions(actions, device='cpu'))