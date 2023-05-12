import json

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from tqdm import tqdm


def is_valid_action_sequence(action_sequence):
    flag = True
    for k, action in enumerate(action_sequence):
        if action == "REDUCE":
            if k <= 1:
                flag = False
                break
            if action_sequence[k-1].startswith('NT('):
                flag = False
                break
    return flag


def is_next_open_bracket(line, start_idx):
    for char in line[(start_idx + 1):]:
        if char == '(':
            return True
        elif char == ')':
            return False
    raise IndexError('Bracket possibly not balanced, open bracket not followed by closed bracket')


def get_between_brackets(line, start_idx):
    output = []
    for char in line[(start_idx + 1):]:
        if char == ')':
            break
        assert not(char == '(')
        output.append(char)
    return ''.join(output)


def get_tags_tokens_lowercase(line):
    output = []
    #print 'curr line', line_strip
    line_strip = line.rstrip()
    #print 'length of the sentence', len(line_strip)
    for i in range(len(line_strip)):
        if i == 0:
            assert line_strip[i] == '('
        if line_strip[i] == '(' and not(is_next_open_bracket(line_strip, i)): # fulfilling this condition means this is a terminal symbol
            output.append(get_between_brackets(line_strip, i))
    #print 'output:',output
    output_tags = []
    output_tokens = []
    output_lowercase = []
    for terminal in output:
        terminal_split = terminal.split()
        assert len(terminal_split) == 2 # each terminal contains a POS tag and word
        output_tags.append(terminal_split[0])
        output_tokens.append(terminal_split[1])
        output_lowercase.append(terminal_split[1].lower())
    return [output_tags, output_tokens, output_lowercase]


def get_nonterminal(line, start_idx):
    assert line[start_idx] == '(' # make sure it's an open bracket
    output = []
    for char in line[(start_idx + 1):]:
        if char == ' ':
            break
        elif char == '#':
            break
        elif char == '-':
            break
        assert not(char == '(') and not(char == ')')
        output.append(char)
    return ''.join(output)


def get_actions_and_terms(line, is_generative):
    output_actions = []
    output_terms = []
    line_strip = line.rstrip()
    i = 0
    max_idx = (len(line_strip) - 1)
    while i <= max_idx:
        assert line_strip[i] == '(' or line_strip[i] == ')'
        if line_strip[i] == '(':
            if is_next_open_bracket(line_strip, i): # open non-terminal
                curr_NT = get_nonterminal(line_strip, i)
                output_actions.append('NT(' + curr_NT + ')')
                i += 1
                while line_strip[i] != '(': # get the next open bracket, which may be a terminal or another non-terminal
                    i += 1
            else: # it's a terminal symbol
                terminal = get_between_brackets(line_strip, i)
                terminal_split = terminal.split()
                assert len(terminal_split) == 2 # each terminal contains a POS tag and word
                token = terminal_split[1]
                output_terms.append(token)
                if is_generative:
                    # generative parsing
                    output_actions.append(token)
                else:
                    # discriminative parsing
                    output_actions += ['SHIFT']
                while line_strip[i] != ')':
                    i += 1
                i += 1
                while line_strip[i] != ')' and line_strip[i] != '(':
                    i += 1
        else:
            output_actions.append('REDUCE()')
            if i == max_idx:
                break
            i += 1
            while line_strip[i] != ')' and line_strip[i] != '(':
                i += 1
    assert i == max_idx
    return output_actions, output_terms


def parse_to_tg(lines, rb=False):
    result = []
    line_ctr = 0
    # get the oracle action sequences for the input file
    for line in tqdm(lines):
        line_ctr += 1
        # assert that the parenthesis are balanced
        if line.count('(') != line.count(')'):
            raise NotImplementedError('Unbalanced number of parenthesis in line ' + str(line_ctr))

        try:
            output_actions, output_terms = get_actions_and_terms(line, is_generative=True)
        except:
            continue

        if output_actions[0] == 'NT(S1)':
            output_actions = output_actions[1:-1]

        if not is_valid_action_sequence(output_actions):
            continue

        if len(output_actions) > 500:
            continue

        if rb:
            new_output_actions = []
            reduce_outputs = []
            for action in output_actions:
                if action.startswith('NT('):
                    nt = action[3:-1]
                    new_output_actions.append(action)
                    reduce_outputs.append('REDUCE({})'.format(nt))
                    reduce_outputs.append('REDUCE({})'.format(nt))
                elif action == 'REDUCE()':
                    pass
                else:
                    new_output_actions.append(action)
            reduce_outputs.reverse()
            new_output_actions.extend(reduce_outputs)
        else:
            new_output_actions = []
            stack = []
            for action in output_actions:
                if action.startswith('NT('):
                    stack.append(action[3:-1])
                    new_output_actions.append(action)
                elif action == 'REDUCE()':
                    nt = stack.pop()
                    new_output_actions.append('REDUCE({})'.format(nt))
                    new_output_actions.append('REDUCE({})'.format(nt))
                else:
                    new_output_actions.append(action)

        result.append(new_output_actions)

    return result



def _calc_constituents(tree, tg_parsing):
    assert tree[0][:3] == 'NT('
    consts = []
    stack = []
    i = 0
    if tg_parsing:
        reduce_before = False
        for token in tree:
            if token[:3] == 'NT(':
                reduce_before = False
                stack.append([token[3:-1], [i, i]])
            elif token[:7] == 'REDUCE(':
                if reduce_before:
                    reduce_before = False
                    const = stack.pop(-1)
                    assert const[0] == token[7:-1]
                    const[1][1] = i
                    consts.append(const)
                else:
                    reduce_before = True
            else:
                reduce_before = False
                i += 1
    else:
        for token in tree:
            if token[:3] == 'NT(':
                stack.append([token[3:-1], [i, i]])
            elif token[:7] == 'REDUCE(':
                const = stack.pop(-1)
                const[1][1] = i
                consts.append(const)
            else:
                i += 1

    res = []
    for const in consts:
        res.append(const[0] + '(' + str(const[1][0]) + '-' +  str(const[1][1]) + ')')
    return res


def calculate_f1(tree, gt_tree, tg_parsing=True):
    tree_const, gt_const = _calc_constituents(tree, tg_parsing=tg_parsing), _calc_constituents(gt_tree, tg_parsing=tg_parsing)
    true_pos = 0
    for const in tree_const:
        if const in gt_const:
            true_pos += 1
    precision = true_pos / len(tree_const)
    recall = true_pos / len (gt_const)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def load_data(path):
    with open(path) as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line != '']
    return lines


'''gold_tree = 'NT(S) NT(NP) Sales executives REDUCE() NT(VP) were NT(VP) examining NT(NP) the figures REDUCE() NT(PP) with NT(NP) great care REDUCE() REDUCE() REDUCE() REDUCE() NT(NP) yesterday REDUCE() REDUCE()'
c_tree = 'NT(S) NT(NP) Sales executives REDUCE() NT(VP) were NT(VP) examining NT(NP) the figures REDUCE() NT(PP) with NT(NP) great care yesterday REDUCE() REDUCE() REDUCE() REDUCE() REDUCE()'
rb_tree = 'NT(S) NT(NP) Sales executives NT(VP) were NT(VP) examining NT(NP) the figures NT(PP) with NT(NP) great care NT(NP) yesterday REDUCE() REDUCE() REDUCE() REDUCE() REDUCE() REDUCE() REDUCE() REDUCE()'
a = 'NT(S) NT(NP) Industry sources NT(VP) put NT(NP) NT(NP) the value NT(PP) of NT(NP) the proposed acquisition NT(PP) at NT(NP) NT(QP) more than $ 100 million *U* . REDUCE(QP) REDUCE(QP) REDUCE(NP) REDUCE(NP) REDUCE(PP) REDUCE(PP) REDUCE(NP) REDUCE(NP) REDUCE(PP) REDUCE(PP) REDUCE(NP) REDUCE(NP) REDUCE(NP) REDUCE(NP) REDUCE(VP) REDUCE(VP) REDUCE(NP) REDUCE(NP) REDUCE(S) REDUCE(S)'
b = _calc_constituents(a.split(' '), tg_parsing=True)
print(calculate_f1(c_tree.split(' '), gold_tree.split(' ')))
print(calculate_f1(rb_tree.split(' '), gold_tree.split(' ')))
'''

'''data = load_data('../data/tg_dev_gen.oracle')
data_rb = load_data('../data/rightBranch_tg_dev_gen.oracle')'''

ann_trees_train_json = '/cw/liir/NoCsBack/testliir/rubenc/vpcfg-dev/data_v2/train2017_trees_with_tags.json'
ann_trees_val_json = '/cw/liir/NoCsBack/testliir/rubenc/vpcfg-dev/data_v2/val2017_trees_with_tags.json'
ann_trees_absurd_json = '/cw/liir/NoCsBack/testliir/rubenc/vpcfg-dev/data_v2/verified_absurd_amt_trees_with_tags.json'

with open(ann_trees_val_json) as f:
    file = f.readlines()
lines = []
for line in file:
    lines.append(json.loads(line)[2])
trees = parse_to_tg(lines, rb=False)
trees_rb = parse_to_tg(lines, rb=True)

res = []
for d, drb in tqdm(zip(trees, trees_rb)):
    res.append(calculate_f1(d, drb))
print('result: {}'.format(sum(res)/len(res)))
print('done')
