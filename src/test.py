from tg_gen import RNNGMachine


rnngm = RNNGMachine()

actions = ['[START]', 'NT(S)', 'NT(NP)', 'The', 'blue', 'bird', 'REDUCE(NP)', 'REDUCE(NP)', 'NT(VP)', 'sings', 'REDUCE(VP)', 'REDUCE(VP)', 'REDUCE(S)', 'REDUCE(S)']
for action in actions:
    print(rnngm.update(action, len(actions)))