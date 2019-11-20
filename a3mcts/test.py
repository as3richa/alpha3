import a3mcts
mcts = a3mcts.MCTS(None, 0.1, 0.1)
print(mcts.game_state())
print(mcts.expanded())

leaf = mcts.select_leaf()
print(leaf)
print(mcts.expand_leaf(leaf, 1, [(0, False, 0.5), (1, True, 0.5)]))
print(mcts.expand_leaf(leaf, 1, [[], (1, True, 0.5)]))
