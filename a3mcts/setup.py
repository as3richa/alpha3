from distutils.core import setup, Extension

a3mcts = Extension('a3mcts', sources = ['module.cpp'])

setup(name = 'ac3mcts', version = '1.0', description = 'C++ Monte-Carlo tree search implementation', ext_modules = [a3mcts])
