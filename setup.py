from distutils.core import setup, Extension

a3mcts = Extension('alpha3.a3mcts', sources=['alpha3/a3mcts.cpp'])

setup(name='alpha3',
      version='1.0.0',
      author='Adam Richardson (as3richa)', 
      packages=['alpha3'],
      description='Pedagogical reimplementation of AlphaZero',
      ext_modules=[a3mcts])
