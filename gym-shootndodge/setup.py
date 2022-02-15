from setuptools import setup

setup(name='gym_shootndodge',
      version='1.2',
      author='Hamidreza Kamkari',
      description='''
      A new atari game involving a shooter that can go sideways and shoot and multiple deadly astroid blocks comming at it!
      The more you stay alive the more scores you gain and hitting astrolds reward additional points and destroying them 
      rewards a lot of points. 

      The package is created so that it can work in the OpenAI Gym environment and Reinforcement Learning (RL) methods
      can be applied to get better results from this problem.
      ''',
      author_email='hamidrezakamkari@gmail.com',
      install_requires=['gym', 'opencv-python', 'pillow', 'numpy']
)
