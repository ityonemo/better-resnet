#!/bin/sh

if [ ! -d cifar-10-python ]; then
  wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  tar -xvf cifar-10-python.tar.gz
  rm cifar-10-python.tar.gz
else
  echo "cifar directory already exists!"
fi