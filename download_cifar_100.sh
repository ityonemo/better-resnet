#!/bin/sh

if [ ! -d cifar-100-python ]; then
  wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
  tar -xvf cifar-100-python.tar.gz
  rm cifar-100-python.tar.gz
else
  echo "cifar directory already exists!"
fi