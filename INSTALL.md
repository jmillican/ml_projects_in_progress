# Installation instructions

We want to use tensorflow-metal in this project. This, unfortunately, seems to require specific Python and Tensorflow versions on OSX.

As such, the environment setup I'm using is

python3.11 -m venv venv
source venv/bin/activate
pip install tensorflow==2.15.0
pip install install tensorflow-metal
