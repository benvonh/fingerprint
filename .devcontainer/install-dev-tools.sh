#!/bin/bash
# Put any custom installs in this file
# pip3 cache purge
# sudo apt-get autoremove -y
# sudo apt-get clean
sudo apt-get update
sudo apt-get install -y libgl1
pip install opencv_contrib_python