=====
sdcli
=====
-------------------------------------
Generate images from the command-line
-------------------------------------

Description
===========
sdcli helps you generate images from the CLI. It's under heavy development so it should only work with:

* linux
* python3.12
* rocm

for now...

Usage
=====

.. code-block:: bash

    # create a venv
    python3.12 -m venv venv

    # install dependencies
    pip install --upgrade pip
    pip install --upgrade -r requirements.txt

    # edit the prompts (positive and negative)
    # feel free to use AUTOMATIC1111 syntax: "(huge breasts:1.5)", for example.
    vim prompts/default.*

    # generate an image
    ./sdcli.py

    # generate 10 images in 10 iterations (not working yet)
    ./sdcli.py 10

    # generate 10 images in 2 iterations (5 at once) (not working yet)
    ./sdcli.py 2 5

Wildcards
=========
I have a ton of wildcards in ``wildcards/*``. You can edit them.

.. warning::
    The name of each file needs to end in "s" for now.

Disclaimer
==========
This script sucks. I know it sucks. I am working so that it doesn't.
