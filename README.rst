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
* AMD RTX 6800 (512x512 size for now only... need more VRAM.)
* python3.12
* rocm

for now...

You could try it on an nvidia card with more RAM. You would need to:

* add xformers to requirements.txt and install it.
* remove the environ variables in the script.

I think that would get it to work.

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

Prompts
=======
We have positive ``prompts/default.prompt`` and negative ``prompts/default.nprompt``. You can edit them while generating images.
They get read in every iteration so you can start generating 100 images and edit that in real time.

Wildcards
=========
I have a ton of wildcards in ``wildcards/*``. You can edit them.

To use them, just imagine you're inside an f-string in python. One will get picked randomly for you and you can just use it in
singular mode. For exmaple: `{character}` always has one character there.

You could do some more complicated thing in the prompt, like, choosing randomly between the randomly chosen wildcard or some other
string. For example: `{random.choice(['side view', pov])}`. This will choose `side view` 50% of the time (yeah, right) and a
randomly picked `pov` the other 50% of the time.

.. warning::
    The name of each file needs to end in "s" for now.

Disclaimer
==========
This script sucks. I know it sucks. I am working so that it doesn't.
