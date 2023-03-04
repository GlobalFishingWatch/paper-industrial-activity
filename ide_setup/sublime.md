# SUBLIME TEXT

    Text Editing, Done Right

Note that while Sublime Text has an evaluation mode that lasts basically forever,
it is not free. So if you decide to use it long term, it will mean forking over
$100 or forever putting up with nag dialogs and a guilty consistence.

## Installation

Sublime can be downloaded and installed from https://www.sublimetext.com/download. 


## Setup

There are many, many extensions available for sublime. Instructions for
installing some of the most useful for our workflow are listed below.

#### Package Control

Package Control makes it easier to find and install extensions for Sublime Text.
To install package control, follow the instructions at 
https://packagecontrol.io/installation

With package control installed, new packages can be added using `<CMD>-<SHIFT>-P`
to get to the command palette, then type *Package Control: Install Package*, the
auto-complete is very helpful here. Most packages can be installed just by typing
their name and hitting return, but a few requiring going to settings to setup
paths.

#### Useful Packages

Here are some useful packages for GFW workflows:


** Formatting / Linting Plugins **

* *isort*
* *sublack*
* *SublimeLinter*
* *SublimeLinter-flake8*
  - Edit package settings under SublimeLinter preference to add:

        "args" : "--max-line-length 88 --extend-ignore E203"
  
    Under the `flake8` section. This is for `black` compatibility.


** Other Stuff **

* *MarkdownPreview*
* *rsub* – interfaces with [rmate](https://github.com/textmate/rmate) to allow text editing
  with Sublime Text on remote instances. Not perfect, but very handy.


There also a whole host of theme plugins to change the display style, which we won't
go into here.
  
