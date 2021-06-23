.. LBweights documentation master file, created by
   sphinx-quickstart on Wed Jan 29 14:10:49 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the LBweights documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Lattice-Boltzmann-weights
=========================

"LBweights.py" is a Python script that calculates the weights of the (user-supplied) shells of a Lattice Boltzmann model on a simple cubic lattice, based upon numerically solving the Maxwell-Boltzmann constraint (MBC) equations. The script supports arbitrary spacial dimensions and arbitrary tensor ranks up to which the MBCs must hold. The script requires a Python installation (version 3.5 or 2.7) as well as NumPy. It assumes that the speed of sound is a free parameter and hence needs more shells than a model whose speed of sound takes on a definite value that is required for consistency. The output is typically given in the form: weights as a function of sound speed. There are cases where the supplied set of velocities does not admit any solution; in this case the script aborts. There are also cases where it admits infinitely many solutions; in this case an additional script "Continue.py" is used, which builds upon data that the main script stores on file.

In case of a unique solution, the script also calculates the interval(s) of sound speed for which all the weights are positive. At the borders of these intervals, at least one of the weights is zero, such that the corresponding shell may be discarded and one obtains a "reduced model". In this way, the script is able to reproduce well-known models like D2Q9, D3Q19, D3Q15, etc., but can also easily find higher-order models with significantly more speeds.

For Continue.py, the user has to supply a well-defined value of the sound speed (or an interval plus step size for scanning several values). Moreover, it requires the specification of a shell (or of a set of shells) whose weight (or sum of weights) is to be minimized. Continue.py then finds an optimal solution to the thus-specified linear programming problem. Continue.py therefore requires the package cvxpy, see http://www.cvxpy.org/ .

A significant part of the code is not in the main scripts but rather in a collection of functions in "Functions.py", which must be available to "LBweights.py" and "Continue.py".

Tedious tasks like the construction of velocity shells from the velocity modulus are done by the script.

Apart from being useful for researchers and practitioners, the script may perhaps also be used in a classroom setting.

A detailed description of the underlying mathematical theory, together with illustrative examples, is given in the
paper "Semi-automatic construction of Lattice Boltzmann models" by Dominic Spiller and Burkhard Duenweg, see http://arxiv.org/abs/2004.03509 (original at Physical Review E, https://journals.aps.org/pre/abstract/10.1103/PhysRevE.101.043310 / open access).

Installation
------------
::

    $ git clone https://github.com/BDuenweg/Lattice-Boltzmann-weights.git
    $ virtualenv venv
    $ source venv/bin/activate
    $ cd Lattice-Boltzmann-weights
    $ pip install -r requirements.txt

LBweights.py
============

Calculation of the weights of an LB model. You can either supply the input
data interactively or by the following command line arguments:

Usage
-----
.. code-block:: none

    LBweights.py [-h] [-d D] [-m M] [-c C [C ...]] [-s S] 
                        [-y] [--test] [--quiet] [--write-latex]

    optional arguments:
      -h, --help     show this help message and exit
      -d D           spacial dimension of the lattice
      -m M           Maximum tensor rank
      -c C [C ...]   Space separated list of the radii c_i^2 of
                     the desired velocity shells
      -s S           Random number generator seed
      -y             Answer all prompts with yes (may overwrite
                     file data.npz)
      --test         Test, whether a set of weights that can be
                     written as a linear parametric equation
                     w = w_0 + lambda_1 w_1 + lambda_2 w_2
                     solves the equation A.w == b for given
                     speed of sound.
                     Weights and speed of sound are entered
                     interactively by the user.
      --quiet        Turn off most of the output
      --write-latex  Write unique solution to the file
                     "latex_tables.dat" in form of a latex
                     table. This will append to any existing
                     file.



.. automodule:: LBweights
   :members:

Continue.py
===========

Find optimal weights for an underdetermined problem. This requires the file 
``data.npz`` to be present in the directory that can be written by 
``LBweights.py`` if an underdetermined problem is encountered.
You can either supply the input data interactively or by the following command
line arguments:

Usage
-----
.. code-block:: none

    Continue.py [-h] [-c C [C ...]] [-m M [M ...]]


    optional arguments:
      -h, --help    show this help message and exit
      -c C [C ...]  Range/value of c_s^2 to consider, either in 
                    the form <min> <max> <incr> or a single 
                    value.
      -m M [M ...]  List of indices of the weights that are to 
                    be minimized. You can use -1 to refer to the
                    last shell etc.

.. automodule:: Continue
   :members:

Functions.py
============
.. automodule:: Functions
   :members:
