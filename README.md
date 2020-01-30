# Lattice-Boltzmann-weights

"LBweights.py" is a Python script that calculates the weights of the (user-supplied) shells of a Lattice Boltzmann model on a simple cubic lattice, based upon numerically solving the Maxwell-Boltzmann constraint (MBC) equations. The script supports arbitrary spacial dimensions and arbitrary tensor ranks up to which the MBCs must hold. The script requires a Python installation (version 3.5 or 2.7) as well as NumPy. It assumes that the speed of sound is a free parameter and hence needs more shells than a model whose speed of sound takes on a definite value that is required for consistency. The output is typically given in the form: weights as a function of sound speed. There are cases where the supplied set of velocities does not admit any solution; in this case the script aborts. There are also cases where it admits infinitely many solutions; in this case an additional script "Continue.py" is used, which builds upon data that the main script stores on file.

In case of a unique solution, the script also calculates the interval(s) of sound speed for which all the weights are positive. At the borders of these intervals, at least one of the weights is zero, such that the corresponding shell may be discarded and one obtains a "reduced model". In this way, the script is able to reproduce well-known models like D2Q9, D3Q19, D3Q15, etc., but can also easily find higher-order models with significantly more speeds.

For Continue.py, the user has to supply a well-defined value of the sound speed (or an interval plus step size for scanning several values). Moreover, it requires the specification of a shell (or of a set of shells) whose weight (or sum of weights) is to be minimized. Continue.py then finds an optimal solution to the thus-specified linear programming problem. Continue.py therefore requires the package cvxpy, see http://www.cvxpy.org/ .

A significant part of the code is not in the main scripts but rather in a collection of functions in "Functions.py", which must be available to "LBweights.py" and "Continue.py".

Tedious tasks like the construction of velocity shells from the velocity modulus are done by the script.

Apart from being useful for researchers and practitioners, the script may perhaps also be used in a classroom setting.

A detailed description of the underlying mathematical theory is given in the paper "Semi-automatic construction of Lattice Boltzmann models" by Dominic Spiller and Burkhard Duenweg. A link to that paper will be given as soon as it is published.

More extensive documentation can be found at https://BDuenweg.github.io/Lattice-Boltzmann-weights/

## Installation

    $ git clone https://github.com/BDuenweg/Lattice-Boltzmann-weights.git
    $ virtualenv venv
    $ source venv/bin/activate
    $ cd Lattice-Boltzmann-weights
    $ pip install -r requirements.txt

# Usage

    $ python LBweights.py

