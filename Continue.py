"""Contains routines to treat the case of infinitely many solutions.

Exit codes\:
    - 0\:   No optimal solution found
    - 1\:   Optimal solution found
    - 127\: General error

"""

import argparse
import sys
import numpy as np
import cvxpy as cvx
from Functions import YesNo, Echo, EchoError


def ParseArguments():
    """Function to parse command line options.

    Returns:
        dict: Dictionary of command line options

    """

    parser = argparse.ArgumentParser(description="""Find optimal weights for
an underdetermined problem.\nYou can either supply the input data interactively
or by the following command line arguments:""")

    parser.add_argument(
        "-c", nargs='+', type=float,
        help="""Range/value of c_s^2 to consider, either in the form <min>
<max> <incr> or a single value.""")

    parser.add_argument("-m", nargs='+', type=int,
        help="""List of indices of the weights that are to be minimized. You
can use -1 to refer to the last shell etc.""")

    # if no arguments are given, print help text
    if len(sys.argv) == 1:
        parser.print_help()
        Echo('\n')

    return vars(parser.parse_args())


def Solve(V, ReducedRhs, NumberOfRows, ShellSizes, CsSquared, MinimizeWeights):
    """Solve the minimization problem via convex optimization.
    See: https://www.cvxpy.org/

    Args:
        V (numpy.ndarray): Orthogonal matrix that results from the singular
            value decomposition A=U.S.V
        ReducedRhs (numpy.ndarray): Pruned matrix that has the inverse singular
            values on the diagonal.
        NumberOfRows (int): Number of rows of A
        ShellSizes (list): List of shell sizes (int) NOT including zero shell
        CsSquared (float): Speed of sound squared
        MinimizeWeights (list): List of indices of the weights that shall be
            minimized in the procedure

    Returns:
        cvxpy.problems.problem.Problem: cvxpy problem. Problem.status indicates
            whether or not the problem could be solved.

    """

    TotalNumberOfShells = len(ShellSizes) # without zero shell

    Eye = np.eye(NumberOfRows, TotalNumberOfShells)
    A0 = Eye.dot(V)
    # pad A0 in order to include normalization condition
    A0 = np.vstack((ShellSizes, A0))
    A0 = np.hstack((np.eye(NumberOfRows + 1, 1), A0))
    A = cvx.Parameter(A0.shape)
    A.value = A0

    B0 = np.array([CsSquared**(i+1) for i in range(ReducedRhs.shape[1])])
    B0 = ReducedRhs.dot(B0)
    # pad B0 in order to include normalization condition
    B0 = np.append([1.], B0)
    B = cvx.Parameter(B0.shape[0])
    B.value = B0

    W = cvx.Variable(TotalNumberOfShells + 1)
    C = np.zeros(TotalNumberOfShells + 1)
    for i in MinimizeWeights:
        C[i] = 1

    Objective = cvx.Minimize(C*W)
    Constraints = [W >= 0., A*W == B]
    Problem = cvx.Problem(Objective, Constraints)

    Options = {
        "max_iters" : 100,
        "abstol" : 1e-7,
        "reltol" : 1e-6,
        "feastol" : 1e-7,
        "abstol_inacc" : 5e-5,
        "reltol_inacc" : 5e-5,
        "feastol_inacc" : 1e-4,
        "verbose" : False
        }

    Problem.solve(cvx.ECOS, **Options)
    # Problem.solve(cvx.GLPK, verbose=False)
    return Problem


if __name__ == '__main__':
    # Set speed of sound
    Arguments = ParseArguments()
    if Arguments["c"] is None:
        Echo("""The script needs a value for c_s^2 in order to operate. You can
give a range of values in the format <min> <max> <step>. Alternatively you can
provide a single value for c_s^2:\n""")
        Range = str(input())
        Range = list(map(float, Range.split()))
    else:
        Range = Arguments['c']

    if Arguments['m'] is None:
        Echo("""Please enter the indices of the weights that you want to
be minimized in the format 1 2 3. You can use -1 to refer to the last shell
etc.:\n""")
        MinimizeWeights = str(input())
        MinimizeWeights = list(map(int, MinimizeWeights.split()))
    else:
        MinimizeWeights = Arguments['m']

    # Load data from disk
    Echo("Loading data from file data.npz")
    Data = np.load("data.npz")
    V = Data['V']
    ReducedRhs = Data['ReducedRhs']
    NumberOfRows = Data['NumberOfRows']
    ShellSizes = Data['ShellSizes']

    CsSquared  = Range[0]

    # run for single value
    if len(Range) == 1:
        Echo("Using c_s^2 = %f" % CsSquared)
        Problem = Solve(V, ReducedRhs, NumberOfRows, ShellSizes, CsSquared,
                MinimizeWeights)
        Echo('\n')
        Solution = Problem.status == "optimal"
        if Solution:
            Echo("Optimal solution found: ")
            for i, val in enumerate(Problem.variables()[0].value):
                EchoError("  w[%d] = %16.10e" % (i, val))

        else:
            EchoError("Could not find optimal solution.")

        exit(Solution)


    # run for range of values
    SolutionFound = False
    if len(Range) == 3:
        Echo("Using range = %s" % Range)
        Outfilename = "results.dat"
        Echo("""Valid results are written to the file %s in the format: c_s^2
w_0 w_1 ... This will overwrite any file called %s that already exists."""
        % (Outfilename, Outfilename))

        Outfile = open(Outfilename, 'w')
        if not YesNo("Is this OK? [Yn]"):
            Echo("Aborting the procedure.")
            exit(127)
        if Range[0] > Range[1]:
            Echo("Invalid range %s" % Range)
            exit(127)
        while CsSquared < Range[1]:
            Problem = Solve(V, ReducedRhs, NumberOfRows, ShellSizes,
                    CsSquared, MinimizeWeights)
            Solution = Problem.status == "optimal"

            Echo("  c_s^2 = %f: %s" % (CsSquared, Problem.status))

            if Solution:
                SolutionFound = True
                Outfile.write("%17.10e " % CsSquared)
                for Weight in Problem.variables()[0].value:
                    Outfile.write("%17.10e " % Weight)

                Outfile.write('\n')

            CsSquared += Range[2]
        Outfile.close()
        exit(SolutionFound)

    else:
        Echo("Invalid range %s" % Range)
        exit(127)
