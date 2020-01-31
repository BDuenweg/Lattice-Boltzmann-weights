"""
Calculate LB model vectors and weights for a simple
cubic lattice of arbitrary dimension

The method is described in D. Spiller's and B. Duenweg's paper
"Semi-automatic construction of Lattice Boltzmann models"
Therefore explanations in the code are not very detailed

Exit codes\:
    - 0\:   System has unique solution
    - 1\:   System has no solution
    - 2\:   System is underdetermined and requires further examination
    - 3\:   System has unique solution but there is no physically valid range of
      existence
    - 127\: General error

"""

import sys
import random
import numpy as np
from Functions import *


# Gather input data
def GetInputData(Arguments=None, ListOfThrowawayStrings=None):
    """Parse command line arguments. You can optionally give a list with the
    subshells that you want to discard.

    Args:
        Arguments (dict): Dictionary of command line arguments. This is
            useful, if the function is used in an automated script that does
            not rely on user input.
        ListOfThrowawayStrings (list): List of indices of the subshells to be
            discarded. This is useful, if the function is used in an automated
            script that does not rely on user input.

    Returns:
        tuple: Tuple ``(SpacialDimension, MaxTensorRank,
        ListOfTensorDimensions, GrandTotalList, Arguments)``

    """

    if Arguments is None:
        Arguments = ParseArguments()
    else:
        Arguments = Arguments

    Echo("""First I need to know in which spacial dimension the LB model
shall live. Please note that the model will live on a simple cubic lattice.""")

    if Arguments['d'] is None:
        SpacialDimension = int(input("Spacial dimension = ? "))
    else:
        SpacialDimension = Arguments['d']

    Echo("Confirmation: spacial dimension = %d" % SpacialDimension)
    Echo('\n')

    Echo("""Now please tell me up to which tensor rank you wish to satisfy the
Maxwell-Boltzmann constraints (for example, 2nd rank, 4th rank, etc.). Please
note that this should be an even number.""")

    if Arguments['m'] is None:
        MaxTensorRank = int(input("Maximum tensor rank = ? "))
    else:
        MaxTensorRank = Arguments['m']

    Echo("Confirmation: maximum tensor rank = %d" % MaxTensorRank)
    Echo('\n')

    DimensionOfTensorSpace = 0
    ListOfTensorDimensions = []

    for k in range(MaxTensorRank // 2):
        CurrentTensorRank = 2 * k + 2
        TensorDimension, ListOfPossibleTensors = \
            AnalyzeTensorDimension(CurrentTensorRank)
        ListOfTensorDimensions.append(TensorDimension)
        DimensionOfTensorSpace += TensorDimension

    Echo("""I expect that you need %d velocity shells plus the zero velocity
shell, which we do not need to consider explicitly.  Perhaps, however, you can
get away with less - just try!""" % DimensionOfTensorSpace)
    Echo('\n')

    if Arguments['c'] is None:
        Echo("""Please give me the squared lengths of the velocity shells
that you wish to analyze (excluding the zero velocity shell) in the simple
format: 1 2 3 4 5 """)
        ShellString = input()
        ShellList = ShellString.split()
        ShellList = list(map(int, ShellList))
    else:
        ShellList = Arguments['c']

    TotalNumberOfShells = len(ShellList)

    Echo("""I understand that you want %d shells with squared
velocities""" % (TotalNumberOfShells))
    Echo("%s" % ShellList)
    Echo('\n')

    # Subshell analysis
    # the initial value one corresponds to the zero velocity
    TotalNumberOfVelocities = 1
    GrandTotalList = []
    TotalListOfSubshells = []

    # Calculate cubic group for subshell analysis
    Group = GetGroup(SpacialDimension)

    for i_shell in range(TotalNumberOfShells):
        SquaredVelocity = ShellList[i_shell]
        ListOfVelocities = FindVelocities(SpacialDimension, SquaredVelocity)
        NumberOfVelocities = len(ListOfVelocities)

        if NumberOfVelocities == 0:
            EchoError("""The shell with squared velocity = %d is empty. I
assume that is not intended. Therefore I abort.""" % SquaredVelocity)
            exit(127)

        # This will let the user choose which subshells to remove from the
        # current shell and then return the possibly reduced set.
        ListOfSubshells = GetListOfSubshells(ListOfVelocities, Group)
        NumberOfSubshells = len(ListOfSubshells)
        TotalListOfSubshells.append(NumberOfSubshells)
        ListOfUsedSubshells = []
        NumberOfVelocities = 0

        if NumberOfSubshells > 1:
            Echo("Shell %d with c_i^2 = %d consists of %d subshells:" % \
                (i_shell + 1, AbsSquared(ListOfVelocities[0]),
                 NumberOfSubshells))

            for i_subs, Subshell in enumerate(ListOfSubshells):
                Echo("  Subshell %d containing %2d velocities of type %s" \
                    % (i_subs, len(Subshell), Type(Subshell)))


            if ListOfThrowawayStrings is None:
                EchoError("""Please give me the numbers of those subshells,
that you wish to EXCLUDE from the analysis in the established format: 1 2 3.
Press return to keep all subshells.""")
                ThrowawayString = input()
            else:
                ThrowawayString = ListOfThrowawayStrings[i_shell]

            # Keep all
            if ThrowawayString == '':
                Echo('  Keeping all subshells\n')
                NumberOfVelocities = len(ListOfVelocities)
                ListOfUsedSubshells = ListOfSubshells
            # Remove selected
            else:
                Echo('  Discarding subshells with indices %s\n' % ThrowawayString)
                ThrowawayList = ThrowawayString.split()
                ThrowawayList = list(map(int, ThrowawayList))
                ThrowawaySet = set(ThrowawayList)

                for j in range(NumberOfSubshells):
                    if j not in ThrowawaySet:
                        ListOfUsedSubshells.append(ListOfSubshells[j])
                        NumberOfVelocities += len(ListOfSubshells[j])
        else:
            Echo("Shell %d with c_i^2 = %d is irreducible."
                    % (i_shell + 1, AbsSquared(ListOfVelocities[0])))
            NumberOfVelocities = len(ListOfVelocities)
            ListOfUsedSubshells = [ListOfVelocities]

        GrandTotalList.extend(ListOfUsedSubshells)
        TotalNumberOfVelocities += NumberOfVelocities

    TotalNumberOfShells = len(GrandTotalList)

    # Now give the user the trivial facts about the selected model
    Echo("""Let me summarize: Your LB model comprises altogether %d
velocities in %d shells (including c_i^2 = 0 shell).""" \
        % (TotalNumberOfVelocities, TotalNumberOfShells + 1), LINEWIDTH)

    Echo("The non-trivial shells are:")
    for NumberOfShell, Shell in enumerate(GrandTotalList):
        NumberOfVelocities = len(Shell)
        Echo("  Shell number %d with c_i^2 = %2d and %2d velocities of type %s" \
            % (NumberOfShell + 1, AbsSquared(Shell[0]), NumberOfVelocities,
                Type(Shell)))

    Echo('\n')

    if Arguments['s'] is None:
        EchoError("""The procedure is based upon random vectors, therefore
please give me a start value for the random number generator""", LINEWIDTH)
        seed = int(input("Random number seed = ? "))
    else:
        seed = Arguments['s']

    Echo("Confirmation: random seed = %d" % seed)
    Echo('\n')
    random.seed(seed)

    return SpacialDimension, MaxTensorRank, ListOfTensorDimensions, \
            GrandTotalList, Arguments


# Analysis
def Analysis(SpacialDimension, MaxTensorRank, ListOfTensorDimensions,
        GrandTotalList, Arguments):
    """Performs the analysis for a given set of parameters

    Args:
        SpacialDimension (int): Spacial dimension
        MaxTensorRank (int): Maximum tensor rank :math:`M`
        ListOfTensorDimensions (list): List of the dimensions of tensor space
            for tensors of rank :math:`2,4,\\dots, M`.
        GrandTotalList (list): List of lists. The :math:`s`-th sublist
            contains all velocity vectors of shell :math:`s`.
        Arguments (dict): Dictionary of arguments as returned by function
            ParseArguments()

    Returns:
        int: Return codes\:
            - 0\:   System has unique solution
            - 1\:   System has no solution
            - 2\:   System is underdetermined and requires further examination
            - 3\:   System has unique solution but there is no physically
              valid range of existence
            - 127\: General error

    """

    Echo("Now the analysis starts ...")
    TotalNumberOfShells = len(GrandTotalList)
    ShellSizes = np.array([len(Shell) for Shell in GrandTotalList])

    LeftHandSideMatrix = FillLeftHandSide(
        SpacialDimension, MaxTensorRank, ListOfTensorDimensions,
        TotalNumberOfShells, GrandTotalList)

    # Keep in mind: This is a (NumberOfRows x TotalNumberOfShells) matrix

    RightHandSideMatrix = FillRightHandSide(
        MaxTensorRank, ListOfTensorDimensions)

    # Test given solution
    if Arguments['test']:
        Echo()
        Echo("You have chosen to test a solution.")
        return TestSolution(GrandTotalList, MaxTensorRank,
                SpacialDimension, ListOfTensorDimensions, None)

    # First do a singular-value decomposition (SVD)
    # of the left-hand side.
    # For background on SVD, see
    # https://en.wikipedia.org/wiki/Singular_value_decomposition
    # For the numpy syntax, see
    # https://docs.scipy.org/doc/numpy/reference/generated/
    # numpy.linalg.svd.html#numpy.linalg.svd

    U, s, V = np.linalg.svd(LeftHandSideMatrix, full_matrices=True)

    # -------------------------------------------------------------------------
    MatrixShape = LeftHandSideMatrix.shape
    Rows = MatrixShape[0]
    Columns = MatrixShape[1]

    # Identify very small singular values with zero
    TOL = 1.e-8
    Rank = 0
    for i, SingularValue in enumerate(s):
        if SingularValue < TOL:
            s[i] = 0.
        else:
            Rank += 1

    # -------------------------------------------------------------------------

    # U: orthogonal matrix (NumberOfRows x NumberOfRows)
    # V: orthogonal matrix (TotalNumberOfShells x TotalNumberOfShells)
    # s: stores the singular values as a 1d array
    # The actual decomposition is
    # A = U S V
    # where A = LeftHandSideMatrix
    # and S is a matrix of size (NumberOfRows x TotalNumberOfShells)
    # that contains the singular values on the diagonal
    # and is zero elsewhere

    # Move U to the right-hand side
    NewRhs = np.dot(np.transpose(U), RightHandSideMatrix)

    if Rank < Rows:
        AdditionalLines = Rows - Rank
        TestNorm = np.linalg.norm(NewRhs[-AdditionalLines:])
        TOL = 1.e-8
        if TestNorm > TOL:
            # Terminate script
            Echo("The system does not have a solution.")
            return 1
        else:
            # Prune system
            Echo("""There are %d trivial equations (0 = 0) in the system which
I shall remove for you now.""" % AdditionalLines)
            Echo('\n')
            Rows = Rank
            s.resize(Rows)
            NewRhs.resize((Rows,NewRhs.shape[1]))

    Echo("The system has at least one solution.")
    Echo('\n')

    ReducedRhs = np.zeros((Rows,NewRhs.shape[1]))
    for i, SingularValue in enumerate(s):
        for j in range(NewRhs.shape[1]):
            ReducedRhs[i,j] = NewRhs[i,j] / SingularValue

    Excess = Columns - Rows

    if Excess > 0:
        Echo("""We have %d velocity shells but only %d independent equations.
Therefore the problem has infinitely many solutions. The data will be written
to a file called 'data.npz' for further processing. If such a file already
exists it will be overwritten.\n""" % (Columns, Rows))

        if Arguments['y'] or YesNo("Is this OK? [Yn]"):

            # file output
            np.savez("data.npz", V=V, ReducedRhs=ReducedRhs,
                    NumberOfRows=Rows, ShellSizes=ShellSizes)
            Echo('\n')
            Echo("""Data has been stored. It can be processed by the secondary
script 'Continue.py'.""")
        else:
            Echo("You have chosen not to save the data.")

        return 2

    else:
        Echo("""The problem has one unique solution which is:""")

    # Now calculate the unique solution
    SolutionMatrix = np.dot(np.transpose(V), ReducedRhs)
    if not Arguments["quiet"]:
        print(SolutionMatrix)
    Echo('\n')

    # Some post-processing I:
    # Coefficients for the zero velocity shell
    NumberOfColumns = MaxTensorRank // 2

    W0List = [1.]
    for j in range(NumberOfColumns):
        MySum = 0.
        for i in range(TotalNumberOfShells):
            ListOfVelocities = GrandTotalList[i]
            NumberOfVelocities = len(ListOfVelocities)
            MySum += SolutionMatrix[i, j] * float(NumberOfVelocities)

        MySum = -MySum
        W0List.append(MySum)

    # Some post-processing II:
    # Nice output
    Echo("Coefficients, nice output:")

    TotalString = "  w[0] = 1"
    for j in range(NumberOfColumns):
        Power = 2 * (1 + j)
        PowerString = str(Power)
        CoeffString = RatApprox(W0List[j + 1])
        CoeffString = " + (" + CoeffString + ")" + " * c_s^" + PowerString
        TotalString = TotalString + CoeffString

    Echo("%s" % TotalString)

    for i in range(TotalNumberOfShells):
        index = i + 1
        TotalString = "  w[" + str(index) + "] = 0"
        for j in range(NumberOfColumns):
            Power = 2 * (1 + j)
            PowerString = str(Power)
            CoeffString = RatApprox(SolutionMatrix[i, j])
            CoeffString = " + (" + CoeffString + ")" + " * c_s^" + PowerString
            TotalString = TotalString + CoeffString

        Echo("%s" % TotalString)

    Echo('\n')

    # Some post-processing III:
    # Range of existence

    Echo("Find the range(s) of c_s^2 that yield(s) positive weights")
    CompressedRoots = FindRangeOfExistence(W0List, SolutionMatrix)
    NumberOfIntervals = OutputRangeOfExistence(CompressedRoots)
    OutputMagicNumbers(CompressedRoots, W0List, SolutionMatrix)

    if Arguments['write_latex']:
        WriteLatexTables(CompressedRoots, W0List, SolutionMatrix,
            GrandTotalList, MaxTensorRank)

    if NumberOfIntervals == 0:
        return 3
    else:
        return 0


# Run
if __name__ == '__main__':
    ExitStatus = Analysis(*GetInputData())
    Echo('\n')
    Echo("Thank you very much for using %s" % sys.argv[0])
    exit(ExitStatus)
