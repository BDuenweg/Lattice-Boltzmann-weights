"""Collection of helper functions for LBweights.py and Continue.py

Attributes:
    LINEWIDTH (int): Line width for console output.
    QUIET: Flag to suppress standard output.

"""

from __future__ import print_function
import random
import math
import os
import sys
import itertools
import logging
import argparse
import numpy as np
import textwrap as tw
from decimal import Decimal
from fractions import Fraction

# use raw_input instead of input when python2 is used
try:
    input = raw_input
except NameError:
    pass

# set print options
LINEWIDTH = 70
np.set_printoptions(linewidth=LINEWIDTH)

QUIET = "--quiet" in sys.argv
logging.basicConfig(level=logging.WARNING if QUIET else
        logging.INFO, format="%(message)s")

print = logging.info
iprint = logging.warning

# Helper functions
def ParseArguments():
    """Function to parse command line options.

    Returns:
        dict: Dictionary of command line options

    """

    parser = argparse.ArgumentParser(description="""Calculation of the
weights of an LB model.\nYou can either supply the input data interactively or
by the following command line arguments:""")

    parser.add_argument(
        "-d", type=int,
        help="spacial dimension of the lattice")

    parser.add_argument("-m", type=int, help="Maximum tensor rank")

    parser.add_argument(
        "-c", nargs='+', type=int,
        help="Space separated list of the radii c_i^2 of the desired "
        "velocity shells")

    parser.add_argument("-s", type=int, help="Random number generator seed")

    parser.add_argument(
        "-y", action='store_true',
        help="Answer all prompts with yes (may overwrite file data.npz)")

    parser.add_argument(
        "--test", action='store_true',
        help="Test, whether a set of weights that can be written as a linear "
        "parametric equation "
        " w = w_0 + lambda_1 w_1 + lambda_2 w_2 "
        "solves the equation A.w == b for given speed of sound. "
        "Weights and speed of sound are entered interactively by the user."
        )

    parser.add_argument(
        "--quiet", action='store_true', help="Turn off most of the output")

    parser.add_argument(
        "--write-latex", action='store_true',
        help="Write unique solution to the file \"latex_tables.dat\" in form "
        "of a latex table. This will append to any existing file.")

    # if no arguments are given, print help text
    if len(sys.argv) == 1:
        parser.print_help()
        Echo('\n')


    return vars(parser.parse_args())


def YesNo(Question):
    """Ask for yes or no answer and return a Boolean.

    Args:
        Question (str): String that is printed when function is called.

    Returns:
        bool: True, if answer is in ``["YES", "Y", "yes", "y", "Yes", \CR]``
        False, if answer is in ``["NO", "N", "no", "n", "No"]``

    """

    Yes = set(["YES", "Y", "yes", "y", "Yes", ""])
    No = set(["NO", "N", "no", "n", "No"])

    Question = tw.fill(Question, LINEWIDTH)

    while True:
        Answer = input(Question).lower()
        if Answer in Yes:
            return True
        elif Answer in No:
            return False
        else:
            Echo("Possible answers:")
            Echo("  %s" % sorted(list(Yes)))
            Echo("or")
            Echo("  %s" % sorted(list(No)))


def Echo(String="\n", Linewidth = LINEWIDTH):
    """Formatted printing
    If QUIET is set (i.e. via command line option --quiet) this is suppressed.

    Args:
        String (str): String to be printed to the console
        Linewidth (int): Maximum line width of console output
    Returns:
        None

    """

    print(tw.fill(String, Linewidth))


def EchoError(String="\n", Linewidth = LINEWIDTH):
    """Formatted printing
    Prints irregardless of value of QUIET

    Args:
        String (str): String to be printed to the console
        Linewidth (int): Maximum line width of console output
    Returns:
        None

    """

    iprint(tw.fill(String, Linewidth))


def AbsSquared(Vector):
    """Return the squared absolute value of numpy array.

    Args:
        Vector (numpy.ndarray): Vector that is supposed to be squared
    Returns:
        Return the squared absolute value of Vector

    """

    return np.sum(np.square(Vector))


def AnalyzeTensorDimension(CurrentTensorRank):
    """Recursive generation of lists that specify what types of tensors of rank
    CurrentTensorRank are compatible with cubic invariance and also fully
    symmetric under index exchange. For rank 2, these are just multiples of the
    2nd rank unit tensor :math:`\\delta_{ij}`. Thus tensor dimension is one.
    For rank 4, these are multiples of :math:`\\delta_{ijkl}` and multiples of
    :math:`(\\delta_{ij}`, :math:`\\delta_{kl} + \\textrm{perm.})`. Thus tensor
    dimension is two. For rank 6, we get another tensor
    :math:`\\delta_{ijklmn}`, but also all possible products of the lower-rank
    deltas.
    Hence tensor dimension is three. For each new (even) rank M we get another
    delta with M indexes, plus all possible products of the lower-order delta
    tensors So, for rank two we get ``[[2]]`` (1d) for rank four
    ``[[4], [2,2]]`` (2d) for rank six ``[[6], [4,2], [2,2,2]]`` (3d) for rank
    eight ``[[8], [6,2], [4,4], [4,2,2], [2,2,2,2]]`` (5d) and so on.
    The routine takes care of that "and so on". This is most easily done in a
    recursive fashion.

    Args:
        CurrentTensorRank (int): Tensor rank

    Returns:
        int: Dimension of tensor space
        list: List compatible tensors

    """

    if CurrentTensorRank < 2:
        EchoError("Error: Tensor rank too small")
        exit(127)

    if CurrentTensorRank % 2 == 1:
        EchoError("Error: Tensor rank uneven")
        exit(127)

    FirstEntry = CurrentTensorRank
    FirstEntryList = []
    FirstEntryList.append(FirstEntry)

    ListOfPossibleTensors = []
    ListOfPossibleTensors.append(FirstEntryList)

    if FirstEntry == 2:
        return 1, ListOfPossibleTensors

    while FirstEntry > 2:
        FirstEntry = FirstEntry - 2
        FirstEntryList = []
        FirstEntryList.append(FirstEntry)
        Rest = CurrentTensorRank - FirstEntry
        TensorDimension, ReducedListOfPossibleTensors = \
            AnalyzeTensorDimension(Rest)
        for i in range(TensorDimension):
            ReducedListOfArrangements = ReducedListOfPossibleTensors[i]

            if(ReducedListOfArrangements[0] <= FirstEntry):
                ListOfArrangements = FirstEntryList + ReducedListOfArrangements
                ListOfPossibleTensors.append(ListOfArrangements)

    TensorDimension = len(ListOfPossibleTensors)

    return TensorDimension, ListOfPossibleTensors


def FindVelocities(SpacialDimension, SquaredVelocity):
    """Scans the cubic lattice for lattice velocity with squared length
    SquaredVelocity

    Args:
        SpacialDimension (int): SpacialDimension
        SquaredVelocity (int): Squared length of compatible lattice velocities

    Returns:
        list: List of compatible lattice velocity vectors

    """

    # The list to be returned at the end of the routine
    ListOfVelocities = []

    # number of lattice sites to be scanned
    LinearLatticeSize = 2 * SquaredVelocity + 1
    FullLatticeSize = LinearLatticeSize ** SpacialDimension

    for Site in range(FullLatticeSize):
        WorkNumber = Site
        CurrentVelocitySquared = 0
        TempVector = []

        for dim in range(SpacialDimension):
            Coordinate = WorkNumber % LinearLatticeSize
            WorkNumber = (WorkNumber - Coordinate) // LinearLatticeSize
            ShiftedCoordinate = Coordinate - SquaredVelocity
            TempVector.append(ShiftedCoordinate)
            CurrentVelocitySquared += ShiftedCoordinate ** 2

        if CurrentVelocitySquared == SquaredVelocity:
            ListOfVelocities.append(np.array(TempVector, dtype=int))

    return ListOfVelocities


def DoubleFactorial(Number):
    """Implementation of the double factorial.
        :math:`n!! = n(n-2)(n-4)\\dots`

    Args:
        Number (int): Number

    Returns:
        int: Number :math:`!!`

    """

    if Number == 0 or Number == 1:
        return 1
    else:
        return Number * DoubleFactorial(Number - 2)


def MakeRandomVector(SpacialDimension):
    """Generate a random vector uniformly distributed on the unit sphere.

    Args:
        SpacialDimension (int): Spacial dimension d

    Returns:
        list: Vector of length one with random orientation in d-dimensional
        space.

    """

    MySum = 2.
    while MySum > 1.:
        MySum = 0.
        RandomVector = []
        for dim in range(SpacialDimension):
            RandomNumber = 2. * random.random() - 1.
            RandomVector.append(RandomNumber)
            MySum += RandomNumber ** 2

    Factor = 1. / math.sqrt(MySum)
    for dim in range(SpacialDimension):
        RandomVector[dim] *= Factor

    return RandomVector


def LatticeSum(RandomVector, ListOfVelocities, TensorRank):
    """Calculate the sum
    :math:`A_{rs} = \\frac{1}{(m_r-1)!!} \sum_{i \in s} (\\vec{c}_i\cdot\\vec{n}_r)^{m_r}`

    for tensor rank :math:`r` and shell :math:`s`.

    Args:
        RandomVector (numpy.ndarray): :math:`r`-th random unit vector
        ListOfVelocities (list): List of velocity vectors in shell :math:`s`
        TensorRank (int): Tensor rank :math:`r`

    Returns:
        float: :math:`A_{rs}`

    """

    SpacialDimension = len(RandomVector)
    NumberOfVelocities = len(ListOfVelocities)
    MySum = 0.
    for Velocity in range(NumberOfVelocities):
        VelocityVector = ListOfVelocities[Velocity]
        ScalarProduct = 0.
        for dim in range(SpacialDimension):
            ScalarProduct += RandomVector[dim] * VelocityVector[dim]

        ScalarProduct = ScalarProduct ** TensorRank
        MySum += ScalarProduct

    MySum = MySum / float(DoubleFactorial(TensorRank - 1))

    return MySum


def FillLeftHandSide(SpacialDimension, MaxTensorRank, ListOfTensorDimensions,
                     TotalNumberOfShells, GrandTotalList):
    """Construct the :math:`R \\times N_s` matrix :math:`A`

    Args:
        SpacialDimension (int): Spacial dimension
        MaxTensorRank (int): Highest tensor rank (M) to consider.
        ListOfTensorDimensions (list): List of the dimensions of tensor space
            for tensors of rank :math:`2,4,\\dots, M`.
        TotalNumberOfShells (int): Total number of velocity shells :math:`N_s`
        GrandTotalList (list): List of lists. The :math:`s`-th sublist
            contains all velocity vectors of shell :math:`s`.

    Returns:
        numpy.ndarray: Matrix :math:`A`

    """

    LeftHandSideList = []

    # k loop is loop over tensor ranks
    for k in range(MaxTensorRank // 2):
        TensorRank = 2 * k + 2
        LocalDimensionOfTensorSpace = ListOfTensorDimensions[k]

        # j loop is loop over random vectors
        for j in range(LocalDimensionOfTensorSpace):
            RandomVector = MakeRandomVector(SpacialDimension)
            RowList = []

            # i loop is loop over velocity shells
            for i in range(TotalNumberOfShells):
                ListOfVelocities = GrandTotalList[i]
                ShellSum = LatticeSum(
                    RandomVector, ListOfVelocities, TensorRank)
                RowList.append(ShellSum)

            LeftHandSideList.append(RowList)

    LeftHandSideMatrix = np.array(LeftHandSideList)
    return np.array(LeftHandSideMatrix)


def FillRightHandSide(MaxTensorRank, ListOfTensorDimensions):
    """Construct the matrix :math:`D: D_{r\\mu} = \delta_{m_r \\mu}`

    Args:
        MaxTensorRank (int): Maximum tensor rank :math:`M`
        ListOfTensorDimensions (list): List of the dimensions of tensor space
            for tensors of rank :math:`2,4,\\dots, M`.

    Returns:
        numpy.ndarray: Matrix :math:`D`

    """

    RightHandSideList = []
    NumberOfColumns = MaxTensorRank // 2

    # k loop is loop over tensor ranks
    for k in range(MaxTensorRank // 2):
        TensorRank = 2 * k + 2
        LocalDimensionOfTensorSpace = ListOfTensorDimensions[k]

        # j loop is loop over random vectors
        for j in range(LocalDimensionOfTensorSpace):
            RowList = []

            # i loop is loop over c_s powers
            for i in range(NumberOfColumns):
                CsPower = 2 * i + 2
                Element = 0.
                if CsPower == TensorRank:
                    Element = 1.

                RowList.append(Element)

            RightHandSideList.append(RowList)

    RightHandSideMatrix = np.array(RightHandSideList)
    return RightHandSideMatrix


def RatApprox(x):
    """Calculates numerator and denominator for a floating point number x and
    returns the output as a string.

    Args:
        x (float): Number to approximate as fraction.

    Returns:
        str: Approximate fraction as string

    """

    TOL = 1.e-12
    if abs(x) < TOL:
        return "0"

    if (abs(x) >= 0.1):
        LimitDenominator = 1000000
    else:
        LimitDenominator = int(1. / abs(x)) * 1000000

    MyFraction = Fraction(x).limit_denominator(LimitDenominator)
    MyFraction = str(MyFraction)

    return MyFraction


def EvaluateWeights(W0List, SolutionMatrix, CsSquared):
    """Calculate numerical weights from their polynomial coefficients

    Args:
        W0List (list): List of polynomial coefficients for zero shell
        SolutionMatrix (numpy.ndarray): Solution matrix :math:`Q`
        CsSquared (float): Speed of sound squared

    Returns:
        list: List of numerical weights :math:`[w_0, w_1, \dots]`

    """

    ListOfWeightValues = []
    MySum = 0.
    MyRange = len(W0List)
    for i in range(MyRange):
        MySum += W0List[i] * (CsSquared ** i)

    ListOfWeightValues.append(MySum)
    NumberOfRows = np.size(SolutionMatrix, 0)
    NumberOfColumns = np.size(SolutionMatrix, 1)
    for i in range(NumberOfRows):
        MySum = 0.
        for j in range(NumberOfColumns):
            MySum += SolutionMatrix[i, j] * (CsSquared ** (j + 1))

        ListOfWeightValues.append(MySum)

    return ListOfWeightValues


def IndicatorFunction(W0List, SolutionMatrix, CsSquared):
    """Tests, whether solution yields all positive weights.

    Args:
        W0List (list): List of polynomial coefficients for zero shell
        SolutionMatrix (numpy.ndarray): Solution matrix :math:`Q`
        CsSquared (float): Speed of sound squared

    Returns:
        bool: True if all weights positive, False otherwise

    """

    ListOfWeightValues = EvaluateWeights(W0List, SolutionMatrix, CsSquared)
    MyRange = len(ListOfWeightValues)
    for i in range(MyRange):
        if(ListOfWeightValues[i] < 0.):
            return 0
    return 1


def FindRangeOfExistence(W0List, SolutionMatrix):
    """Make use of the function "roots" that needs the coefficients in reverse
    order, in order to find the roots of the weight polynomials. If inbetween
    two roots all weights are positive, add them to list CompressedRoots

    Args:
        W0List (list): List of polynomial coefficients for zero shell
        SolutionMatrix (numpy.ndarray): Solution matrix :math:`Q`

    Returns:
        list: List CompressedRoots of roots that form valid intervals for
        the speed of sound.

    """

    TOL = 1.e-10
    TotalRoots = []

    temp_list = []
    my_length = len(W0List)
    for i in range(my_length):
        temp_list.append(W0List[my_length - 1 - i])

    temp_array = np.array(temp_list)
    my_complex_roots = np.roots(temp_array)
    NumberOfRoots = len(my_complex_roots)
    for i in range(NumberOfRoots):
        current_root = my_complex_roots[i]
        if abs(current_root.imag) < TOL and current_root.real > TOL:
            TotalRoots.append(current_root.real)

    NumberOfRows = np.size(SolutionMatrix, 0)
    NumberOfColumns = np.size(SolutionMatrix, 1)
    for j in range(NumberOfRows):
        temp_list = []
        for i in range(NumberOfColumns):
            temp_list.append(SolutionMatrix[j, NumberOfColumns - 1 - i])

        temp_array = np.array(temp_list)
        my_complex_roots = np.roots(temp_array)
        NumberOfRoots = len(my_complex_roots)
        for i in range(NumberOfRoots):
            current_root = my_complex_roots[i]
            if abs(current_root.imag) < TOL and current_root.real > TOL:
                TotalRoots.append(current_root.real)

    TotalRoots.sort()

    TotalNumberOfRoots = len(TotalRoots)
    if TotalNumberOfRoots == 0:
        return []

    DummyRoot = TotalRoots[TotalNumberOfRoots - 1] * 2.
    TotalRoots.append(DummyRoot)
    TotalNumberOfRoots += 1

    CompressedRoots = []

    CsSquared = 0.5 * TotalRoots[0]
    Oldtest = IndicatorFunction(W0List, SolutionMatrix, CsSquared)

    if Oldtest == 1:
        CompressedRoots.append(0.)

    for i in range(TotalNumberOfRoots - 1):
        CsSquared = 0.5 * (TotalRoots[i] + TotalRoots[i + 1])
        Test = IndicatorFunction(W0List, SolutionMatrix, CsSquared)
        if Test != Oldtest:
            CompressedRoots.append(TotalRoots[i])

        Oldtest = Test

    return CompressedRoots


def OutputRangeOfExistence(CompressedRoots):
    """Screen output of the intervals of the speed of sound that yield all
    positive weights.

    Args:
        CompressedRoots (list): List of roots that form valid intervals for
            the speed of sound.

    Returns:
        int: Number of valid intervals

    """

    NumberOfPoints = len(CompressedRoots)

    if NumberOfPoints == 0:
        Echo("No interval of validity found!")
        return 0

    if NumberOfPoints == 1:
        Echo("Interval of validity from %e to infinity" % CompressedRoots[0])
        return 1

    NumberOfIntervals = NumberOfPoints // 2
    Rest = NumberOfPoints - NumberOfIntervals * 2
    for i in range(NumberOfIntervals):
        Echo("Interval of validity %d : %e, %e" \
                % (i + 1, CompressedRoots[2 * i], CompressedRoots[2 * i + 1]))

    if Rest == 1:
        Echo("Interval of validity %d : %e, infinity" % \
            (NumberOfIntervals + 1, CompressedRoots[NumberOfPoints - 1]))

    return NumberOfIntervals


def OutputMagicNumbers(CompressedRoots, W0List, SolutionMatrix):
    MyRange = len(CompressedRoots)
    """Prints the valid intervals to the console and shows the reduced models
    at the borders of the intervals.

    Args:
        CompressedRoots (list): List of roots that form valid intervals for the
        speed of sound.
        W0List (list): List of polynomial coefficients for zero shell
        SolutionMatrix (numpy.ndarray): Solution matrix :math:`Q`

    Returns:
        None

    """

    if MyRange == 0:
        return

    Echo("\nThe limits of validity are:")

    for i in range(MyRange):
        CsSquared = CompressedRoots[i]
        String = RatApprox(CsSquared)
        Echo("  c_s^2 = %e = (possibly) %s" % (CsSquared, String))

    for i in range(MyRange):
        CsSquared = CompressedRoots[i]
        Echo("\nReduced model at c_s^2 = %e:" % CsSquared)
        ListOfWeightValues = EvaluateWeights(W0List, SolutionMatrix, CsSquared)
        WeightRange = len(ListOfWeightValues)
        for j in range(WeightRange):
            MyWeight = ListOfWeightValues[j]
            String = RatApprox(MyWeight)
            Echo("  w[%d] = %e = (possibly) %s" % (j, MyWeight, String))

    return


def Frexp10(Float):
    """Returns exponent and mantissa in base 10

    Args:
        Float (float): Original number

    Returns:
        tuple: ``(Mantissa, Exponent)``

    """

    if Float == 0:
        return (0, 0)

    (Sign, Digits, Exponent) = Decimal(Float).as_tuple()
    Exponent = len(Digits) + Exponent - 1
    Mantissa = Float/10**Exponent

    assert(abs(Mantissa * 10**Exponent - Float) < 1e-12)
    assert(1. <= abs(Mantissa) <= 10.)

    return (Mantissa, Exponent)


def Type(Shell):
    """Method to determine typical velocity vector for Shell.

    Args:
        Shell (list): List of velocity vectors, e.g. ``[[0,1],[1,0]]``

    Returns:
        tuple: Typical velocity vector, e.g. ``(0,1)``

    """

    return tuple(np.sort(list(map(abs, Shell[0]))))


def WriteLatexNumber(Value, Outfile, Precision = 8, Rational = False):
    """Write Value to ``Outfile`` in a Latex compatible way

    Args:
        Value (float): Value
        Outfile: Output file
        Precision (int): Number of digits
        Rational (bool): Approximate numbers by fractions

    Returns:
        None

    """

    if Rational:
        Outfile.write("$%10s$" % RatApprox(Value))

    else:
        if(abs(Value) < 10**(-14)):
            Outfile.write("$0$" + (Precision + 18) * " ")

        else:
            Mantissa, Exponent = Frexp10(Value)
            if(Exponent == 0):
                Outfile.write("$%*.*f$               " % (Precision + 4,
                    Precision, Value))
            else:
                Outfile.write("$%*.*f \\times 10^{%d}$" % (Precision + 4, Precision,
                    Mantissa, Exponent))


def WriteLatexTables(CompressedRoots, W0List, SolutionMatrix, GrandTotalList,
        MaxTensorRank, Precision = 8, Rational = False,
        Filename="latex_tables.tex"):
    """Write unique solution to a file in form of a latex table. This will
    append to any existing file.

    Args:
        CompressedRoots (list): List of roots that form the valid intervals for
            the speed of sound
        W0List (list): List of polynomial coefficients for zero shell
        SolutionMatrix (numpy.ndarray): Solution matrix :math:`Q`
        GrandTotalList (list): List of lists. The :math:`s`-th sublist
            contains all velocity vectors of shell :math:`s`.
        MaxTensorRank (int): Maximum tensor rank :math:`M`
        Precision (int): Number of digits
        Rational (bool): Approximate numbers by fractions

    Returns:
        None

    """

    if os.path.isfile(Filename):
        if not "-y" in sys.argv:
            if not YesNo("Latex table will be appended to the file "+Filename+
                ". Is this OK? [Yn]"):
                return

    with open(Filename, "a") as Outfile:
        MyRange = len(CompressedRoots)
        SpacialDimension = len(GrandTotalList[0][0])
        # include zero shell
        GrandTotalList = [[np.zeros(SpacialDimension, dtype=int)]] + GrandTotalList
        TotalNumberOfVelocities = 0
        for Shell in GrandTotalList:
            for Velocity in Shell:
                TotalNumberOfVelocities += 1

        if MyRange == 0:
            return

        Outfile.write("\\documentclass{article}\n")
        Outfile.write("\\begin{document}\n")
        Outfile.write("\\begin{table}\n")
        Outfile.write("  \\begin{tabular}{| c | c |")

        for i in range(MyRange):
            Outfile.write(" c |")

        Outfile.write("}\n")
        Outfile.write("    \\hline\n")
        Outfile.write("    shell & typical")
        for i in range(MyRange):
            Outfile.write(" & weight at $c_\\mathrm{s}^2 = $")

        Outfile.write("\\\\\n")
        Outfile.write("    size  & vector  ")

        # print c_s^2 at interval boundary
        for CsSquared in CompressedRoots:
            Mantissa, Exponent = Frexp10(CsSquared)

            Value = CsSquared
            Outfile.write("& ")
            WriteLatexNumber(Value, Outfile, Precision, Rational)
            Outfile.write(" ")

        Outfile.write("\\\\\n")
        Outfile.write("    \hline\n")

        WeightLists = []
        for CsSquared in CompressedRoots:
            WeightLists += [EvaluateWeights(W0List, SolutionMatrix, CsSquared)]

        for i_shell in range(len(GrandTotalList)):
            Shell = GrandTotalList[i_shell]
            # write number of velocities in shell
            Outfile.write("    %3d & (" % len(Shell))

            # write type of shell
            for i_type in range(SpacialDimension):
                if i_type < SpacialDimension - 1:
                    Outfile.write("%d, " % Type(Shell)[i_type])
                else:
                    Outfile.write("%d) " % Type(Shell)[i_type])

            # write weights at interval edges
            for i_cs in range(MyRange):
                Value = WeightLists[i_cs][i_shell]
                Outfile.write("& ")
                WriteLatexNumber(Value, Outfile, Precision, Rational)
                Outfile.write(" ")


            Outfile.write("\\\\")
            Outfile.write("\n")

        Outfile.write("    \hline\n")
        Outfile.write("  \\end{tabular}\n")
        Outfile.write("  \\caption{Properties of a %d--speed model in %d dimensions that is isotropic up to tensor rank %d.}\n" % (TotalNumberOfVelocities, SpacialDimension, MaxTensorRank))
        Outfile.write("  \\label{tab:weights_d%dm%dv%d}\n" % (SpacialDimension,
            MaxTensorRank, TotalNumberOfVelocities))
        Outfile.write("\\end{table}\n")
        Outfile.write("\n")

        # Write Coefficients
        (Rows, Columns) = SolutionMatrix.shape
        ZerothColumn  = np.eye(Rows + 1, 1)

        Outfile.write("\\begin{table}\n")
        Outfile.write("  \\begin{tabular}{| c | c |")
        for i_col in range(Columns + 1):
            Outfile.write(" c |")

        Outfile.write("}")
        Outfile.write("\\\\\n")

        Outfile.write("    \\hline\n")

        Outfile.write("    shell & typical  ")
        for i_col in range(Columns + 1):
            Outfile.write("& coefficient of    ")

        Outfile.write("\\\\\n")

        Outfile.write("    size  & vector  ")
        for i_col in range(Columns + 1):
            Outfile.write(" & $c_\mathrm{s}^%d$ " % (2*i_col))

        Outfile.write("\\\\\n")
        Outfile.write("    \\hline\n")

        for i_shell in range(len(GrandTotalList)):
            Shell = GrandTotalList[i_shell]
            # write number of velocities in shell
            Outfile.write("    %3d & (" % len(Shell))

            # write type of shell
            for i_type in range(SpacialDimension):
                if i_type < SpacialDimension - 1:
                    Outfile.write("%d, " % Type(Shell)[i_type])
                else:
                    Outfile.write("%d) " % Type(Shell)[i_type])

            # write zeroth shell
            if i_shell == 0:
                Outfile.write("& 1 ")
                for i_col in range(Columns):
                    Value = W0List[i_col + 1]
                    Outfile.write("& ")
                    WriteLatexNumber(Value, Outfile, Precision, Rational)
                    Outfile.write(" ")
            # write remaining shells
            else:
                # write zeroth coefficient
                Outfile.write("& 0 " % ZerothColumn[i_shell])
                # write other coefficients
                for i_col in range(Columns):
                    Value = SolutionMatrix[i_shell - 1][i_col]
                    Outfile.write("& ")
                    WriteLatexNumber(Value, Outfile, Precision, Rational)
                    Outfile.write(" ")

            Outfile.write("\\\\\n")

        Outfile.write("    \hline\n")
        Outfile.write("  \\end{tabular}\n")
        Outfile.write("  \\caption{Coefficients of a %d--speed model in %d dimensions that is isotropic up to tensor rank %d.}\n" % (TotalNumberOfVelocities, SpacialDimension, MaxTensorRank))
        Outfile.write("  \\label{tab:coefficients_d%dm%dv%d}\n" %
                (SpacialDimension, MaxTensorRank, TotalNumberOfVelocities))
        Outfile.write("\\end{table}\n")
        Outfile.write("\\end{document}\n")
        return


def EnterWeights(TotalNumberOfShells, i_par=0):
    """Gets vector of weights from user input

    Args:
        TotalNumberOfShells (int): Number of shells INCLUDING zero-shell
        i_par (int): Solution vector index (parametric solutions are written as
            :math:`\\vec{w} = \\vec{w}_0 + \\lambda_1 \\vec{w}_1 + \\lambda_2 \\vec{w}_2 + \\dots`)

    Returns:
        numpy.ndarray: Vector of weights

    """

    EchoError("Please enter the weights w_%di:" % i_par)
    WTemp = []
    for i_shell in range(TotalNumberOfShells + 1):
        Val = float(input("  w_%d%d = " % (i_par, i_shell)))
        WTemp.append(Val)

    return np.array(WTemp)


def CloseEnough(A, W, B, M, RelTol=1e-5):
    """Test the condition

    .. math::

        \\left\\lvert \\sum_j A_{ij} w_j - b_i \\right\\rvert &<
        \\varepsilon \\sqrt{\\textstyle{\\sum_j} \\left(A_{ij} w_j\\right)^2
        + \\left(\\frac{m_i}{2}\\right)^2 b_i}
        \\text{ for all } i

    Args:
        A (numpy.ndarray): Matrix :math:`A`
        W (numpy.ndarray): Vector :math:`\\vec{w}`
        B (numpy.ndarray): Vector :math:`\\vec{b},~b_i = c_\\mathrm{s}^{m_i}`
        M (numpy.ndarray): Vector :math:`\\vec{m}`
        RelTol (float): Relative tolerance :math:`\\varepsilon`

    Returns:
        bool: True if condition satisfied, False otherwise.

    """

    assert(np.all(B >= 0))
    assert(np.all(M >= 0))

    Delta = np.abs(A.dot(W) - B)
    Sum = np.array([AbsSquared(np.multiply(A_i, W)) for A_i in A])

    Thresh = RelTol * np.sqrt(Sum + np.square(0.5*np.multiply(M,B)))

    return np.all(Delta < Thresh)


def TestSolution(GrandTotalList, MaxTensorRank, SpacialDimension,
        ListOfTensorDimensions, Solution=None, RelTol=1e-5):
    """Test validity of the equation :math:`A\\vec{w} = \\vec{b}` for given
    weights w and speed of sound :math:`c_s^2`.
    A solution is deemed valid, if

    .. math::

        \\left\\lvert \\sum_j A_{ij} w_j - b_i \\right\\rvert
        &< \\varepsilon_0
        + \\varepsilon \\sqrt{\\left(\\textstyle{\\sum_j} A_{ij} w_j\\right)^2
        + \\left(\\frac{m_i}{2}\\right)^2 b_i}
        \\text{ for all } i

    The weights can be given as a linear parametric equation

    .. math::

        \\vec{w} = \\vec{w}_0 + \lambda_1 \\vec{w}_1 + \lambda_2 \\vec{w}_2 +
        \\dots

    Args:
        GrandTotalList (list): List of lists. The :math:`s`-th sublist
            contains all velocity vectors of shell :math:`s`.
        MaxTensorRank (int): Maximum tensor rank :math:`M`
        SpacialDimension (int): SpacialDimension
        ListOfTensorDimensions (list): List of the dimensions of tensor space
            for tensors of rank :math:`2,4,\\dots, M`.
        Solution (list): Solution that is to be tested in the form
            ``[CsSquared, [[w_00, w_01,...], [[w_10, w_11, ...], ...]``
            If None is given, the user is prompted to enter a solution by hand.
        RelTol (float): Relative tolerance :math:`\\varepsilon`

    Returns:
        int: 0 if solution is valid, otherwise 1

    """

    ShellSizes = np.array([1] + [len(Shell) for Shell in GrandTotalList])
    TotalNumberOfShells = len(GrandTotalList) # NOT including zero shell!
    # Type solution by hand
    if Solution is None:
        # Input speed of sound
        EchoError("Please enter the speed of sound squared: ")
        CsSquared = float(input("c_s^2 = "))
        assert(CsSquared >= 0)

        ListOfWeightVectors = []

        # Input w_0
        i_par = 0
        WTemp = EnterWeights(TotalNumberOfShells, i_par)
        ListOfWeightVectors.append(WTemp)

        # Input w_i, i > 0
        while YesNo("Do you want to add further solution vectors for a parametric solution? [Yn]"):
            i_par += 1
            WTemp = EnterWeights(TotalNumberOfShells, i_par)
            ListOfWeightVectors.append(WTemp)

        Solution = [CsSquared, ListOfWeightVectors]
        Echo()

    A = FillLeftHandSide(
        SpacialDimension, MaxTensorRank, ListOfTensorDimensions,
        TotalNumberOfShells, GrandTotalList)


    # set B, M
    CsSquared = Solution[0]
    B = []
    M = []
    for k in range(MaxTensorRank // 2):
        # TensorRank = 2 * k + 2
        LocalDimensionOfTensorSpace = ListOfTensorDimensions[k]

        for _ in range(LocalDimensionOfTensorSpace):
            B.append(CsSquared**(k+1))
            M.append(2 * k + 2)

    B = np.array(B)
    M = np.array(M)

    # test
    for i_W, W in enumerate(Solution[1]):
        assert(len(W) == TotalNumberOfShells + 1)
        if i_W == 0:
            CNorm = 1.
            C = B
        else:
            CNorm = 0.
            C = np.zeros(len(B))

        # first, test normalization condition
        if abs(ShellSizes.dot(W) - CNorm) >= RelTol:
            EchoError()
            EchoError("The weights w_%d do not satisfy normalization condition!" % i_W)
            return 1
        # test A.w == b
        if not CloseEnough(A, W[1:], C, M, RelTol):
            EchoError()
            EchoError("The given solution does NOT solve the system, solution vector w_%d is not compatible." % i_W)
            EchoError('A.w_%d%s = ' % (i_W, '-b' if i_W == 0 else ''))
            EchoError(str(A.dot(W[1:]) - C))
            return 1

    Echo("The given solution solves the system.")
    return 0


# Subshell analysis
def ToMatrix(Array):
    """Convert an array of unit vector representations to proper matrix.
    For example
    ``[0,2,1]`` will be converted to
    ``[[1,0,0], [0,0,1], [0,1,0]]``.

    Args:
        Array (numpy.ndarray): Array of integers

    Returns:
        numpy.ndarray: Transformation matrix

    """

    Dim = len(Array)
    Matrix = np.zeros((Dim, Dim), dtype=np.int8)

    for Column in range(Dim):
        Row = Array[Column] - 1
        Matrix[Row][Column] = 1

    return Matrix


def GetGroup(SpacialDimension):
    """Compute the cubic group. Each transformation matrix in the group is made
    up of 2d unit vectors of type :math:`(0\\dots0,+-1,0\\dots0)`.
    We will identify a vector with i-th component 1 and 0 elsewhere by the
    number :math:`i`. A vector with :math:`i`-th component -1 and 0 elsewhere
    is identified by the number :math:`-i`.
    The cubic group then consists of all orthogonal matrices, with columns
    made up of the above unit vectors.
    In general there are :math:`d! 2^d` such transformations.

    Args:
        SpacialDimension (int): Spacial dimension

    Returns:
        list: A list of all transformation matrices in the cubic group

    """

    Echo("Computing cubic group in %d dimensions..." % SpacialDimension)
    if SpacialDimension > 7:
        Echo("This might take a while...")

    # Build list of all unit vectors
    UnitVectors = [i + 1 for i in range(SpacialDimension)]

    UVCombinations = itertools.permutations(UnitVectors, SpacialDimension)
    # the possible combinations of signs are equivalent to the binary
    # representations of the numbers 0 to (SpacialDimension - 1).
    SignCombinations = [None]*pow(2, SpacialDimension)
    Format="0" + str(SpacialDimension) + "b"

    for i in range(pow(2, SpacialDimension)):
        Binary = format(i, Format)
        Combination = tuple(2*int(Binary[j]) - 1
                for j in range(SpacialDimension))
        SignCombinations[i] = Combination

    # apply all possible sign combinations
    Group = []
    for CombinationUV in UVCombinations:
        Matrix = ToMatrix(CombinationUV)
        for CombinationS in SignCombinations:
            Group.append(np.multiply(CombinationS, Matrix))

    NumberOfTransformations = \
        math.factorial(SpacialDimension) * pow(2, SpacialDimension)
    assert(len(Group) == NumberOfTransformations)
    Echo("The group consists of %d transformations in total." % len(Group))

    return Group


def Contains(Array, List):
    """Checks whether given numpy array is contained in list. The all()
    function is defined on numpy arrays and evaluates True if all elements are
    True.

    Args:
        Array (numpy.ndarray): numpy array
        List (list): List of numpy arrays

    Returns:
        bool: True if Array is contained in List, False otherwise.

    """

    return any((Array == X).all() for X in List)


def ContainsInSublist(Array, ListOfLists):
    """Checks whether given numpy array is contained in a list of lists.
    The all() function is defined on numpy arrays and evaluates True if all
    elements are True.

    Args:
        Array (numpy.ndarray): numpy array
        List (list): List of Lists of numpy arrays

    Returns:
        bool: True if Array is contained in ListOfLists, False otherwise.

    """

    return any(Contains(Array, List) for List in ListOfLists)


def ComputeSubshell(Velocity, Group):
    """Compute the (sub)shell that is being spanned by Velocity wrt. Group.

    Args:
        Velocity (numpy.ndarray): Velocity vector
        Group (list): List of transformation matrices that form the cubic group

    Returns:
        list: List of velocity vectors that form the velocity shell spanned by
        Group

    """

    Subshell = []
    for i in range(len(Group)):
        New = np.dot(Group[i], Velocity)
        if not Contains(New, Subshell):
            Subshell.append(New)

    return Subshell


def GetListOfSubshells(Shell, Group):
    """Applies all group transformations to all velocities in shell and returns
    all distinct shells that result.

    Args:
        Shell (list): List of velocity vectors
        Group (list): List of transformation matrices that form the cubic group

    Returns:
        list: List of distinct velocity shells

    """

    ListOfSubshells = []
    for Velocity in Shell:
        Subshell = ComputeSubshell(Velocity, Group)

        if not ContainsInSublist(Velocity, ListOfSubshells):
            ListOfSubshells.append(Subshell)

    return ListOfSubshells
