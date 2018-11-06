from __future__ import print_function
import random
import math
import sys
import itertools
import logging
import argparse
import numpy as np
import textwrap as tw
from fractions import Fraction

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
    parser = argparse.ArgumentParser(description="""Calculation of the
weights of an LB model.\nYou can either supply the input data interactively or
by the following command line arguments:""")

    parser.add_argument(
        "-d",
        type=int,
        help="spacial dimension of the lattice")

    parser.add_argument("-m", type=int, help="Maximum tensor rank")

    parser.add_argument(
        "-c", nargs='+', type=int,
        help="Space separated list of the radii c_i^2 of the desired "
        "velocity shells")

    parser.add_argument("-s", type=int, help="Randon number generator seed")

    parser.add_argument(
        "-y",
        action='store_true',
        help="Answer all prompts with yes (may overwrite file data.npz)")

    parser.add_argument(
        "--test", action='store_true',
        help="Test wether a given set of weights is a solution. Weights "
        "must be given as polynomials in the speed of sound.")

    parser.add_argument(
        "--quiet", action='store_true', help="Turn off most of the output")

    # if no arguments are given, print help text
    if len(sys.argv) == 1:
        parser.print_help()
        Echo('\n')


    return vars(parser.parse_args())
    


def YesNo(Question):
    """Ask for yes or no answer and return a boolean."""

    Yes = set(["YES", "Y", "yes", "y", "Yes", ""])
    No = set(["NO", "N", "no", "n", "No"])

    Question = tw.fill(Question, LINEWIDTH)

    while True:
        Answer = raw_input(Question).lower()
        if Answer in Yes:
            return True
        elif Answer in No:
            return False
        else:
            Echo("Possible answers:")
            Echo("  %s" % sorted(list(Yes)))
            Echo("or")
            Echo("  %s" % sorted(list(No)))


def Echo(String, Linewidth = LINEWIDTH):
    """Formatted printing"""
    print(tw.fill(String, Linewidth))


def EchoError(String, Linewidth = LINEWIDTH):
    """This will print irregardless of --quiet"""
    iprint(tw.fill(String, Linewidth))


def AbsSquared(Vector):
    """Return the squared absolute value of numpy array."""
    return np.sum(np.square(Vector))


def AnalyzeTensorDimension(CurrentTensorRank):
    """Recursive generation of lists that specify what types of tensors of rank
    CurrentTensorRank are compatible with cubic invariance and also fully
    symmetric under index exchange. For rank 2, these are just multiples of the
    2nd rank unit tensor delta_{ij}. Thus tensor dimension is one. For rank 4,
    these are multiples of delta_{ijkl} and multiples of (delta_{ij} delta_{kl}
    + perm.). Thus tensor dimension is two. For rank 6, we get another tensor
    delta_{ijklmn}, but also all possible products of the lower-rank deltas.
    Hence tensor dimension is three. For each new (even) rank M we get another
    delta with M indexes, plus all possible products of the lower-order delta
    tensors So, for rank two we get [[2]] (1d) for rank four [ [4] , [2,2] ]
    (2d) for rank six [ [6], [4,2], [2,2,2]] (3d) for rank eight [ [8], [6,2],
    [4,4], [4,2,2], [2,2,2,2] ] (5d) and so on.  The routine takes care of that
    "and so on". This is most easily done in a recursive fashion.
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
    """Calculates a list of lattice velocity vectors whose squared length
    matches the input value SquaredVelocity.
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
            WorkNumber = (WorkNumber - Coordinate) / LinearLatticeSize
            ShiftedCoordinate = Coordinate - SquaredVelocity
            TempVector.append(ShiftedCoordinate)
            CurrentVelocitySquared += ShiftedCoordinate ** 2

        if CurrentVelocitySquared == SquaredVelocity:
            ListOfVelocities.append(np.array(TempVector, dtype=int))

    return ListOfVelocities


def DoubleFactorial(Number):
    """Implementation of the double factorial."""
    if Number == 0 or Number == 1:
        return 1
    else:
        return Number * DoubleFactorial(Number - 2)


def MakeRandomVector(SpacialDimension):
    """Generate a random vector uniformly distributed on the unit sphere."""
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
    """Fill the matrix of left-hand sides."""
    LeftHandSideList = []

    # k loop is loop over tensor ranks
    for k in range(MaxTensorRank / 2):
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
    """Fill the matrix of right-hand sides."""
    RightHandSideList = []
    NumberOfColumns = MaxTensorRank / 2

    # k loop is loop over tensor ranks
    for k in range(MaxTensorRank / 2):
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
    ListOfWeightValues = EvaluateWeights(W0List, SolutionMatrix, CsSquared)
    MyRange = len(ListOfWeightValues)
    for i in range(MyRange):
        if(ListOfWeightValues[i] < 0.):
            return 0
    return 1


def FindRangeOfExistence(W0List, SolutionMatrix):
    """Make use of the function "roots" that needs the coefficients in reverse
    order."""

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

    NumberOfPoints = len(CompressedRoots)

    if NumberOfPoints == 0:
        Echo("No interval of validity found!")
        return 0

    if NumberOfPoints == 1:
        Echo("Interval of validity from %e to infinity" % CompressedRoots[0])
        return 1

    NumberOfIntervals = NumberOfPoints / 2
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


def EnterSolution(TotalNumberOfShells, MaxTensorRank):
    SolutionMatrix = np.zeros((TotalNumberOfShells, MaxTensorRank / 2))

    for i_shell in range(TotalNumberOfShells):
        EchoError("For shell %d please enter the coefficient of:" % (i_shell + 1))

        for i_rank in range(MaxTensorRank / 2):
            Rank = 2 * (i_rank + 1)
            SolutionMatrix[i_shell, i_rank] \
                = float(raw_input("  c_s^%d: " % Rank))

    return SolutionMatrix

# Subshell analysis

def ToMatrix(Array):
    """Convert an array of unit vectors to proper matrix."""
    Dim = len(Array)
    Matrix = np.zeros((Dim, Dim), dtype=np.int8)

    for Column in range(Dim):
        Row = Array[Column] - 1
        Matrix[Row][Column] = 1

    return Matrix


def GetGroup(SpacialDimension):
    """Compute the cubic group. Each transformation matrix in the group is made
    up of 2d unit vectors of type (0...0,+-1,0...0). We will
    identify a vector whose i-th component is 1 and 0 elsewhere by the number
    i. A vector whose i-th component is -1 and 0 elsewhere is identified by
    the number -i. 
    The cubic group then consists of all orthogonal matrices, whose columns
    consist of the above unit vectors.
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
    Echo("The group consists of %d transformations in total." % len(Group))
    assert(len(Group) == NumberOfTransformations)

    return Group


def Contains(Array, List):
    """Checks whether given numpy array is contained in list. The all()
function is defined on numpy arrays and evaluates True if all elements are
True."""

    return any((Array == X).all() for X in List)


def ContainsInSublist(Array, ListOfLists):
    return any(Contains(Array, List) for List in ListOfLists)


def ComputeSubshell(Velocity, Group):
    """Compute the (sub)shell that is being spanned by Velocity wrt. Group."""
    Subshell = []
    for i in range(len(Group)):
        New = np.dot(Group[i], Velocity)
        if not Contains(New, Subshell):
            Subshell.append(New)

    return Subshell


def GetListOfSubshells(Shell, Group):
    ListOfSubshells = []
    for Velocity in Shell:
        Subshell = ComputeSubshell(Velocity, Group)

        if not ContainsInSublist(Velocity, ListOfSubshells):
            ListOfSubshells.append(Subshell)

    return ListOfSubshells
