#!/usr/bin/python

#
# Calculate LB model vectors and weights for a simple
# cubic lattice of arbitrary dimension
#
# The method is described in B. Duenweg's paper
# "Semi-automatic construction of Lattice Boltzmann models"
# Therefore explanations in the code are not very detailed
#

import random
import math
import numpy as np
from numpy import *

def AnalyzeTensorDimension(CurrentTensorRank):

#
# Recursive generation of lists that specify what types
# of tensors of rank CurrentTensor are compatible
# with cubic invariance and also fully symmetric under
# index exchange.
#
# For rank 2, these are just multiples of the 2nd rank
# unit tensor \delta_{ij}. Thus tensor dimension is two.
# For rank 4, these are multiples of
# \delta_{ijkl} and multiples of
# (\delta_{ij} \delta_{kl} + perm.).
# Thus tensor dimension is four.
# For rank 6, we get another tensor
# \delta_{ijklmn}, but also all possible
# products of the lower-rank deltas.
# Hence tensor dimension is three.
# For each new (even) rank M we get another
# \delta with M indexes, plus all possible
# products of the lower-order delta tensors
# So, for rank two we get [[2]]  (1d)
# for rank four [ [4] , [2,2] ]  (2d)
# for rank six [ [6], [4,2], [2,2,2]] (3d)
# for rank eight [ [8], [6,2], [4,4], [4,2,2], [2,2,2,2] ]
# (5d) and so on. The routine takes care of
# that "and so on". This is most easily done in a
# recursive fashion.
#

    if CurrentTensorRank < 2:
        print "Error: Tensor rank too small"
        exit(0)
    if CurrentTensorRank % 2 == 1:
        print "Error: Tensor rank uneven"
        exit(0)

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
        for i in range(0, TensorDimension):
            ReducedListOfArrangements = ReducedListOfPossibleTensors[i]
            if(ReducedListOfArrangements[0] <= FirstEntry):
                ListOfArrangements = FirstEntryList \
                    + ReducedListOfArrangements
                ListOfPossibleTensors.append(ListOfArrangements)

    TensorDimension = len(ListOfPossibleTensors)

    return TensorDimension, ListOfPossibleTensors


def FindVelocities(SpacialDimension, SquaredVelocity):

#    
# Calculates a list of lattice velocity vectors
# whose squared length matches the input value
# SquaredVelocity
#

# The list to be returned at the end of the routine
    ListOfVelocities = []

# number of lattice sites to be scanned
    linear_lattice_size = 2 * SquaredVelocity + 1
    full_lattice_size = linear_lattice_size ** SpacialDimension

    for site in range(0, full_lattice_size):
        work_number = site
        current_velocity_squared = 0
        temp_vector = []
        for dim in range(0, SpacialDimension):
            coordinate = work_number % linear_lattice_size
            work_number = (work_number - coordinate) / linear_lattice_size
            shifted_coordinate = coordinate - SquaredVelocity
            temp_vector.append(shifted_coordinate)
            current_velocity_squared += shifted_coordinate ** 2
        if current_velocity_squared == SquaredVelocity:
            ListOfVelocities.append(temp_vector)

    return ListOfVelocities


def DoubleFactorial(number):
    if number == 0 or number == 1:
        return 1
    else:
        return number * DoubleFactorial(number - 2)


def MakeRandomVector(SpacialDimension):
#
# Generate a random vector uniformly distributed
# on the unit sphere
#
    my_sum = 2.
    while my_sum > 1.:
        my_sum = 0.
        RandomVector = []
        for dim in range(0, SpacialDimension):
            random_number = 2. * random.random() - 1.
            RandomVector.append(random_number)
            my_sum += random_number ** 2
    factor = 1. / math.sqrt(my_sum)
    for dim in range(0, SpacialDimension):
        RandomVector[dim] *= factor

    return RandomVector


def LatticeSum(RandomVector, ListOfVelocities, TensorRank):
    SpacialDimension = len(RandomVector)
    NumberOfVelocities = len(ListOfVelocities)
    my_sum = 0.
    for velocity in range(0, NumberOfVelocities):
        VelocityVector = ListOfVelocities[velocity]
        scalar_product = 0.
        for dim in range(0, SpacialDimension):
            scalar_product += RandomVector[dim] * VelocityVector[dim]
        scalar_product = scalar_product ** TensorRank
        my_sum += scalar_product
    my_sum = my_sum / float(DoubleFactorial(TensorRank - 1))

    return my_sum


def ContFrac(x):
#
# Calculates the continued-fraction expansion
# of the floating-point number x
# See https://en.wikipedia.org/wiki/Continued_fraction
# Here we assume that x is of order unity and positive!
# Very small contributions are truncated
# because they are believed to be a result
# of roundoff errors!
#
    TOL_ROUNDOFF = 1.e-9
    TOL_TRUNCATE = 1.e-6
    MAX_ITER = 10
    ListOfIntegers = []
    if x <= 0.:
        print "Error in the continued-fraction expansion!"
        print "Please supply a positive number!"
        exit(0)
    IntegerContribution = int(x + TOL_ROUNDOFF)
    ListOfIntegers.append(IntegerContribution)
    Rest = x - float(IntegerContribution)
    NIter = 0
    while Rest > TOL_TRUNCATE and NIter < MAX_ITER :
        Inverse = 1. / Rest
        IntegerNumber = int(Inverse + TOL_ROUNDOFF)
        ListOfIntegers.append(IntegerNumber)
        Rest = Inverse - float(IntegerNumber)
        NIter += 1
        
    return ListOfIntegers


def NumeratorOfConvergent(n, ListOfIntegers):
#
# calculates numerator of a fraction from a
# continued-fraction expansion
#
    if n == 0:
        return ListOfIntegers[0]
    if n == 1:
        return ListOfIntegers[0] * ListOfIntegers[1] + 1
    return ListOfIntegers[n] \
           * NumeratorOfConvergent(n - 1, ListOfIntegers) \
           + NumeratorOfConvergent(n - 2, ListOfIntegers)


def DenominatorOfConvergent(n, ListOfIntegers):
#
# calculates denominator of a fraction from a
# continued-fraction expansion
#
    if n == 0:
        return 1
    if n == 1:
        return ListOfIntegers[1]
    return ListOfIntegers[n] \
           * DenominatorOfConvergent(n - 1, ListOfIntegers) \
           + DenominatorOfConvergent(n - 2, ListOfIntegers)


def RatApprox(x):
#
# Calculates numerator and denominator for a
# floating point number x
# and returns the output as a string
#
    TOL = 1.e-7
    y = abs(x)
    if y < TOL:
        return "0"
    sign = 1
    if x < 0.:
        sign = -1
    ListOfIntegers = ContFrac(y)
    DepthOfExpansion = len(ListOfIntegers) - 1
    Numerator = \
        NumeratorOfConvergent(DepthOfExpansion, ListOfIntegers)    
    Denominator = \
        DenominatorOfConvergent(DepthOfExpansion, ListOfIntegers)    
    Numerator *= sign
    NumeratorString = str(Numerator)
    if Denominator == 1:
        return NumeratorString
    DenominatorString = str(Denominator)
    TotalString = NumeratorString+"/"+DenominatorString

    return TotalString


def EvaluateWeights(W0List, SolutionMatrix, cs2):
    ListOfWeightValues = []
    my_sum = 0.
    my_range = len(W0List)
    for i in range(0, my_range):
        my_sum += W0List[i] * (cs2 ** i)
    ListOfWeightValues.append(my_sum)
    NumberOfRows = np.size(SolutionMatrix,0)
    NumberOfColumns = np.size(SolutionMatrix,1)
    for i in range(0, NumberOfRows):
        my_sum = 0.
        for j in range(0, NumberOfColumns):
            my_sum += SolutionMatrix[i,j] * (cs2 ** (j + 1))
        ListOfWeightValues.append(my_sum)

    return ListOfWeightValues


def IndicatorFunction(W0List, SolutionMatrix, cs2):
    ListOfWeightValues = EvaluateWeights(W0List, SolutionMatrix, cs2)
    my_range = len(ListOfWeightValues)
    for i in range(0, my_range):
        if(ListOfWeightValues[i] < 0.):
            return 0
    return 1


def FindRangeOfExistence(W0List, SolutionMatrix):
#
# make use of the function "roots"
# that needs the coefficients in
# reverse order
#
    TOL = 1.e-10
    TotalRoots = []

    temp_list = []
    my_length = len(W0List)
    for i in range(0, my_length):
        temp_list.append(W0List[my_length - 1 - i])
    temp_array = np.array(temp_list)
    my_complex_roots = np.roots(temp_array)
    NumberOfRoots = len(my_complex_roots)
    for i in range(0, NumberOfRoots):
        current_root = my_complex_roots[i]
        if abs(current_root.imag) < TOL and \
           current_root.real > TOL:
            TotalRoots.append(current_root.real)
            
    NumberOfRows = np.size(SolutionMatrix,0)
    NumberOfColumns = np.size(SolutionMatrix,1)
    for j in range(0, NumberOfRows):
        temp_list = []
        for i in range(0, NumberOfColumns):
            temp_list.append(SolutionMatrix[j, \
                             NumberOfColumns - 1 - i])
        temp_array = np.array(temp_list)
        my_complex_roots = np.roots(temp_array)
        NumberOfRoots = len(my_complex_roots)
        for i in range(0, NumberOfRoots):
            current_root = my_complex_roots[i]
            if abs(current_root.imag) < TOL and \
               current_root.real > TOL:
                TotalRoots.append(current_root.real)

    TotalRoots.sort()

    TotalNumberOfRoots = len(TotalRoots)

    dummy_root = TotalRoots[TotalNumberOfRoots - 1] * 2.
    TotalRoots.append(dummy_root)
    TotalNumberOfRoots += 1
    
    CompressedRoots = []
    
    cs2 = 0.5 * TotalRoots[0]
    oldtest = IndicatorFunction(W0List, SolutionMatrix, cs2)
    if oldtest == 1:
        CompressedRoots.append(0.)
        
    for i in range(0, TotalNumberOfRoots - 1):
        cs2 = 0.5 * (TotalRoots[i] + TotalRoots[i + 1])
        test = IndicatorFunction(W0List, SolutionMatrix, cs2)
        if test != oldtest:
            CompressedRoots.append(TotalRoots[i])
        oldtest = test

    return CompressedRoots


def OutputRangeOfExistence(CompressedRoots):
        
    NumberOfPoints = len(CompressedRoots)
    
    if NumberOfPoints == 0:
        print "No interval of validity found!"
        return

    if NumberOfPoints == 1:
        print "Interval of validity from %e to infinity" \
            % CompressedRoots[0]
        return
            
    NumberOfIntervals = NumberOfPoints / 2
    Rest = NumberOfPoints - NumberOfIntervals * 2
    for i in range(0, NumberOfIntervals):
        print "Interval of validity %d : %e, %e" \
            % (i + 1, CompressedRoots[2 * i], \
               CompressedRoots[2 * i + 1])
    if Rest == 1:
        print "Interval of validity %d : %e, infinity" \
            % (NumberOfIntervals + 1, \
               CompressedRoots[NumberOfPoints - 1])

    return


def OutputMagicNumbers(CompressedRoots, W0List, SolutionMatrix):
    my_range = len(CompressedRoots)

    if len == 0:
        return

    print "The limits of validity are:"

    for i in range(0, my_range):
        cs2 = CompressedRoots[i]
        String = RatApprox(cs2)
        print "c_s^2 = %e =(possibly) %s" % (cs2, String)
    
    for i in range(0, my_range):
        cs2 = CompressedRoots[i]
        print "Reduced model at c_s^2 = %e:" % cs2
        ListOfWeightValues = EvaluateWeights(W0List, SolutionMatrix, cs2)
        weight_range = len(ListOfWeightValues)
        for j in range(0, weight_range):
            my_weight = ListOfWeightValues[j]
            String = RatApprox(my_weight)
            print "w[%d] = %e =(possibly) %s" % (j, my_weight, String)

    return


def FillLeftHandSide(SpacialDimension, MaxTensorRank, \
                         ListOfTensorDimensions, \
                         TotalNumberOfShells, GrandTotalList):

# Fill the matrix of left-hand sides

    left_hand_side_list = []

# k loop is loop over tensor ranks
    for k in range(0, MaxTensorRank / 2):
        TensorRank = 2 * k + 2
        LocalDimensionOfTensorSpace = ListOfTensorDimensions[k]
# j loop is loop over random vectors
        for j in range (0, LocalDimensionOfTensorSpace):
            RandomVector = MakeRandomVector(SpacialDimension)
            RowList = []
# i loop is loop over velocity shells
            for i in range(0, TotalNumberOfShells):
                ListOfVelocities = GrandTotalList[i]
                ShellSum = LatticeSum(RandomVector,ListOfVelocities,TensorRank)
                RowList.append(ShellSum)
            left_hand_side_list.append(RowList)

    left_hand_side_matrix = np.array(left_hand_side_list)
    return left_hand_side_matrix


def FillRightHandSide(MaxTensorRank, ListOfTensorDimensions):

# Fill the matrix of right-hand sides

    right_hand_side_list = []

    NumberOfColumns = MaxTensorRank / 2

# k loop is loop over tensor ranks
    for k in range(0, MaxTensorRank / 2):
        TensorRank = 2 * k + 2
        LocalDimensionOfTensorSpace = ListOfTensorDimensions[k]
# j loop is loop over random vectors
        for j in range (0, LocalDimensionOfTensorSpace):
            RowList = []
# i loop is loop over c_s powers
            for i in range(0, NumberOfColumns):
                cs_power = 2 * i + 2
                element = 0.
                if cs_power == TensorRank:
                    element = 1.
                RowList.append(element)
            right_hand_side_list.append(RowList)

    right_hand_side_matrix = np.array(right_hand_side_list)
    return right_hand_side_matrix



# BEGINNING OF MAIN PROGRAM

# First get input data from the user

print "Calculation of the weights of an LB model."

print " "

print "First I need to know in which spacial dimension"
print "the LB model shall live."
print "Please note that the model will live on a"
print "simple cubic lattice."
SpacialDimension = int(raw_input("Spacial dimension = ? "))
print "Confirmation: spacial dimension = %d" % SpacialDimension

print " "

print "Now please tell me up to which tensor rank"
print "you wish to satisfy the Maxwell-Boltzmann constraints"
print "(for example, 2nd rank, 4th rank, etc.)."
print "Please note that this should be an even number."
MaxTensorRank = int(raw_input("Maximum tensor rank = ? "))
print "Confirmation: maximum tensor rank = %d" % MaxTensorRank

print " "

DimensionOfTensorSpace = 0
ListOfTensorDimensions = []

for k in range(0, MaxTensorRank/2):
    CurrentTensorRank = 2 * k + 2
    TensorDimension, ListOfPossibleTensors = \
        AnalyzeTensorDimension(CurrentTensorRank)
    ListOfTensorDimensions.append(TensorDimension)
    DimensionOfTensorSpace += TensorDimension

print "I expect that you need %d velocity shells" \
      % DimensionOfTensorSpace
print "plus the zero velocity shell, which we do need to"
print "consider explicitly."
print "Perhaps, however, you can get away with less - just try!"

print " "

print "Please give me the squared lengths of the velocity"
print "shells that you wish to analyze"
print "(excluding the zero velocity shell)."
print "in the simple format: 1 2 3 4 5"

ShellString = raw_input()
ShellList = ShellString.split()
ShellList = map(int, ShellList)
TotalNumberOfShells = len(ShellList)
print "I understand that you want %d shells" % TotalNumberOfShells
print "with squared velocities", ShellList

if TotalNumberOfShells > DimensionOfTensorSpace:
    print "These are too many shells!"
    print "This would results in a singular problem!"
    print "Aborting the procedure."
    exit(0)
    
# the initial value one corresponds to the zero velocity
TotalNumberOfVelocities = 1

GrandTotalList = []

for i in range(0, TotalNumberOfShells):
    SquaredVelocity = ShellList[i]
    ListOfVelocities = FindVelocities(SpacialDimension, SquaredVelocity)
    NumberOfVelocities = len(ListOfVelocities)
    if NumberOfVelocities == 0:
        print "The shell with squared velocity = %d is empty" \
            % SquaredVelocity
        print "I assume that is not intended. Therefore I abort."
        exit(0)
    GrandTotalList.append(ListOfVelocities)
    TotalNumberOfVelocities += NumberOfVelocities

print "The procedure is based upon random vectors,"
print "therefore please give me a start value for"
print "the random number generator"
seed = int(raw_input("Random number seed = ? "))
print "Confirmation: random seed = %d" % seed
random.seed(seed)

# At this point, the input is done!

print " "

# First give the user the trivial facts about
# the selected model
    
print "Let me summarize: Your LB model comprises"
print "altogether %d velocities (including zero)." \
      % TotalNumberOfVelocities
print "The non-trivial shells are:"
for NumberOfShell in range(0, TotalNumberOfShells):
    print "Shell number %d :" % (NumberOfShell + 1)
    ListOfVelocities = GrandTotalList[NumberOfShell]
    NumberOfVelocities = len(ListOfVelocities)
    print "comprises %d lattice vectors:" % NumberOfVelocities
    print ListOfVelocities

print " "

print "Now the analysis starts ..."

left_hand_side_matrix = FillLeftHandSide(SpacialDimension, \
                                             MaxTensorRank,
                                             ListOfTensorDimensions, \
                                             TotalNumberOfShells, \
                                             GrandTotalList)

# Keep in mind: This is a (NumberOfRows x TotalNumberOfShells) matrix

NumberOfRows = left_hand_side_matrix.shape[0]

right_hand_side_matrix = FillRightHandSide(MaxTensorRank, \
                                               ListOfTensorDimensions)

# First do a singular-value decomposition (SVD)
# of the left-hand side.
# For background on SVD, see
# https://en.wikipedia.org/wiki/Singular_value_decomposition
# For the numpy syntax, see
# https://docs.scipy.org/doc/numpy/reference/generated/
# numpy.linalg.svd.html#numpy.linalg.svd

U, s, V = np.linalg.svd(left_hand_side_matrix, full_matrices=True)

# U: orthogonal matrix (NumberOfRows x NumberOfRows)
# V: orthogonal matrix (TotalNumberOfShells x TotalNumberOfShells)
# s: stores the singular values as a 1d array
# The actual decomposition is
# A = U S V
# where A = left_hand_side_matrix
# and S is a matrix of size (NumberOfRows x TotalNumberOfShells)
# that contains the singular values on the diagonal
# and is zero elsewhere

# Move U to the right-hand side

new_rhs = np.dot(np.transpose(U), right_hand_side_matrix)

# Identify very small singular values with zero
# and use this for checking the rank

NumberOfSingularValues = s.size

ListOfSingValTags = []

ProductOfTags = 1

TOL = 1.e-8
for i in range(0, NumberOfSingularValues):
    if s[i] > TOL:
        Tag = 1
    else:
        Tag = 0
    ListOfSingValTags.append(Tag)
    ProductOfTags *= Tag

if ProductOfTags == 0:
    RankDeficiency = True
else:
    RankDeficiency = False


# Define a projection operator Q that projects
# onto the null space of SV

ProjectorQ = np.zeros( (NumberOfRows, NumberOfRows) )

for i in range(0, NumberOfRows):
    if i < NumberOfSingularValues and ListOfSingValTags[i] == 1:
        ProjectorQ[i,i] = 0.
    else:
        ProjectorQ[i,i] = 1.

# Apply this to the rhs

projected_rhs = np.dot(ProjectorQ, new_rhs)

# The system has a solution if and only if this is just zero

test_rhs = np.linalg.norm(projected_rhs)
TOL = 1.e-6
if test_rhs < TOL:
    NoSolution = False
else:
    NoSolution = True

# Eliminate all undesired cases

if RankDeficiency:
    print "This is a rank-deficient problem"
    if NoSolution:
        print "with no solution whatsoever"
    else:
        print "with infinitely many solutions."
    print "Aborting."
    exit(0)
else:
    print "Matrix has maximum rank"
    if NoSolution:
        print "but the system has no solution."
        print "Aborting."
        exit(0)
    else:
        print "and the problem has one unique solution ..."

# Now calculate the unique solution

ScaleMatrix = np.zeros( (TotalNumberOfShells, NumberOfRows) )

for i in range(0, NumberOfSingularValues):
    ScaleMatrix[i,i] = 1. / s[i]

reduced_rhs = np.dot(ScaleMatrix, new_rhs)
    
SolutionMatrix = np.dot(np.transpose(V), reduced_rhs)

print "... which is:"
print SolutionMatrix

print " "

# Some post-processing I:
# Coefficients for the zero velocity shell

NumberOfColumns = MaxTensorRank / 2

W0List = [1.]
for j in range(0, NumberOfColumns):
    my_sum = 0.
    for i in range(0, TotalNumberOfShells):
        ListOfVelocities = GrandTotalList[i]
        NumberOfVelocities = len(ListOfVelocities)
        my_sum += SolutionMatrix[i,j] * float(NumberOfVelocities)
    my_sum = -my_sum
    W0List.append(my_sum)

# Some post-processing II:
# Nice output

print "Coefficients, nice output:"

TotalString = "w[0] = 1"
for j in range(0, NumberOfColumns):
    power = 2 * (1 + j)
    PowerString = str(power)
    CoeffString = RatApprox(W0List[j + 1])
    CoeffString = " + (" + CoeffString + ")" + " * c_s^" + PowerString
    TotalString = TotalString + CoeffString
print TotalString

for i in range(0, TotalNumberOfShells):
    index = i + 1
    TotalString = "w[" + str(index) + "] = 0"
    for j in range(0, NumberOfColumns):
        power = 2 * (1 + j)
        PowerString = str(power)
        CoeffString = RatApprox(SolutionMatrix[i,j])
        CoeffString = " + (" + CoeffString + ")" + " * c_s^" + PowerString
        TotalString = TotalString + CoeffString
    print TotalString

print " "

# Some post-processing III:
# Range of existence

print "Find the range(s) of c_s^2"
print "that yields positive weights"
CompressedRoots = FindRangeOfExistence(W0List, SolutionMatrix)
OutputRangeOfExistence(CompressedRoots)
OutputMagicNumbers(CompressedRoots, W0List, SolutionMatrix)

exit(0)
