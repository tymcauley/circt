#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .value import (BitVectorValue, ListValue, get_slice_idxs)
from .pycde_types import types, dim
from .support import get_user_loc
from pycde.dialects import hw, sv
import math


def is_divisor(n, d):
  return d % n == 0


def greatest_divisor(number):
  for i in range(2, int(math.sqrt(number))):
    if is_divisor(i, number):
      return int(number / i)
  return 1


def splice(value, at):
  l = len(value)
  lower_type = hw.ArrayType.get(value.type.element_type, at)
  upper_type = hw.ArrayType.get(value.type.element_type, l - at)
  lower_slice = hw.ArraySliceOp(value, 0, lower_type)
  upper_slice = hw.ArraySliceOp(value, at, upper_type)
  return lower_slice, upper_slice


class Matrix:

  store = {}
  # The SSA value of this matrix after it has been materialized.
  # Once set, the matrix is immutable.
  circt_output = None

  class Range:
    """A range class representing a 1-dimensional continuous range.
       The .end field is exclusive.
    """

    def __init__(self, start, end):
      self.start = start
      self.end = end

      if start == 0 and end == 0:
        raise ValueError("Null-ranges not allowed")

      if start > end:
        raise ValueError("Start must be less than end")

    def overlaps(self, other):
      return self.start < other.end and other.start < self.end

    def __hash__(self) -> int:
      return hash((self.start, self.end))

    def __eq__(self, other) -> bool:
      return self.start == other.start and self.end == other.end

    def __lt__(self, other) -> bool:
      return self.start < other.start

    def __str__(self) -> str:
      return f"[{self.start}:{self.end}]"

    def __len__(self) -> int:
      return self.end - self.start

  def get_indexed_type(self, range):
    # Returns the type of this assignment where the first dimension
    # is based on the provided range.
    sizes = [range.end - range.start]
    return dim(self.nested_type, *sizes)

  def is_assigned(v) -> bool:
    return v != None

  def assign(self, range, value):
    """ Assign a value to a range of the matrix. If the range does not
      cover the entirety of the matrix, the matrix will be split in two
      and the value will be assigned to the appropriate sub-matrix.
      The provided value must be type-compatible with the type generated
      by the provided range."""
    if range.start < 0 or range.end > self.shape[0]:
      raise ValueError("Range must be within the bounds of the matrix.")

    if not isinstance(value, ListValue):
      raise ValueError("Value must be a ListValue.")

    # Ensure this is a unique, non-overlapping assignment. Furthermore,
    # there can only exist one unassigned overlapping range.
    overlapping_range = None
    for r, v in self.store.items():
      if r.overlaps(range):
        overlapping_range = r
        if v is not None:
          raise ValueError(f"Assignment range {range} overlaps "
                           f"with existing assignment to range {r}")
    assert overlapping_range is not None

    # Split the current overlapping range into unassigned pre- and post-ranges
    del self.store[overlapping_range]
    if range.start > overlapping_range.start:
      pre_range = Matrix.Range(0, range.start)
      self.store[pre_range] = None

    if range.end < overlapping_range.end:
      post_range = Matrix.Range(range.end, overlapping_range.end)
      self.store[post_range] = None

    # Ensure that the inner range is compatible with the provided value,
    # and assign.
    inner_range = Matrix.Range(range.start, range.end)
    index_type = self.get_indexed_type(inner_range)
    if index_type != value.type:
      raise ValueError(
          f"Assignment value type {value.type} is incompatible with "
          f"the assigned range type {index_type}")
    self.store[inner_range] = value

  def __init__(self, shape: list, inner_width: int = 1, name:str = None, default_driver=None) -> None:
    """Construct a matrix with the given shape and dtype.
      Args:
        shape: A tuple of integers representing the shape of the matrix.
        inner: An integer representing the bitwidth of matrix values.
                This defaults to 1.
        
        The semantics of the inner dimension are such that the following two matrices
        will be identical:
          pycde.Matrix(2,3,4, inner=32)
          pycde.Matrix(2,3,4, 32, inner=1)
      """
    self.name = name
    self.shape: list = shape
    self.inner_width: int = inner_width

    if inner_width < 1:
      raise ValueError("Inner dimension must be at least 1.")


    # Determine the nested type of the matrix.
    # e.g. 
    #   matrix type         nested type
    #   16x32xinner_width   32xinner_width
    #   32xinner_width      inner_width
    if len(shape) > 1:
      self.nested_type = dim(types.int(inner_width), *shape[1:])
    else:
      self.nested_type = types.int(inner_width)

    # some sanity checking
    if len(shape) == 0:
      raise ValueError("Shape must have at least one dimension.")

    for s in shape:
      if s < 1:
        raise ValueError("Shape dimensions must be > 0.")

    # Initialize the store with null in the outer dimension
    self.store[Matrix.Range(0, shape[0])] = None

  @property
  def bits(self):
    n = self.inner
    for dim in self.shape:
      n *= dim
    return n

  def __setitem__(self, idxOrSlices, value):
    if self.circt_output is not None:
      raise ValueError("Cannot assign to a materialized matrix.")

    if not isinstance(idxOrSlices, tuple):
      idxOrSlices = [idxOrSlices]

    if len(idxOrSlices) > len(self.shape):
      raise ValueError("Too many indices for matrix.")

    d0_lo, d0_hi = get_slice_idxs(self.shape[0], idxOrSlices[0])

    # The matrix implementation is anchored around a hw.array of bits implementation.
    # Perform any necessary convertions to the value to comply with this.
    if isinstance(value, BitVectorValue):
      # BitVectors should be converted to a ListValue through a bitcast.
      sizes = [len(value)]
      if sizes[0] != 1:
        sizes.append(1)
      value = hw.BitcastOp(dim(types.int(self.inner_width), *sizes), value)

    if not isinstance(value, ListValue):
      raise ValueError("Incompatible value type for matrix assignment.")

    self.assign(Matrix.Range(d0_lo, d0_hi), value)
    return

  def validate_assignments(self):
    """ Checks that all sub-matrices have been fully assigned. """
    for range, value in self.store.items():
      try:
        if value is None:
          raise ValueError(f" is unassigned")
        if isinstance(value, Matrix):
          if not value.validate_assignments():
            return False
      except ValueError as e:
        raise ValueError(f"{range}{e}")

  def to_circt(self, create_wire=True):
    """Materializes this matrix to CIRCT array_create operations.
    
    if 'create_wire' is True, the matrix will be materialized to an sv.wire operation
    and the returned value will be a read-only reference to the wire.
    This wire acts as a barrier in CIRCT to prevent dataflow optimizations
    from reordering/optimizing the materialization of the matrix, which might
    reduce debugability.
    """
    if self.circt_output:
      return self.circt_output

    try:
      self.validate_assignments()
    except ValueError as e:
      raise ValueError(f"{self.name}{e}")

    sorted_store = [self.store[r] for r in sorted(list(self.store))]
    materialized_matrix = self.splicemerge_assignments(self.shape[0],
                                                     sorted_store)
    self.circt_output = materialized_matrix
    if create_wire:
      wire = sv.WireOp(materialized_matrix.type, f"{self.name}_wire")
      sv.AssignOp(wire, materialized_matrix)
      self.circt_output = wire.read

    return self.circt_output

  def splicemerge_assignments(self, outer_shape, assignments):
    """ During assignment, a mixture of array dimensions may have been created.
    array_get requires that all values are of the same dimension - as such,
    we iteratively materialize the values such that we eventually reach a
    set of values with identical type, which can constitute the output value.
    In the general case, assignments to matrices tend to be regular, and as such,
    assignment splitting through this method will produce minimal code.

    Given outmost dim = dim, whenever
    * All dimensions are equal, we can use array_get directly.
    * dim is not prime - we materialize the matrix by selectively
      splicing and merging the assignments based on the assignment which
      represents the largest divisor of the dim.  
    * dim is a prime, we have to flatten the assignments to x1's
    """
    # The set of values which we will merge into a single array
    elements = []
    # The # of elements in 'assignments'
    assignments_len = sum([len(a) for a in assignments])

    # When all ranges are of equal width we can use array_get.
    all_ranges_eq_width = all([
        len(assignment) == len(assignments[0]) for assignment in assignments[1:]
    ])
    if assignments_len == outer_shape and all_ranges_eq_width:
      with get_user_loc():
        merged_arr = hw.ArrayCreateOp(assignments)
        # We need to cast the array to the correct type of the requested
        # outer shape.
        # e.g. if self.nested_type is i1 and outer_shape is 8,
        # merged_arr might be <2x<4xi1>> but should be interpreted as <8xi1>
        cast_arr = hw.BitcastOp(dim(self.nested_type, outer_shape), merged_arr)
        return cast_arr
    else:
      # We have to splice and merge.
      # We start by finding the largest divisor of the outer shape. This will
      # be the target width of values in 'elements'.
      divisor = 1
      for assignment in assignments:
        l = len(assignment)
        if is_divisor(l, outer_shape) and l > divisor:
          divisor = l

      # Iterate through the assignments, gather up towards the divisor and
      # splice the differences.
      # 'assignments' is now used as a worklist.
      subelements = []
      subelements_width = 0
      while len(assignments) > 0:
        # Get the next element.
        assignment = assignments[0]
        assignments = assignments[1:]
        subelements_width += len(assignment)
        subelements.append(assignment)
        diff = divisor - subelements_width

        if diff == 0:
          # We've gathered enough elements to reach the divisor size.
          elem = self.splicemerge_assignments(divisor, subelements)
          elements.append(elem)
          subelements = []
          subelements_width = 0
        elif diff < 0:
          # We've gathered too many elements, we need to splice the last element.
          lastv = subelements.pop()
          subelements_width -= len(lastv)
          lhs, rhs = splice(lastv, -diff)
          assignments = [lhs, rhs] + assignments

      if len(subelements) > 0:
        raise ValueError("Leftover elements during assignment splicemerging.")

    return hw.ArrayCreateOp(elements)
