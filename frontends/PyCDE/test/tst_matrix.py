# RUN: %PYTHON% %s | FileCheck %s

from pycde import System, Input, Output, module, generator

from pycde.matrix import Matrix
from pycde.dialects import hw
from pycde.pycde_types import types, dim



# A [32x32x1] matrix

@module
class M1:
  in0 = Input(dim(types.i16, 16))
  in1 = Input(types.i32)
  t_c = dim(types.i32, 32)
  c = Output(t_c)

  @generator
  def build(ports):
    # a 32x32x1 matrix
    m1 = Matrix([32, 32], name='m1')

    for i in range(16):
      m1[i] = ports.in1

    v = hw.BitcastOp(M1.t_c, m1.to_circt())
    ports.c = v

top = System([M1])
top.generate()
top.print()


# A [32x1] matrix
@module
class M2:
  c = Output(types.i32)

  @generator
  def build(ports):
    m1 = Matrix([32], name='m1')
    for i in range(16):
      m1[i] = hw.ConstantOp(types.i1, i)

    m1[16 : 20] = hw.ConstantOp(types.i4, 22)
    m1[20 : 28] = hw.ConstantOp(types.i8, 111)
    m1[28 : 32] = hw.ConstantOp(types.i4, 33)

    v = hw.BitcastOp(types.i32, m1.to_circt())
    ports.c = v

top = System([M2])
top.generate()
top.print()
