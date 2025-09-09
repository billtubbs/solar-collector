"""Simple 1D transient heat conduction example solved 
with Pyomo DAE solver.

Copied from J C Kantor's Pyomo documentation:
 - https://jckantor.github.io/ND-Pyomo-Cookbook/notebooks/
   05.03-Heat_Conduction_in_Various_Geometries.html
"""

import shutil
import os

import numpy as np
import matplotlib.pyplot as plt
from pyomo.dae import DerivativeVar, ContinuousSet
from pyomo.environ import (
    Var,
    Constraint,
    Objective,
    SolverFactory,
    ConcreteModel,
    TransformationFactory
)

assert shutil.which("ipopt") or os.path.isfile("ipopt")


def model_plot(model, figsize=(10, 6)):
    tgrid, rgrid = np.meshgrid(model.t, model.r, indexing='ij')
    Tgrid = np.array([
        model.T[t.item(), r.item()].value
        for t, r in np.nditer([tgrid, rgrid])
    ]).reshape(tgrid.shape)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlabel('Distance r')
    ax.set_ylabel('Time t')
    ax.set_zlabel('Temperature T')
    ax.plot_wireframe(rgrid, tgrid, Tgrid)

    return ax


model = ConcreteModel()
model.r = ContinuousSet(bounds=(0, 1))
model.t = ContinuousSet(bounds=(0, 2))
model.T = Var(model.t, model.r)
model.dTdt = DerivativeVar(model.T, wrt=model.t)
model.dTdr = DerivativeVar(model.T, wrt=model.r)
model.d2Tdr2 = DerivativeVar(model.T, wrt=(model.r, model.r))


@model.Constraint(model.t, model.r)
def pde(m, t, r):
    if t == 0:
        return Constraint.Skip
    if r == 0 or r == 1:
        return Constraint.Skip
    return m.dTdt[t, r] == m.d2Tdr2[t, r]


model.obj = Objective(expr=1)

model.ic = Constraint(
    model.r,
    rule=lambda m,
    r: m.T[0, r] == 0 if r > 0 and r < 1 else Constraint.Skip
)
model.bc1 = Constraint(model.t, rule=lambda m, t: m.T[t, 1] == 1)
model.bc2 = Constraint(model.t, rule=lambda m, t: m.dTdr[t, 0] == 0)

TransformationFactory('dae.finite_difference').apply_to(
    model, nfe=50, scheme='FORWARD', wrt=model.r
)
TransformationFactory('dae.finite_difference').apply_to(
    model, nfe=50, scheme='FORWARD', wrt=model.t
)
SolverFactory('ipopt').solve(model, tee=True).write()

model_plot(model)
plt.tight_layout()
plt.show()
