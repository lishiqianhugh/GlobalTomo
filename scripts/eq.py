from modulus.sym.eq.pde import PDE
from sympy import Symbol, Function

class FreeSurface(PDE):
    def __init__(self, u="u"):
        # set params
        self.u = u
        # coordinates
        x, y = Symbol("x"), Symbol("y")
        z = Symbol("z")
        input_variables = {"x": x, "y": y, "z": z}
        # Scalar function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # set equations
        self.equations = {}
        self.equations["free_surface"] = (
            u.diff(x, 2)
            + u.diff(y, 2)
            + u.diff(z, 2)
        )


class WaveEquation(PDE):
    def __init__(self, u="u"):
        # set params
        self.u = u

        # coordinates
        vs, vp = Symbol("vs"), Symbol("vp")
        x, z, t = Symbol("x"), Symbol("z"), Symbol("t")
        y = Symbol("y")
        input_variables = {"x": x, "y": y, "z": z, "t": t, "vp": vp}
        # Scalar function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)

        # set equations
        self.equations = {}
        self.equations["wave_equation"] = (
            u.diff(t, 2)
            - vp**2 * u.diff(x, 2)
            - vp**2 * u.diff(y, 2)
            - vp**2 * u.diff(z, 2)
        )
            
class LameFreeSurface(PDE):
    def __init__(self, ux, uy, uz, dim=3):
        # coordinates
        x, y, z, t, vs, vp = (
            Symbol("x"),
            Symbol("y"),
            Symbol("z"),
            Symbol("t"),
            Symbol("vs"),
            Symbol("vp"),
        )

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t, "vs": vs, "vp": vp}

        mu = vs**2  # 3/4*(vp_r ** 2 - K)  # shear modulus
        K = vp**2 - 4 / 3 * mu  # bulk modulus

        c1 = K - 2 / 3 * mu
        c2 = K + 4 / 3 * mu

        # define the unit vector
        n1 = x
        n2 = y
        n3 = z

        ux = Function(ux)(*input_variables)
        uy = Function(uy)(*input_variables)
        uz = Function(uz)(*input_variables)

        # set equations
        self.equations = {}

        self.equations["LameFreeSurfacex"] = (
            n1 * (c1 * uz.diff(z) + c1 * uy.diff(y) + c2 * ux.diff(x))
            + n2 * (mu * ux.diff(y) + mu * uy.diff(x))
            + n3 * (mu * ux.diff(z) + mu * uz.diff(x))
        )
        self.equations["LameFreeSurfacey"] = (
            n3 * (mu * uy.diff(z) + mu * uz.diff(y))
            + n2 * (c1 * uz.diff(z) + c2 * uy.diff(y) + c1 * ux.diff(x))
            + n1 * (mu * ux.diff(y) + mu * uy.diff(x))
        )
        self.equations["LameFreeSurfacez"] = (
            n2 * (mu * uy.diff(z) + mu * uz.diff(y))
            + n3 * (c2 * uz.diff(z) + c1 * uy.diff(y) + c1 * ux.diff(x))
            + n1 * (mu * ux.diff(z) + mu * uz.diff(x))
        )


class LameEquation(PDE):
    def __init__(self, ux, uy, uz, dim=3):
        # coordinates
        x, y, z, t = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("t")

        vs, vp = Symbol("vs"), Symbol("vp")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z, "t": t, "vs": vs, "vp": vp}

        ux = Function(ux)(*input_variables)
        uy = Function(uy)(*input_variables)
        uz = Function(uz)(*input_variables)

        tc0 = uz.diff(y) - uy.diff(z)
        tc1 = ux.diff(z) - uz.diff(x)
        tc2 = uy.diff(x) - ux.diff(y)

        out0 = tc2.diff(y) - tc1.diff(z)
        out1 = tc0.diff(z) - tc2.diff(x)
        out2 = tc1.diff(x) - tc0.diff(y)

        # set equations
        self.equations = {}
        grad = ux.diff(x) + uy.diff(y) + uz.diff(z)
        self.equations["LameEqx"] = (
            ux.diff(t, 2) - vp**2 * grad.diff(x) + vs**2 * out0
        )
        self.equations["LameEqy"] = (
            uy.diff(t, 2) - vp**2 * grad.diff(y) + vs**2 * out1
        )
        self.equations["LameEqz"] = (
            uz.diff(t, 2) - vp**2 * grad.diff(z) + vs**2 * out2
        )

