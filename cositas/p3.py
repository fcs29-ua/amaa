# Simple script to generate some static pics for GitHub page etc.

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.special import gamma
from scipy.special import zeta


def phase_portrait(fun, delta=0.01, box=(-1, 1, -1, 1)):

    box = interpret_box(box)
    x = np.arange(box[0], box[1], delta)
    y = np.arange(box[2], box[3], delta)

    # Generate plot domain
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # Evaluate function
    fZ = fun(Z)

    # Pass over for display, and return AxesImage
    return gen_portrait(Z, fZ, box)

########### NON API #####################################################

def interpret_box(box):

    if (isinstance(box, float) or isinstance(box, complex) or
        isinstance(box, int)):
        # Turn scalar into 1-tuple
        box = box,

    if len(box) == 4:
        # Two intervals specified, nothing to do
        outbox = box
    elif len(box) == 3:
        # We have a midpoint and radii for real and imag part
        (mid, r_re, r_im) = box

        # Allow real input, interpret with mid.z == 0
        mid = complex(mid)
        outbox = (
            mid.real - r_re, mid.real + r_re,
            mid.imag - r_im, mid.imag + r_im
            )
    elif len(box) == 2 and type(box[1]) == complex:
        # We have lower-left and upper-right point in the complex plae

        (ll, ur) = box
        ll = complex(ll)

        outbox = (ll.real, ur.real, ll.imag, ur.imag)
    elif len(box) == 2:
        (mid, r) = box
        mid = complex(mid)
        outbox = (mid.real - r, mid.real + r, mid.imag - r, mid.imag + r)
    elif len(box) == 1:
        mid = complex(box[0])
        outbox = (mid.real - 1, mid.real + 1, mid.imag - 1, mid.imag + 1)
    else: assert(False)

    return tuple([float(v) for v in outbox])


def gen_portrait(Z, fZ, box, modulus=False):
    """Show phase portrait represented by given rectangular array of
    complex numbers."""

    # Split f(Z) apart. Phase is oriented such that red >0 and blue <0
    phase = np.angle(-fZ)
    modulus = np.abs(fZ)

    # Get base hsv colormap
    cmap = cm.hsv
    norm = colors.Normalize()
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    imvals = mapper.to_rgba(phase)

    # añadimos un colorbar
    fig, ax = plt.subplots()
    axCbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    axCbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    axCbar.set_ticklabels(['$0$', '$\\frac{\pi}{2}$', '$\pi$', '$\\frac{3\pi}{2}$', '$2\pi$'])

    return plt.imshow(imvals, origin='lower', extent=tuple(box))


def example_polynomial():
    """Standard phase portait of a degree-two polynomial"""

    def polyfun(z):
        return z*z - z + 1

    p = phase_portrait(polyfun)
    return p

def plot_my_stuff(f, resol, boxsize, nombre):
    
    plt.figure()
    plt.colorbar(phase_portrait(*args), location = 'right', orientation = 'vertical')
    plt.title(nombre)
    plt.xlabel()
    plt.ylabel()
    plt.show()

if __name__ == '__main__':
    def f(z):
        return z

    def poly1(z):
        return np.sin(z, dtype=complex)

    def poly2(z):
        return ((z**2-1)*(z-2-1j)**2)/(z**2+2+2j)

    def racional(z):
        return 1/z

    def potfun(z):
        serie = 0
        for i in range (0, 10):
            serie += z**i
        return serie

    def trigonom(z):
        return 0.5*(np.exp(z) + np.exp(-1*z))

    f_gamma = gamma
    f_zeta = zeta


    x0, xf, y0, yf = -10, 10, -10, 10
    caja = [x0, xf, y0, yf]

    ratio = 0.01

    args = (poly2, ratio, caja)

    phase_portrait(*args)
    plt.show()

    print('Finished')
