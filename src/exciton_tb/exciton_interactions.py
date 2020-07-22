import numpy as np
import scipy.special as spec

trunc_thresh = 1e-5
coulombic_prefactor = 14.38
keldysh_prefactor = 22.59


def interaction_potential(radius, potential_name, args, trunc_val):
    """
    Obtain interaction potential, treated at the centre value
    :param radius: Distance from the centre
    :param potential_name: Name of potential:
                           'Keldysh', 'keldysh', 'keld', 'Keld': Keldysh int.
                           'Coulomb', 'coulomb', 'coul', 'Coul': Coulomb int
                           'Yukawa', 'yukawa', 'yuk', 'Yuk': Yukawa type int.
    :param args:
    :param trunc_val:
    :return:
    """

    radius_truncated = trunc_val if radius < trunc_thresh else radius
    lower_potential_str = str(potential_name).lower()
    if lower_potential_str in ['keldysh', 'keld']:
        ep12 = 0.5*(1 + args['substrate_dielectric'])
        radius_0 = args['radius_0']
        potential = keldysh_interaction(radius_truncated, radius_0, ep12)
    elif lower_potential_str in ['coulomb', 'coul']:
        ep12 = 0.5*(1 + args['substrate_dielectric'])
        potential = coulomb_interaction(radius_truncated, ep12)
    elif lower_potential_str in ['yukawa', 'yuk']:
        ep12 = 0.5*(1 + args['substrate_dielectric'])
        gamma = args['gamma']
        potential = yukawa_interaction(radius_truncated, gamma, ep12)
    else:
        raise Exception("potential_name unavailable. See help().")
    return potential

def keldysh_interaction(radius, radius_0, ep12):
    """
    Keldysh Interaction.
    :param radius: Distance from centre
    :param radius_0: Scale distance corresponding to inherent screening.
    :param ep12: average of top and bottome dielectric constants.
    :return: interaction
    """
    term_arg = (radius / radius_0)*ep12
    struve_term = spec.struve(0, term_arg)
    bessel_term = spec.y0(term_arg)
    interaction = keldysh_prefactor/radius_0*(struve_term - bessel_term)
    return interaction

def coulomb_interaction(radius, ep12):
    """
    Coulomb Interaction.
    :param radius: Distance from centre
    :param ep12: average of top and bottome dielectric constants.
    :return: interaction
    """
    interaction = coulombic_prefactor/ep12/radius
    return interaction

def yukawa_interaction(radius, gamma, ep12):
    """
    Yukawa type interaction
    :param radius: Distance from centre
    :param gamma: potential decay exponent
    :param ep12: average of top and bottome dielectric constants.
    :return: interaction
    """
    prefactor = coulombic_prefactor/ep12
    interaction = prefactor*np.exp(-gamma*radius)/radius
    return interaction