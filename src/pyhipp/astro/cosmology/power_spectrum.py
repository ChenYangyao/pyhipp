from __future__ import annotations
import typing
from typing import Self
from numba.experimental import jitclass
import numba
import numpy as np
from .flat_lambda_cdm_nr import _FlatLambdaCDMNR
from scipy import integrate
from pyhipp.numerical import interpolate


@numba.njit
def sqr(x):
    return x*x


@numba.njit
def cube(x):
    return x*x*x


@numba.njit
def pow4(x):
    return x*x*x*x


@numba.njit
def pow(x, y):
    return x**y


class _TransferFunction:
    def transfer_function(self, kk: float) -> float:
        '''
        @kk [h/Mpc]: Wavenumber at which to calculate transfer function.
        
        The returned transfer function may not be normalized.
        '''
        raise NotImplementedError("Do not use this class directly.")


@jitclass
class _TransferFunctionEH99:
    '''    
    Fitting Formulae for CDM + Baryon + Massive Neutrino (MDM) cosmologies.
    CDM Variants accurate for Omega_b / Omega_m<0.3, Omega_nu/Omega_m<0.3.
    Includes degenerate mass neutrinos, cosmological constant, and spatial 
    curvature.
    
    Authors: Daniel J. Eisenstein & Wayne Hu, Institute for Advanced Study.
    ApJ, 1999, 511, 5.
    http://background.uchicago.edu/~whu/transfer/transferpage.html
    (the power.c file)
    
    Must call set_cosmology() before computing the transfer function.
    '''

    # -- cosmology --

    alpha_gamma: numba.float64      # sqrt(alpha_nu)
    alpha_nu: numba.float64         # The small-scale suppression
    beta_c: numba.float64           # The correction to the log in the small-scale
    num_degen_hdm: numba.float64    # Number of degenerate massive neutrino species
    f_baryon: numba.float64         # Baryon fraction
    f_bnu: numba.float64            # Baryon + Massive Neutrino fraction
    f_cb: numba.float64             # Baryon + CDM fraction
    f_cdm: numba.float64            # CDM fraction
    f_hdm: numba.float64            # Massive Neutrino fraction
    growth_k0: numba.float64        # D_1(z) -- the growth function as k->0
    # D_1(z)/D_1(0) -- the growth relative to z=0
    growth_to_z0: numba.float64
    hhubble: numba.float64          # Need to pass Hubble constant to TFmdm_onek_hmpc()
    k_equality: numba.float64           # The comoving wave number of the horizon at equality
    obhh: numba.float64                 # Omega_baryon * hubble^2
    omega_curv: numba.float64           # = 1 - omega_matter - omega_lambda
    omega_lambda_z: numba.float64       # Omega_lambda at the given redshift
    omega_matter_z: numba.float64       # Omega_matter at the given redshift
    omhh: numba.float64                 # Omega_matter * hubble^2
    onhh: numba.float64                 # Omega_hdm * hubble^2
    p_c: numba.float64                  # The correction to the exponent before drag epoch
    p_cb: numba.float64                 # The correction to the exponent after drag epoch
    sound_horizon_fit: numba.float64    # The sound horizon at the drag epoch
    theta_cmb: numba.float64            # The temperature of the CMB, in units of 2.7 K
    y_drag: numba.float64               # Ratio of z_equality to z_drag
    z_drag: numba.float64               # Redshift of the drag epoch
    z_equality: numba.float64           # Redshift of matter-radiation equality

    # variant
    with_neutrino: numba.bool_

    # -- working variables --
    gamma_eff: numba.float64        # Effective \Gamma
    growth_cb: numba.float64        # Growth factor for CDM+Baryon perturbations
    # Growth factor for CDM+Baryon+Neutrino pert.
    growth_cbnu: numba.float64
    max_fs_correction: numba.float64    # Correction near maximal free streaming
    qq: numba.float64                   # Wavenumber rescaled by \Gamma
    qq_eff: numba.float64               # Wavenumber rescaled by effective Gamma

    qq_nu: numba.float64            # Wavenumber compared to maximal free streaming
    tf_master: numba.float64        # Master TF
    tf_sup: numba.float64           # Suppressed TF
    y_freestream: numba.float64     # The epoch of free-streaming for a given scale

    def __init__(self, with_neutrino=False) -> None:
        self.set_variant(with_neutrino)

    def set_cosmology(self,
                      omega_matter: float,
                      omega_baryon: float,
                      omega_hdm: float,
                      degen_hdm: int,
                      omega_lambda: float,
                      hubble: float,
                      redshift: float = 0.0):
        r'''
        (Re)set the cosmological parameters.

        @omega_matter: Density of CDM, baryons, and massive neutrinos,
            in units of the critical density.
        @omega_baryon (>=0): Density of baryons, in units of critical.
        @omega_hdm (>=0):Density of massive neutrinos, in units of critical
        @degen_hdm: Number of degenerate massive neutrino species
        @omega_lambda: Cosmological constant
        @hubble (>0, <=2): Hubble constant, in units of 100 km/s/Mpc
        @redshift (>-1, <=99): The redshift at which to evaluate.
        '''
        omega_matter = np.float64(omega_matter)
        omega_baryon = np.float64(omega_baryon)
        omega_hdm = np.float64(omega_hdm)
        degen_hdm = np.int32(degen_hdm)
        omega_lambda = np.float64(omega_lambda)
        hubble = np.float64(hubble)
        redshift = np.float64(redshift)

        theta_cmb = 2.728/2.7                        # Assuming T_cmb = 2.728 K

        # look for strange input
        if omega_baryon < 0.0:
            raise ValueError("Negative omega_baryon set to trace amount.")
        if omega_hdm < 0.0:
            raise ValueError("Negative omega_hdm set to trace amount.")
        if hubble <= 0.0:
            raise ValueError("Negative Hubble constant illegal.")
        if hubble > 2.0:
            raise ValueError(
                "Hubble constant should be in units of 100 km/s/Mpc.")
        if redshift <= -1.0:
            raise ValueError("Redshift < -1 is illegal.")
        if redshift > 99.0:
            raise ValueError("Large redshift entered.  TF may be inaccurate.")

        if degen_hdm < 1:
            degen_hdm = 1
        num_degen_hdm = np.float64(degen_hdm)

        if omega_baryon <= 0:
            omega_baryon = 1.0e-5
        if omega_hdm <= 0:
            omega_hdm = 1.0e-5

        # compute variables
        hubble_sqr = sqr(hubble)
        omega_curv = 1.0-omega_matter-omega_lambda
        omhh = omega_matter*hubble_sqr
        obhh = omega_baryon*hubble_sqr
        onhh = omega_hdm*hubble_sqr
        f_baryon = omega_baryon/omega_matter
        f_hdm = omega_hdm/omega_matter
        f_cdm = 1.0-f_baryon-f_hdm
        f_cb = f_cdm+f_baryon
        f_bnu = f_baryon+f_hdm

        theta_cmb_sqr = sqr(theta_cmb)
        z_equality = 25000.0*omhh/sqr(theta_cmb_sqr)
        k_equality = 0.0746*omhh/theta_cmb_sqr

        z_drag_b1 = 0.313*pow(omhh, -0.419)*(1+0.607*pow(omhh, 0.674))
        z_drag_b2 = 0.238*pow(omhh, 0.223)
        z_drag = 1291*pow(omhh, 0.251)/(1.0+0.659*pow(omhh, 0.828))\
            * (1.0+z_drag_b1*pow(obhh, z_drag_b2))
        y_drag = z_equality/(1.0+z_drag)

        sound_horizon_fit = \
            44.5*np.log(9.83/omhh) / np.sqrt(1.0+10.0*pow(obhh, 0.75))

        p_c = 0.25*(5.0-np.sqrt(1+24.0*f_cdm))
        p_cb = 0.25*(5.0-np.sqrt(1+24.0*f_cb))

        omega_denom = omega_lambda+sqr(1.0+redshift)*(
            omega_curv+omega_matter*(1.0+redshift)
        )
        omega_lambda_z = omega_lambda/omega_denom
        omega_matter_z = \
            omega_matter * sqr(1.0+redshift)*(1.0+redshift)/omega_denom
        growth_k0 = z_equality/(1.0+redshift)*2.5*omega_matter_z / (
            pow(omega_matter_z, 4.0/7.0)-omega_lambda_z +
            (1.0+omega_matter_z/2.0)*(1.0+omega_lambda_z/70.0)
        )
        growth_to_z0 = z_equality*2.5*omega_matter/(
            pow(omega_matter, 4.0/7.0)
            - omega_lambda + (1.0+omega_matter/2.0)*(1.0+omega_lambda/70.0)
        )
        growth_to_z0 = growth_k0/growth_to_z0

        alpha_nu = f_cdm/f_cb*(5.0-2.*(p_c+p_cb))/(5.-4.*p_cb) *  \
            pow(1+y_drag, p_cb-p_c) *  \
            (1+f_bnu*(-0.553+0.126*f_bnu*f_bnu)) /  \
            (
                1-0.193*np.sqrt(f_hdm*num_degen_hdm)
                + 0.169*f_hdm*pow(num_degen_hdm, 0.2)
        )*(1+(p_c-p_cb)/2*(1+1/(3.-4.*p_c)/(7.-4.*p_cb))/(1+y_drag))
        alpha_gamma = np.sqrt(alpha_nu)
        beta_c = 1/(1-0.949*f_bnu)

        hhubble = hubble

        # set to the instance
        self.alpha_gamma = alpha_gamma
        self.alpha_nu = alpha_nu
        self.beta_c = beta_c
        self.num_degen_hdm = num_degen_hdm
        self.f_baryon = f_baryon
        self.f_bnu = f_bnu
        self.f_cb = f_cb
        self.f_cdm = f_cdm
        self.f_hdm = f_hdm
        self.growth_k0 = growth_k0
        self.growth_to_z0 = growth_to_z0
        self.hhubble = hhubble
        self.k_equality = k_equality
        self.obhh = obhh
        self.omega_curv = omega_curv
        self.omega_lambda_z = omega_lambda_z
        self.omega_matter_z = omega_matter_z
        self.omhh = omhh
        self.onhh = onhh
        self.p_c = p_c
        self.p_cb = p_cb
        self.sound_horizon_fit = sound_horizon_fit
        self.theta_cmb = theta_cmb
        self.y_drag = y_drag
        self.z_drag = z_drag
        self.z_equality = z_equality

    def set_variant(self, with_neutrino=False):
        self.with_neutrino = with_neutrino

    def _transfer_function(self, kk: float) -> tuple[float, float]:
        '''
        @kk [h/Mpc]: the wavenumber. Return the transfer function for the
        cosmology that has been set.
        
        Returns:
        - growth_cb: the transfer function for density-weighted
            CDM + Baryon perturbations. 
        - growth_cbnu: the transfer function for density-weighted
            CDM + Baryon + Massive Neutrino perturbations.
        '''
        kk = np.float64(kk)
        kk = kk * self.hhubble

        qq = kk/self.omhh*sqr(self.theta_cmb)

        y_freestream = 17.2*self.f_hdm*(1+0.488*pow(self.f_hdm, -7.0/6.0)) *  \
            sqr(self.num_degen_hdm*qq/self.f_hdm)
        temp1 = pow(self.growth_k0, 1.0-self.p_cb)
        temp2 = pow(self.growth_k0/(1+y_freestream), 0.7)
        growth_cb = pow(1.0+temp2, self.p_cb/0.7)*temp1
        growth_cbnu = pow(pow(self.f_cb, 0.7/self.p_cb)+temp2,
                          self.p_cb/0.7)*temp1

        gamma_eff = self.omhh*(
            self.alpha_gamma+(1-self.alpha_gamma) /
            (1+sqr(sqr(kk*self.sound_horizon_fit*0.43)))
        )
        qq_eff = qq*self.omhh/gamma_eff

        tf_sup_L = np.log(2.71828+1.84*self.beta_c*self.alpha_gamma*qq_eff)
        tf_sup_C = 14.4+325/(1+60.5*pow(qq_eff, 1.11))
        tf_sup = tf_sup_L/(tf_sup_L+tf_sup_C*sqr(qq_eff))

        qq_nu = 3.92*qq*np.sqrt(self.num_degen_hdm/self.f_hdm)
        max_fs_correction = 1+1.2*pow(self.f_hdm, 0.64) *  \
            pow(self.num_degen_hdm, 0.3+0.6*self.f_hdm) /  \
            (pow(qq_nu, -1.6)+pow(qq_nu, 0.8))
        tf_master = tf_sup*max_fs_correction

        tf_cb = tf_master*growth_cb/self.growth_k0
        tf_cbnu = tf_master*growth_cbnu/self.growth_k0

        return tf_cb, tf_cbnu

    def transfer_function(self, kk: float) -> float:
        cb, cbnu = self._transfer_function(kk)
        out = cbnu if self.with_neutrino else cb
        return out


@jitclass
class _TransferFunctionEH98:
    '''
    Fitting Formulae for high baryon models (use if interested in baryon 
    oscillations significant for Omega_b/Omega_m > Omega_m h^2 + 0.2).
    
    Authors: Daniel J. Eisenstein & Wayne Hu, Institute for Advanced Study.
    ApJ, 1998, 496, 605.
    http://background.uchicago.edu/~whu/transfer/transferpage.html
    (the tf_fit.c file)
    
    Must call set_cosmology() before computing the transfer function.
    '''

    hubble: numba.float64               # Hubble constant, in units of 100 km/s/Mpc
    omhh: numba.float64                 # Omega_matter*h^2
    obhh: numba.float64                 # Omega_baryon*h^2
    theta_cmb: numba.float64            # Tcmb in units of 2.7 K
    z_equality: numba.float64           # Redshift of matter-radiation equality, really 1+z
    k_equality: numba.float64           # Scale of equality, in Mpc^-1
    z_drag: numba.float64               # Redshift of drag epoch
    R_drag: numba.float64               # Photon-baryon ratio at drag epoch
    R_equality: numba.float64           # Photon-baryon ratio at equality epoch
    sound_horizon: numba.float64        # Sound horizon at drag epoch, in Mpc
    k_silk: numba.float64               # Silk damping scale, in Mpc^-1
    alpha_c: numba.float64              # CDM suppression
    beta_c: numba.float64               # CDM log shift
    alpha_b: numba.float64              # Baryon suppression
    beta_b: numba.float64               # Baryon envelope shift
    beta_node: numba.float64            # Sound horizon shift
    k_peak: numba.float64               # Fit to wavenumber of first peak, in Mpc^-1
    sound_horizon_fit: numba.float64    # Fit to sound horizon, in Mpc
    alpha_gamma: numba.float64          # Gamma suppression in approximate TF

    def __init__(self) -> None:
        pass

    def set_cosmology(self,
                      omega0: float,
                      f_baryon: float,
                      hubble: float,
                      Tcmb: float = 2.728):
        '''
        @omega0: the matter density (baryons+CDM) in units of critical .
        @f_baryon: the ratio of baryon density to matter density.
        @hubble: the Hubble constant, in units of 100 km/s/Mpc.
        @Tcmb: the CMB temperature in Kelvin. Default to the COBE value 2.728 .
        '''
        # look for strange input
        omega0 = np.float64(omega0)
        f_baryon = np.float64(f_baryon)
        hubble = np.float64(hubble)
        Tcmb = np.float64(Tcmb)

        if omega0 <= 0.0:
            raise ValueError("Negative omega0 illegal.")
        if f_baryon <= 0.0:
            raise ValueError("Negative f_baryon illegal.")
        if hubble <= 0.0:
            raise ValueError("Negative Hubble constant illegal.")
        if Tcmb <= 0:
            raise ValueError("CMB temperature must be positive.")

        # compute variables
        omega0hh = omega0 * sqr(hubble)

        omhh = omega0hh
        obhh = omhh*f_baryon
        theta_cmb = Tcmb/2.7

        z_equality = 2.50e4*omhh/pow4(theta_cmb)
        k_equality = 0.0746*omhh/sqr(theta_cmb)

        z_drag_b1 = 0.313*pow(omhh, -0.419)*(1+0.607*pow(omhh, 0.674))
        z_drag_b2 = 0.238*pow(omhh, 0.223)
        z_drag = 1291*pow(omhh, 0.251)/(1+0.659*pow(omhh, 0.828)) * \
            (1+z_drag_b1*pow(obhh, z_drag_b2))

        R_drag = 31.5*obhh/pow4(theta_cmb)*(1000/(1+z_drag))
        R_equality = 31.5*obhh/pow4(theta_cmb)*(1000/z_equality)

        sound_horizon = 2./3./k_equality*np.sqrt(6./R_equality) * \
            np.log((np.sqrt(1+R_drag)+np.sqrt(R_drag+R_equality))
                   / (1+np.sqrt(R_equality)))

        k_silk = 1.6*pow(obhh, 0.52)*pow(omhh, 0.73)*(1+pow(10.4*omhh, -0.95))

        alpha_c_a1 = pow(46.9*omhh, 0.670)*(1+pow(32.1*omhh, -0.532))
        alpha_c_a2 = pow(12.0*omhh, 0.424)*(1+pow(45.0*omhh, -0.582))
        alpha_c = pow(alpha_c_a1, -f_baryon) * pow(alpha_c_a2, -cube(f_baryon))

        beta_c_b1 = 0.944/(1+pow(458*omhh, -0.708))
        beta_c_b2 = pow(0.395*omhh, -0.0266)
        beta_c = 1.0/(1+beta_c_b1*(pow(1-f_baryon, beta_c_b2)-1))

        y = z_equality/(1+z_drag)
        alpha_b_G = y*(-6.*np.sqrt(1+y)+(2.+3.*y) *
                       np.log((np.sqrt(1+y)+1)/(np.sqrt(1+y)-1)))
        alpha_b = 2.07*k_equality*sound_horizon*pow(1+R_drag, -0.75)*alpha_b_G

        beta_node = 8.41*pow(omhh, 0.435)
        beta_b = 0.5+f_baryon+(3.-2.*f_baryon)*np.sqrt(pow(17.2*omhh, 2.0)+1)

        k_peak = 2.5*3.14159*(1+0.217*omhh)/sound_horizon
        sound_horizon_fit = \
            44.5*np.log(9.83/omhh)/np.sqrt(1+10.0*pow(obhh, 0.75))

        alpha_gamma = 1-0.328*np.log(431.0*omhh)*f_baryon \
            + 0.38*np.log(22.3*omhh)*sqr(f_baryon)

        # set to the instance
        self.hubble = hubble
        self.omhh = omhh
        self.obhh = obhh
        self.theta_cmb = theta_cmb
        self.z_equality = z_equality
        self.k_equality = k_equality
        self.z_drag = z_drag
        self.R_drag = R_drag
        self.R_equality = R_equality
        self.sound_horizon = sound_horizon
        self.k_silk = k_silk
        self.alpha_c = alpha_c
        self.beta_c = beta_c
        self.alpha_b = alpha_b
        self.beta_b = beta_b
        self.beta_node = beta_node
        self.k_peak = k_peak
        self.sound_horizon_fit = sound_horizon_fit
        self.alpha_gamma = alpha_gamma

    def _transfer_function(self, kk: float) -> tuple[float, float, float]:
        '''
        @kk [h/Mpc]: Wavenumber at which to calculate transfer function.
        
        Returns:
        - tf_full: the full transfer function.
        - tf_baryon: the baryonic contribution to the full fit.
        - tf_cdm: the CDM contribution to the full fit.
                the input was not NULL.
        '''

        kk = np.float64(kk)
        k = kk * self.hubble

        k = np.abs(k)
        if k == 0.0:
            return 1.0, 1.0, 1.0

        q = k/13.41/self.k_equality
        xx = k*self.sound_horizon

        T_c_ln_beta = np.log(2.718282+1.8*self.beta_c*q)
        T_c_ln_nobeta = np.log(2.718282+1.8*q)
        T_c_C_alpha = 14.2/self.alpha_c + 386.0/(1+69.9*pow(q, 1.08))
        T_c_C_noalpha = 14.2 + 386.0/(1+69.9*pow(q, 1.08))

        T_c_f = 1.0/(1.0+pow4(xx/5.4))
        T_c = T_c_f*T_c_ln_beta/(T_c_ln_beta+T_c_C_noalpha*sqr(q)) +  \
            (1-T_c_f)*T_c_ln_beta/(T_c_ln_beta+T_c_C_alpha*sqr(q))

        s_tilde = self.sound_horizon*pow(1+cube(self.beta_node/xx), -1./3.)
        xx_tilde = k*s_tilde

        T_b_T0 = T_c_ln_nobeta/(T_c_ln_nobeta+T_c_C_noalpha*sqr(q))
        T_b = np.sin(xx_tilde)/(xx_tilde)*(
            T_b_T0/(1+sqr(xx/5.2)) +
            self.alpha_b/(1+cube(self.beta_b/xx)) *
            np.exp(-pow(k/self.k_silk, 1.4))
        )

        f_baryon = self.obhh/self.omhh
        T_full = f_baryon*T_b + (1-f_baryon)*T_c

        return T_full, T_b, T_c

    def transfer_function(self, kk: float) -> float:
        return self._transfer_function(kk)[0]


@jitclass
class _TransferFunctionFlatLambdaCDMNR:
    '''
    Transfer function for a flat lambda-CDM cosmology, 
    without massive neutrino.
    '''

    _eh98: _TransferFunctionEH98
    _eh99: _TransferFunctionEH99
    _with_baryon_effect: numba.bool_

    def __init__(self, cosm: _FlatLambdaCDMNR,
                 with_baryon_effect: bool = False) -> None:

        self.set_variant(cosm, with_baryon_effect)

    def set_variant(self, cosm: _FlatLambdaCDMNR,
                    with_baryon_effect: bool = False):

        if with_baryon_effect:
            eh98 = _TransferFunctionEH98()
            eh98.set_cosmology(cosm.omega_m0,
                               cosm.baryon_fraction(),
                               cosm.hubble)
            self._eh98 = eh98
        else:
            eh99 = _TransferFunctionEH99(False)
            eh99.set_cosmology(cosm.omega_m0, cosm.omega_b0, 0.0,
                               0, cosm.omega_l0, cosm.hubble)
            self._eh99 = eh99
        self._with_baryon_effect = with_baryon_effect

    def transfer_function(self, kk: float) -> float:
        if self._with_baryon_effect:
            tf = self._eh98.transfer_function(kk)
        else:
            tf = self._eh99.transfer_function(kk)
        return tf


class _GrowthFactor:
    def growth_factor(self, z: float) -> float:
        '''
        @z: Redshift.
        
        The returned growth factor is normalized as D_1(z=0) = 1.
        '''
        raise NotImplementedError("Do not use this class directly.")


@jitclass
class _GrowthFactorCarroll92:
    '''
    D_1(z), the linear growth function (Peebles 1980).
    
    See: Lahav et al. 1991, Carroll et al. 1992.
    '''

    cosm: _FlatLambdaCDMNR
    norm: numba.float64

    def __init__(self, cosm: _FlatLambdaCDMNR) -> None:

        self.cosm = cosm
        self.norm = 1.0

        self.__normalize()

    def growth_factor(self, z: float):
        cosm, norm = self.cosm, self.norm
        omega_m = cosm.omega_m(z)
        omega_l = cosm.omega_l(z)

        f1 = norm * 2.5 * omega_m / (1. + z)
        f2 = omega_m**(4./7.) - omega_l + \
            (1. + omega_m / 2.)*(1. + omega_l / 70.)
        return f1 / f2

    def __normalize(self):
        D_1_z0 = self.growth_factor(0.0)
        norm = 1.0 / D_1_z0
        self.norm = norm

    def delta_crit(self, z):
        return 1.686 / self.growth_factor(z)


@jitclass
class _PowerSpectrumFlatLambdaCDMNR:
    '''
    The power spectrum P(k, z) or Delta^2(k, z), and the integrated variance 
    of density field, sigma^2(r, z). Other convenience functions are also
    provided.
    
    NR: non-radiative. Massive neutrino is not included in computing T(k).
    The cosmology, `cosm`, does not include any radiative contribution in 
    the late-time growth.
    
    @cosm: the cosmology.
    @with_baryon_effect: whether to include the baryon effect in the transfer
    function. 
    
    The approximation with and without baryon effect are given by 
    Eisenstein & Hu 1998 and 1999, respectively.
    For lg(sigma(M)), the effect is <= 0.01 dex for lg(M) in [-10, 10] for 
    the current cosmology.
    
    The growth factor, D1(z), is given by Carroll et al. 1992.
    
    Normalization: we use a slightly different normalization for the sake 
    of numerical stability:
        _sigma_sqr_norm := (delta_H/10^3)^2 * ((c/H_0)/(Mpc/h))^(3+n),
    where delta_H is the normalization factor commonly used for Delta^2(k).
    
    All input/output quantities are in the cosmological unit system defined 
    by cosm.us,
    e.g.
    - k: [h/Mpc]
    - r: [Mpc/h]
    - mass: [10^10 Msun/h]
    '''
    _cosm: _FlatLambdaCDMNR
    _tf: _TransferFunctionFlatLambdaCDMNR
    _gf: _GrowthFactorCarroll92

    _sigma_sqr_norm: numba.float64

    def __init__(self, cosm: _FlatLambdaCDMNR,
                 with_baryon_effect=False) -> None:

        tf = _TransferFunctionFlatLambdaCDMNR(cosm, with_baryon_effect)
        gf = _GrowthFactorCarroll92(cosm)

        self._cosm = cosm
        self._tf = tf
        self._gf = gf

        self._sigma_sqr_norm = 1.0
        self._normalize_tf()

    def _j1(self, x):
        if x < 1.0e-5:
            return (1./3.) * x
        return (np.sin(x) - x * np.cos(x)) / (x*x)

    def window_func(self, k: float, r: float) -> float:
        x = k*r
        if x < 1.0e-5:
            return 1.0
        return 3.0 * self._j1(x) / x

    def transfer_func(self, k: float):
        '''
        Transfer function T(k), unnormalized.
        '''
        return self._tf.transfer_function(k)

    def growth_fac(self, z: float):
        '''
        D1(z), normalized as D1(z=0) = 1.
        '''
        return self._gf.growth_factor(z)
    
    def delta_crit(self, z: float):
        '''
        Critical overdensity for spherical collapse.
        '''
        return self._gf.delta_crit(z)

    def _sigma_sqr_integrand(self, ln_k, r, z):
        n_spec = self._cosm.n_spec

        k = np.exp(ln_k)
        T = self.transfer_func(k)
        D1 = self.growth_fac(z)
        W = self.window_func(k, r)

        f1 = k**(3.0 + n_spec)
        f2 = 1.0e3 * T * D1 * W         # multiplies 10^3 for stability
        return f1 * f2**2

    def sigma_sqr(self, r: float, z: float):
        '''
        sigma^2(r, z).
        '''
        with numba.objmode(res='float64'):
            # box/unbox is likely a bottleneck. we may change this in the future
            ln_k_min, ln_k_max = -23.0, 23.0
            res = integrate.quad(
                self._sigma_sqr_integrand,
                ln_k_min, ln_k_max, args=(r, z),
                limit=128)[0]
        sigma_sqr = res * self._sigma_sqr_norm
        return sigma_sqr

    def sigma_sqr_at_mass(self, mass: float, z: float):
       '''
       sigma^2(M,z).
       '''
       r = self._cosm.densities().m_to_r(mass)
       return self.sigma_sqr(r, z)

    def interp_lg_sigma_at_lg_mass(
            self,
            lg_mass_range: tuple[float, float] = (-10.0, 10.0),
            n_nodes: int = 256):
        '''
        Return an interpolator for lg(sigma) as a function of lg(mass).
        Usually used for speedup in, e.g. mass function calculation.
        '''
        lm1, lm2 = lg_mass_range
        lm = np.linspace(lm1, lm2, n_nodes)
        m = 10.0**lm
        lsigma = np.empty_like(m)
        for i, _m in enumerate(m):
            sigma_sqr = self.sigma_sqr_at_mass(_m, 0.0)
            lsigma[i] = np.log10(np.sqrt(sigma_sqr))
        return interpolate.Linear(lm, lsigma, True)

    def _normalize_tf(self):
        sigma_8_sqr = self._cosm.sigma_8 ** 2
        sigma_sqr = self.sigma_sqr(8.0, 0.0)
        self._sigma_sqr_norm = sigma_8_sqr / sigma_sqr

    def big_delta_sqr(self, k: float, z: float):
        '''
        Delta^2(k, z),
        defined in EH 98, eq. A1.
        '''
        norm, n_spec = self._sigma_sqr_norm, self._cosm.n_spec

        f1 = k**(3.0 + n_spec)
        T = self.transfer_func(k)
        D1 = self.growth_fac(z)
        f2 = 1.0e3 * T * D1

        return norm * f1 * f2**2

    def power(self, k: float, z: float):
        '''
        P(k, z), in the unit of [Mpc/h]^3,
        defined in EH 98, eq. A1.
        '''
        norm, n_spec = self._sigma_sqr_norm, self._cosm.n_spec
        f1 = 2.0 * np.pi**2 * k**n_spec
        T = self.transfer_func(k)
        D1 = self.growth_fac(z)
        f2 = 1.0e3 * T * D1

        return norm * f1 * f2**2
