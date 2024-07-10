import numpy as np;
from scipy.io import loadmat;
from astropy.io import ascii;
import astropy.io.fits as fits;
from astropy import units;
from astropy.time import Time;
from scipy.interpolate import interp1d,interp2d,RectBivariateSpline;
from scipy.ndimage import uniform_filter, gaussian_filter;
import scipy.ndimage as ndi
import os;
import warnings;
import pickle;
from math import factorial as fac
import matplotlib.pyplot as plt;

# Define the path to locate data files
data_path = os.path.dirname (__file__)+'/data/';
tmp_path  = os.path.expanduser('~')+'/Datas/GPAO/';

def load_DM1():
    '''
    Load the dynamic data from BAX301 from ALPAO
    '''
    data = ascii.read (data_path+'/DM1_tf.dat');
    freq = data['freq [Hz]'];
    amp  = 10.**(data['amp [db]']/20);
    phi  = data['phi [deg]'] / 180.* np.pi;

    return freq, amp*np.exp(1.j*phi);

def load_BAX301():
    '''
    Load the dynamic data from BAX301 from ALPAO
    '''
    data = ascii.read (data_path+'/BAX301_tf.dat');
    freq = data['freq [Hz]'];
    amp  = 10.**(data['amp [db]']/20);
    phi  = data['phi [deg]'] / 180.* np.pi;

    data = ascii.read(data_path+'/BAX301_step.dat');
    time = data['time [ms]'] * 1e-3;
    step = data['amp [a.u]'];

    return freq, amp*np.exp(1.j*phi), time, step;

def load_BAX153():
    '''
    Freq = np.array([795,930,935,1030,1110,1450]);
    Damp = np.array([0.085,0.04,0.05,0.045,0.035,0.2]);
    Amp  = np.array([0,0.18,0,0,0.25,0.4]);
    '''
    filename='/Users/lebouquj/MyCore/NAOMI/DM_ALPAO/BAX153/38mm pupil/report_freq/result.mat';
    result = loadmat(filename)['result'];
    freq = result['freq'][0][0][0,:];
    amp  = result['center'][0,0]['amp0cos'][0][0][:,0]; 
    phi  = result['center'][0,0]['phase0cos'][0][0][:,0]; 

    amp /= amp[0];
    phi -= phi[0];
    TF = amp * np.exp(1.j * phi);

    filename='/Users/lebouquj/MyCore/NAOMI/DM_ALPAO/BAX153/38mm pupil/report_step/result.mat';
    result = loadmat(filename)['result'];
    time = result['center'][0][0][0][0][0][0]['time'][0][0,:];
    STEP = result['center'][0][0][0][0][0][0]['signal'][0][:,0]; 
    
    return freq, TF, time, STEP;

def model_DM1 (f):
    '''
    Model for mail of Marie Laslande 2022-04-21
    '''
    Freq = np.array([1265]);
    Damp = np.array([0.15]);
    Amp  = np.array([1.0]);
    return model_ALPAO(f,Freq,Amp,Damp);

def model_BAX187 (f):
    '''
    Model for CHARA DM (crudely matched to ALPAO data)
    '''
    Freq = np.array([470,480.,500,525]);
    Damp = np.array([0.07,0.1,0.1,0.1]);
    Amp  = np.array([0.02,0.1,0.1,0.1]);
    return model_ALPAO (f,Freq,Amp,Damp);

def model_DM_GPAO (f,fDM=1000.0):
    '''
    Expected DM for GPAO
    '''
    Freq = np.array([1055,1265.,1600]) / 1055. * fDM;
    Damp = np.array([0.025,0.025,0.1]);
    Amp  = np.array([0.33,0.4,0.27]);
    
    return model_ALPAO (f,Freq,Amp,Damp);
    
def model_ALPAO (f,f0,amp,zeta):
    '''
    Sum of second order 
    '''
    
    # Build full frequency response
    amp = amp / amp.sum();
    No = len (f0);
    FR = 0.j * f;
    for o in range(No):
        s = (0. + 1.j * f) / f0[o];
        FR = FR + amp[o] / (s**2 + 2.*zeta[o]*s + 1);
        
    print ('DM f0 = %.2f Hz'%f0[0]);
    return FR;

def model_M2 (f):
    '''
    Frequency Response of M2, according to page 8 of
    VLT-TRE-ESO-17230-6147
    '''
    s = (0. + 2.j*np.pi * f);
    return 1.7865e11 / ( (s+1247) * (s+300.5) * (s**2 + 470.9*s + 4.953e5));
    
def model_stepping (f,multi,T):
    '''
    Frequency Response of multi-stepping
    '''
    
    # Multistep ALPAO, 8x120us
    if multi == 'ALPAO':
        print ('Stepping ALPAO');
        delta = 120e-6;
        amp = np.array([0.2344,0.2031,0.1719,0.1406,0.1094,0.0781,0.0469,0.0156]);

    elif multi == 'AntiAlias':
        delta = 0.5 * T;
        print ('Stepping Anti-alias with delta = %.3fms'%(delta*1e3));
        amp = np.array([0.6,0.4]);

    elif multi == 'AntiAlias2':
        delta = 0.188 * T;
        print ('Stepping Anti-alias2 with delta = %.3fms'%(delta*1e3));
        amp = np.array([0.6,-0.4,0.47,0.]);

    elif multi == 'AntiAlias3':
        delta = 0.25 * T;
        print ('Stepping Anti-alias3 with delta = %.3fms'%(delta*1e3));
        amp = np.array([1.5*0.5,-0.5*0.5,1.5*0.5,-0.5*0.5]) * np.linspace(1,1,4);        
        
    elif multi == 'NAOMI':
        print ('Stepping NAOMI');
        delta = 200e-6;
        amp = np.array ([0.5,0.75,1.0,0.75,0.5,0.25]);
        
    else:
        print ('Stepping NONE');
        delta = 0.0;
        amp = np.array ([1.0]);

    # Compute Frequency Response as sum of time-shifted dirac
    amp /= amp.sum();
    MS = (np.exp(-2.j*np.pi*(np.arange(len(amp))*delta)[:,None]*f[None,:]) * amp[:,None]).sum(axis=0);
    
    return MS;

def model_delay (f,T):
    '''
    TF of a pure delay
    '''
    s = (0. + 2.j*np.pi * f);
    return np.exp (-T*s);
    
def model_AO (f,Ki,T):
    '''
    TF of a pure delay
    '''
    print ('Tint = %.3f ms '%(T*1e3));
    s = (0. + 2.j*np.pi * f);
    Ts = T*s;
    return Ki/Ts * (1. - np.exp(-Ts))/Ts * np.exp(-Ts);

def model_wfs (f,T,Tread=None):
    '''
    TF of a WF of repetition time T and 
    readout fime Tread. If Tread is None,
    then the readout time is the repetition time
    '''
    if Tread is None: Tread = T;
    print ('Tint = %.3f ms Tread= %.3f ms'%(T*1e3,Tread*1e3));
    s = (0. + 2.j*np.pi * f);
    Ts  = T*s;
    Trs = Tread*s;
    return (1. - np.exp(-Ts))/Ts * np.exp(-Trs);

def model_controller (f,Ki,T):
    '''
    TF of a pure integrator og gain Ki
    and repetation time T
    '''
    s = (0. + 2.j*np.pi * f);
    Ts = T*s;
    return Ki / (1-np.exp(-Ts));

def model_derivator (f,Kd,T):
    '''
    TF of a pure derivatpr of gain Kd
    and repetation time T. To be verified.
    '''
    warnings.warn ('model_derivator has not been verified');
    s = (0. + 2.j*np.pi * f);
    Ts = T*s;
    return Kd * (1-np.exp(-Ts));

def model_hold (f,T):
    '''
    TF of a hold
    '''
    s = (0. + 2.j*np.pi * f);
    Ts = T*s;
    return (1. - np.exp(-Ts))/Ts;

def model_lowpass (f,T):
    '''
    TF of a low pass
    '''
    s = (0. + 2.j*np.pi * f);
    Ts = T*s;
    return 1./(1+Ts);

def aliasing (f,TF,T, zero=True):
    '''
    Alias the Tranfer Function TF
    assuming a repetition time T
    '''
    # Aliasing
    Fep = int(1./T / (f[10]-f[9]) );
    TFA = TF.copy();
    if zero is False: TFA *= 0.0;
    for i in range(1,10):
        TFA += np.conj(np.roll(TF[::-1],Fep*i+1));
        TFA += np.roll(TF,Fep*i-1);
    TFA[0] = TF[0] if zero is True else 0.0;
    return TFA;
    
def todb(TF):
    return 20.*np.log10(np.abs(TF));

def todeg(TF):
    return np.angle(TF,deg=True);

def topower(amp):
    return np.abs(amp)**2;

def total_power(amp):
    return (np.abs(amp)**2).sum();

def get_margins (TF):
    '''
    Return the gain (dB) and phase (deg) margins of the
    Open Loop transfer function.
    '''
    # First frequency with phase 180deg (first flip)
    f0 = np.argmax (np.angle(TF,deg=True)>0); 
    Gm = np.abs(TF)[f0:].max();
    # Last frequency with gain > 1.0
    f0 = len(TF) -1 - np.argmax (np.abs(TF)[::-1] > 1.0);
    Pm = np.abs(np.angle(-TF[:f0],deg=True)).min();
    Pm = np.abs(np.angle(-TF[f0],deg=True));
    return -todb(Gm),Pm;

def optimise_gain (f,TF0,Gmargin=6, Pmargin=45, Overshoot=7, gain=1.0, optimise=True):
    '''
    Return optimal gain of the Open Loop transfer function TF0
    '''
    nfast,nslow = 0,0;

    while True:
        TF = TF0[0:10000] * gain;
        # Get margins and overshoot
        Gm, Pm = get_margins (TF);
        RTF = 1/(1+TF);
        over = todb(RTF).max();
        # Evolve gain
        if optimise is False:
            break;
        if Gm > Gmargin and Pm > Pmargin and over < Overshoot:
            break;
        if Gm < Gmargin-1 and Pm < Pmargin-10 and over > Overshoot+1:
            gain *= 0.95;
            nfast += 1;
        else:
            gain *= 0.99;
            nslow += 1;

    wrn = '(WARNING: not enough slow)' if nslow < 2 else '';
    print ('Optimise in %i fast and %i slow step %s'%(nfast, nslow, wrn));
        
    f0 = f[np.argmax (todb(RTF)>-3)];
    print ('Optimisation:');
    print ('Gain = %.2f '%gain);
    print ('Gain margin  = %.2f db'%Gm);
    print ('Phase margin = %.2f deg'%Pm);
    print ('F_rej = %.2f Hz'%f0);
    print ('Over  = %.2f db'%over);
    
    return gain,Pm,Gm,f0,over;

def alias_and_smooth (f, TF, T):
    # Aliasing
    Fep = int(1./T / (f[10]-f[9]) );
    TFA = TF.copy();
    for i in range(1,10):
        TFA += np.conj(np.roll(TF[::-1],Fep*i));
        TFA += np.roll(TF,Fep*i);
    # Smooth
    return TFA * np.sinc (f*T);

def model_turbulence (f,random_phase=True):
    '''
    Amplitude with PSD -11/3
    '''
    if random_phase:
        phi = 2*np.pi*np.random.random(len(f));
    else:
        phi = 0.0;
    return np.minimum(f**(-5.5/3),1) * 68.0769 * np.exp(1.j * phi);

def analyse_system (fDM,multi,T, delay=0, Tread=None):
    '''
    High level function to analyse the closed-loop performance
    '''
    
    # Frequencies [Hz]
    f = 1.0 * np.arange(100000);
    f[0] = 1e-10;

    # Check sampling
    if (f.max() < 5*fDM) or (f.max() < 5*fDM):
        print ('WARNING: frequency sampling !');
    
    # Model for DM
    # Freq = np.array([795,930,935,1030,1110,1450]) / 930.*fDM;
    # Damp = np.array([0.085,0.04,0.05,0.045,0.035,0.2]);
    # Amp  = np.array([0,0.18,0,0,0.25,0.4]);
    # Freq = np.array([1055,1265.,1600]) / 1055.*fDM;
    # Damp = np.array([0.025,0.025,0.1]);
    # Amp  = np.array([0.33,0.4,0.27]);
    Freq = np.array([1265]) / 1265. * fDM;
    Damp = np.array([0.15]);
    Amp  = np.array([1.0]);
    
    DM = model_ALPAO (f,Freq,Amp,Damp);
    
    # Multistep
    DM = DM * model_stepping (f,multi,T);

    # Model for WFS and controller
    W = model_wfs (f,T,Tread=Tread);
    D = model_delay (f, delay);
    C = model_controller (f,1.0,T);
    H = model_hold (f,T);

    # Optimise gain
    gain,Pm,Gm,f3db,over = optimise_gain (f,DM*H*C*D*W);
    C *= gain;

    # Close loop residuals and command
    OL = DM*H*C*D*W;
    CL = OL / (1+OL);

    # Model for input turbulence
    TURB  = model_turbulence (f);

    # Continuous residuals
    FCMR = (1.-CL) * TURB;
    FCMA = FCMR - DM*H*C*D*aliasing (f, W*FCMR, T, zero=False);

    # Compute residual power
    powerA = total_power (FCMA);
    print ('total power = %.3f'%powerA, flush=True);
    
    return gain,Pm,Gm,f3db,over,powerA;

def save_phase_screens (ns, seeing, pxl=0.0625, L0=100, l0=0.001, name=None):
    '''
    pxl_scale [m]
    seeing [as]
    L0 [m]
    return phase in [um]
    '''
    print ('Compute turbulence');
    
    rad2um = 0.5 / (2*np.pi);
    r0 = 0.98 * 0.5e-6 / (seeing*units.arcsec.to('rad'));

    # Create phase screen
    from aotools.turbulence.phasescreen import ft_phase_screen
    im = ft_phase_screen (r0, ns, pxl, L0, l0)* rad2um;

    hdu = fits.PrimaryHDU (im);
    hdu.header.set ('CDELT1',pxl,'[m] increment x');
    hdu.header.set ('CDELT2',pxl,'[m] increment y');
    hdu.header.set ('BUNUT','um','micrometers');
    hdu.header.set ('KOL_SE',seeing,'[arcsec] @ 500nm');
    hdu.header.set ('KOL_R0',r0,'[m] @ 500nm');
    hdu.header.set ('KOL_L0',L0,'[m] @ 500nm');
    hdu.header.set ('KOL_l0',l0,'[m] @ 500nm');

    if name is not False:
        if name is None: name = 'LARGE_SCRN_%.1fas_new.fits'%seeing;
        if os.path.exists(name): os.remove(name);
        hdu.writeto (name);
        print ('Save file '+name);

    return im;

def load_if_interp (filename=data_path+'/IFM_UNMR_2019-02-25T14-22-00.fits'):
    '''
    Return an interp handler of the IM with x in unit mm
    '''
    IF  = fits.getdata (filename);
    nx = IF.shape[-1];

    ym = (IF[120][63]+0.15) / 7.85 / 0.9976263337520707;
    xx = (np.arange (nx) - 63) / 5.21;
    ym[ym!=ym] = 0;
    ym = ym * np.exp (-(xx/5)**8) * (ym>0);

    return interp1d (xx*2.0-0.0548168481684817, ym, kind='quadratic',fill_value=0.0, bounds_error=False);

try:
    if_interp = load_if_interp();
except:
    print ('Not able to load if_interp');
    
def if_analytic (d, width=2.0):
    '''
    d: distance from actuator in [mm]
    width: size of IF in [mm], from NAOMI
    return normalized 
    '''
    return np.exp (-0.68*(d**2/width**2)**0.85);

def if_analytic_DM1 (d, width=2.0):
    '''
    d: distance from actuator in [mm]
    width: size of IF in [mm], from DM1 2022-06-08
    return normalized 
    '''
    return np.exp (-0.67152845*(d**2/width**2)**0.6862577);

def if_interaction (dx,dy,width=2e-6, if_function=if_analytic):
    '''
    dx,dy: distance from actuator in [mm]
    return the slope in x and y direction
    should be calibrated
    '''
    eps = width/2.0;
    d = np.sqrt (dx**2+dy**2) + 1e-10;
    amp = (if_function (d+eps) - if_function (d-eps)) / (2*eps);
    slopes = -np.array ([amp*dx/d, amp*dy/d]);
    return slopes;

def build_im (xm,xs,ys,xa,ya,flat=True,analytic=False):
    '''
    xs,ys,xa,ya in [mm] on DM
    '''
    
    # Coordinates
    n = len(xm);
    xx,yy = np.meshgrid (xm,xm);
    IM = np.zeros ((2,len(xs),len(xa)));
    IF = np.zeros ((n,n,len(xa)));
    
    # Loop on actuator
    for i,x,y in zip(range(len(xa)),xa,ya):
        if i%100 == 0: print ('IF:', i);
        d = np.sqrt ((xx - x)**2 + (yy - y)**2);
        IF[:,:,i] = if_interp (d);
        if analytic is True:
            IM[:,:,i] = if_interaction (xs-x,ys-y);
        else:
            IM[:,:,i] = wfs_measurement (IF[:,:,i], xm, xs, ys, 2.5);

    # Flatten
    if flat: IM = IM.reshape (2*len(xs),len(xa));
        
    return IM,IF;

def wfs_measurement (phase, xm, xp, yp, size, grad=True):
    '''
    xm is the 1D array of coordinates in pupil
    xp,yp are the 1D array of apperture position
    x and y in pixel
    convolution size in pixel
    Assume phase is rectangular pattern
    '''
    # FIXME: This is not great, as it create some
    # spatial histeris
    # x0 = np.digitize(xp,xm);
    # y0 = np.digitize(yp,xm);
    scale = xm[1] - xm[0];
    x0 = np.round ((xp-xm[0])/scale).astype(int);
    y0 = np.round ((yp-xm[0])/scale).astype(int);
    
    # Filter for aliasing
    tmp = uniform_filter (phase,int(size/scale));

    # Return phase or gradient
    if grad:
        gy,gx = np.gradient (tmp);
        return -np.array ([gx[y0,x0], gy[y0,x0]])/scale;
    else:
        return tmp[y0,x0];

def get_phase (screen, delta, nx):
    '''
    Extract a phase screen of size nx,nx from the input
    large screen, centered at the position delta (pix).
    The motion of outout screen inside the large screen
    is a circle.
    '''
    ns = screen.shape[0];
    rs = ns/2 - nx/2 - 3;
    x0 = int(ns/2 + rs * np.cos(delta/rs));
    y0 = int(ns/2 + rs * np.sin(delta/rs));
    return screen[x0-nx//2:x0-nx//2+nx,y0-nx//2:y0-nx//2+nx];

def rest_shape_DM0 (x,y=None):
    '''
    Load and interpolate the rest shape of DM0 prototype
    Scale of the file is 3.93 mm/pixel
    x and y should be in [mm] on the DM.

    FIXME: cannot make sens of the flip (no flip!!!!)
    when doing x -> -x
    '''
    # Unpack tupple
    if y is None: x,y = x;
    # Load data
    zz = np.loadtxt (data_path+'/DM1_rest_shape_05deg_2022-06-09.txt');
    zz[np.isnan(zz)] = 0.0;
    scale = 1./3.931052631578947;
    # Input Grid
    x0 = np.arange(len(zz)) * scale;
    x0 -= x0.mean();
    # Interp 2D
    return -interp2d(x0,x0,zz)(x,y);

def rest_shape_DM1 (x,y=None,scale=3.931052631578947):
    '''
    Load and interpolate the rest shape of DM1 (2023-08-10)
    Scale of the file is to be verified
    x and y should be in [mm] on the DM.

    FIXME: cannot make sens of the flip (no flip!!!!)
    when doing x -> -x
    '''
    # Unpack tupple
    if y is None: x,y = x;
    # Load data
    from scipy.io import loadmat;
    zz = loadmat('/Users/lebouquj/Datas/ALPAO/2023-08-18/Flat/Flat_T23.mat')['dataFlat23'][0][0][0];
    zz[np.isnan(zz)] = 0.0;
    # Recenter data
    zz = ndi.shift(zz, np.array(zz.shape)/2 - np.array(ndi.center_of_mass(zz!=0)));
    # Input Grid
    x0 = np.arange(len(zz)) / scale;
    x0 -= x0.mean();
    # Interp 2D
    return -interp2d(x0,x0,zz)(x,y)[::-1];

def build_pattern (n,w,Ri,Ro):
    '''
    Build a pattern of actuator or sub-apertures
    n: number of instance
    w: spacing
    Ri, Ro: inner and outer radius
    Ri and Ro can be scalars, or tupples of 2 values,
    one for x and one for y.
    '''
    # if len(Ri) == 1:
    Rix,Riy = Ri+1e-10,Ri+1e-10;
    # if len(Ro) == 1:
    Rox,Roy = Ro,Ro;
        
    x0 = 1.0 * np.arange (n);
    x0 -= x0.mean();
    x,y = np.meshgrid (w*x0, w*x0);
    
    # r2i = (x/Rix)**2 + (y/Riy)**2;
    # r2o = (x/Rox)**2 + (y/Roy)**2;
    # ok  = (r2i >= 1) * (r2o <= 1);
    rr = (x)**2 + (y)**2;
    ok  = (rr >= Ri**2) * (rr <= Ro**2);
    
    return x[ok], y[ok], ok;

def show_pattern(xa,ya,oka):
    '''
    Print the numbering of actuator in the pattern.
    Actuators >=1000 are printed in red, without the 
    firt digit.
    '''
    ii = range(len(xa));
    names = ['%i'%(i%1000) for i in ii];
        
    plt.figure(figsize=(7, 7));
    
    for x,y,n,i in zip(xa,ya,names,ii):
        color = 'r' if i>=1000 else 'k';
        plt.annotate(n, (x,y), ha='center', va='center',
                         size=4, color=color);
        
    mmax = 1.05 * np.maximum(np.abs(xa).max(),np.abs(ya).max());
    plt.xlim(-mmax,mmax);
    plt.ylim(-mmax,mmax);
    plt.gca().set_aspect('equal', adjustable='box');
    plt.margins(0,0);
    plt.show();
    
def sphere_dm_pattern():
    '''
    SPHERE-like 41x41 DM pattern
    with pitch 2.5mm. No central
    obscuration.
    '''
    return build_pattern (41,2.5,0,52.6);

def alpao41_dm_pattern(pitch=2.5):
    '''
    ALPAO proposal for 41x41
    '''
    return build_pattern (41,pitch,0,51.9*pitch/2.5);

def alpao42_dm_pattern():
    '''
    ALPAO proposal for 42x42
    '''
    return build_pattern (42,2.5,0,51.9);

def alpao43_dm_pattern(pitch=2.62,remove=738):
    '''
    ALPAO 43x43
    '''
    x,y,ok = build_pattern (43,pitch,0,55.8*pitch/2.62);
    l = ok[ok];
    l[remove] = False;
    x,y = x[l],y[l];
    ok[ok] = l;
    return x,y,ok;

def alpao11_dm_pattern():
    '''
    ALPAO pattern for the DM97-25
    '''
    return build_pattern (11,2.5,0,14);

def sphere_wfs_pattern():
    '''
    SPHERE-like 40x40 WFS pattern
    with pitch 2.5mm. Has a central
    obscuration.
    '''
    return build_pattern (40,2.5,7.2,50);

def lgs_wfs_pattern():
    '''
    30x30 WFS pattern with 704 subap
    across 100mm apperture
    '''
    return build_pattern (30,100./30,50*0.13,50*1.0)

def dsm_dm_pattern():
    '''
    DSM pattern (absolute scale not checked)
    '''
    c = fits.getdata (data_path+'DSM_pattern.fits').T;
    c = c / c.ptp(axis=1,keepdims=True);
    c = c - c.mean(axis=1,keepdims=True);
    c *= 100;
    return c[0], c[1], c[0]<1e10;

def alpao_power (dm,axis=-1):
    '''
    Return the max power per electronic, for the DM command dm,
    assuming the command is split in 2 electronic racks.
    '''
    tmp = np.moveaxis(dm,axis,0);
    limit = len(tmp)//2;
    return np.maximum((tmp[:limit]**2).sum(axis=0), (tmp[limit:]**2).sum(axis=0));

def binary_pup (n, Ro=1.0, Ri=0.1395, full=False):
    x = np.linspace (-1,1,n);
    xx,yy = np.meshgrid (x,x);
    dd = (xx*xx + yy*yy);
    pup = (dd >= Ri*Ri) * (dd <= Ro*Ro);
    if full: return pup,xx,yy;
    return pup;

def nan_pup (n, Ro=1.0, Ri=0.1395, full=False):
    x = np.linspace (-1,1,n);
    xx,yy = np.meshgrid (x,x);
    dd = (xx*xx + yy*yy);
    pup = 1.0 * (dd >= Ri*Ri) * (dd <= Ro*Ro);
    pup[pup<1] = np.nan;
    if full: return pup,xx,yy;
    return pup;

def as_screen (data,ok,nan=True):
    '''
    convert vector at screen
    '''
    if nan is True: img = np.nan * ok;
    else:           img = 0.0 * ok;
    img[ok] = data;
    return img;

def compute_KL (xa,ya,okc,filter_piston=False,forced_modes=None):
    '''
    compute KL modes with Eric Gendron method.
    xa,ya: coordinates of actuators
    okc: valid controlled actuators (~okc are slaved)
    return the KL[actuator,mode]

    If filter_piston is true, the piston mode (sum of all actuators)
    is filtered from the covariance matrix. As a consequence, the 
    focus mode is number 4 is a not a pure focus.

    If filter_piston is false, the piston apears as the last mode.
    The focus is number 2 and is quite pure, with some piston.
    '''
    if okc is None: okc = np.ones(xa.shape).astype(bool);
    n  = len(xa);
    no = okc.sum();
    
    # Covariance for all actuators
    Call = (xa[None,:]-xa[:,None])**2 + (ya[None,:]-ya[:,None])**2;
    Call = Call**(5./6);

    # Filter piston
    if filter_piston:
        P = np.zeros((n, n));
        P[:,:] = -1./n; np.fill_diagonal (P,1-1./n);
        Call = P @ (Call @ P.T);

    # Filter other modes
    if forced_modes is not None:
        print ('filter modes');
        P = np.identity(n) - forced_modes @ np.linalg.pinv(forced_modes);
        Call = P @ (Call @ P.T);
    else:
        print ('dont filter modes')
    
    # Covariance for active/active actuators
    Caa = Call[okc,:][:,okc];

    # Covariance for active/passive actuators
    Cpa = Call[~okc,:][:,okc];

    # Compute the eigenmodes
    G = np.linalg.eigh(Caa)[1];
    
    # Compute the passive command
    RG = (Cpa @ np.linalg.pinv(Caa,rcond=1e-6)) @ G;

    # Fill the forced mode
    KL = np.zeros ((len(xa),okc.sum()));
    if forced_modes is not None:
        nf = forced_modes.shape[1];
        KL[:,:nf] = forced_modes;
    else:
        nf = 0;

    # Fill in the active in the full KL
    KL[okc,nf:]  = G[:,:no-nf];

    # Fill in the passive up to maximum mode
    KL[~okc,nf:] = RG[:,:no-nf];

    return KL;

def filter_KL_edges (xa,ya,r,w):
    '''
    be weighted at radius r with w.
    '''
    d = np.sqrt(xa*xa + ya*ya);
    w = (np.cos(np.pi*(d-r)/w)+1)/2 * (d<=r+w) * (d>=r) + (d<r);
    return w[:,None];

def compute_KL_GPAO (nmodes=800,piston='subtract', returnOk=False, nact=41,
                     obs_fraction=0.1395, pup_fraction=0.95):
    '''
    Return the list of GPAO KL modes.
    '''
    # Pattern of the ALPAO DM
    if nact == 41:
        xt,yt,__ = alpao41_dm_pattern ();
    elif nact == 43:
        xt,yt,__ = alpao43_dm_pattern ();
    else:
        raise ValueError;
        
    rt  = np.sqrt (xt**2 + yt**2);
    
    # Active actuators
    limin, limout = obs_fraction * 50, pup_fraction * 50;
    okt = (rt > limin) * (rt < limout);

    # Filter piston from
    if piston == 'filter': filter_piston = True;
    else                 : filter_piston = False;
        
    # Approximate KL modes, break symetry
    kl = compute_KL (xt, yt * 1.001, okt, filter_piston=filter_piston);
    kl = kl[:,:nmodes];
    
    # Filter piston modes of active actuators
    if piston == 'subtract':
        kl -= kl[okt,:].mean(axis=0,keepdims=True);
    
    # Incorporate the trick of Henri ??
    # kl = np.max(kl)*100 - kl;    

    if returnOk: return kl, okt;
    return kl;

def compute_leak_GPAO (nmodes=800):
    '''
    Typical values of leak for a leaky integrator,
    versus the mode, defined as:
    a_i+1 = (1-leak) * a_i + K_i delta_i
    The leak is increased every 100
    modes, with value 0 for modes 0 to 100, and value
    0.17 for modes 700 to 800.
    See section 7.2 of https://arxiv.org/abs/1911.05989
    '''
    m = np.arange (nmodes);
    return (0.06*(m//100))**2;

def m1_natural_modes (x,y,phi=0.0):
    '''
    Load the natural elastic modes of M1 from document
    VLT-SPE-ESO-11110-0006

    x,y should be in [m] in M1 unit (8m)
    Modes are returned with STD = 1.0
    '''
    # Convert to radial
    rho   = np.expand_dims (np.sqrt(x**2 + y**2) / 4.0, axis=-1);
    theta = np.expand_dims (np.arctan2(y,x), axis=-1);
    
    # Load the modes
    data = np.genfromtxt(data_path+'/m1_modes_full.txt');

    # Normalize and rotate
    data[...,2] /= 1e3;

    # Phase of each quasi-Zernikes
    phase = data[:,2] * rho**data[:,1] * np.cos( data[:,0] * theta + np.pi*data[:,3] + phi);
    phase *= (rho <= 1.0);
    
    # Reform the natural modes
    s = list(phase.shape);
    s[-1] = -1;
    s.append(7);

    # Sum the quasi-zernikes for each modes
    out = phase.reshape(s).sum(axis=-1)
    return out;

def m1_control_modes (x,y,phi=0.0):
    '''
    Load the control modes of M1 from document
    ESO-252817

    x,y should be in [m] in M1 unit (8m)
    Modes are returned with STD = 1.0
    '''

    rho = np.sqrt(x**2 + y**2) / 4.0;
    
    # Compute the modes
    m1 = m1_natural_modes (x,y,phi=phi);
    
    # Mapping from ESO-252817
    C = np.array([-1,  3,  8, 16, -1,  5, 11,  1,  7, 14,  2, 10,  4, 13,  6,  9, 12, 15]);

    # Index in python, and also compute the non-existing modes but force them to zero
    m1 = m1[...,np.maximum(C-1,0)];
    m1 = m1 * (C>0);

    # Fill the piston and tip, with STD=1
    m1[...,0] = (rho <= 1.0);
    m1[...,4] = (x * np.cos(phi) + y * np.sin(phi)) / 4 * (rho <= 1.0) / 0.5;

    return m1;
    
def dsm_loworder_modes (x,y):
    '''
    Load the low order modes of DSM from document
    ESO-252817

    x,y should be in [m] in M1 unit (8m)
    Modes are returned with STD = 1.0
    '''

    # Control modes
    m1_s = m1_control_modes (x,y,phi=0.0);
    m1_p = m1_control_modes (x,y,phi=np.pi/2);
    
    # Mapping from ESO-252817
    n,p = np.array ([[2,  0.0],[3,  0.0],[4,  0.0],[6,  0.0],[6,  0.5],[7,  0.0],[7,  0.5],[8,  0.0],[8,  0.5],
         [9,  0.0],[9,  0.5],[10, 0.0],[10, 0.5],[11, 0.0],[11, 0.5],[12, 0.0],[12, 0.5],[13, 0.0],
         [13, 0.5],[14, 0.0],[14, 0.5],[15, 0.0],[15, 0.5],[16, 0.0],[16, 0.5],[17, 0.0],[17, 0.5],
         [18, 0.0],[18, 0.5]]).T;

    # Make integer
    n = n.astype(int) - 1;

    dsm = m1_s[...,n] * (p==0.0) + m1_p[...,n] * (p==0.5);
    return dsm;

def gpao_m1_offload_modes (x,y):
    '''
    Load the low order modes of DM from document
    ESO-252817, but including tip/tilt as first modes.

    PROPOSAL TO BE VALIDATED

    x,y should be in [m] in M1 unit (8m)
    Modes are returned with STD = 1.0
    '''
    
    # Mapping from ESO-252817 updated for GPAO
    n,p = np.array ([
         [5,  0.0],[5,  0.5],
         [2,  0.0],[3,  0.0],[4,  0.0],
         [6,  0.0],[6,  0.5],[7,  0.0],[7,  0.5],[8,  0.0],[8,  0.5],
         [9,  0.0],[9,  0.5],[10, 0.0],[10, 0.5],[11, 0.0],[11, 0.5],[12, 0.0],[12, 0.5],[13, 0.0],
         [13, 0.5],[14, 0.0],[14, 0.5],[15, 0.0],[15, 0.5],[16, 0.0],[16, 0.5],[17, 0.0],[17, 0.5],
         [18, 0.0],[18, 0.5]]).T;

    # Control modes
    m1_s = m1_control_modes (x,y,phi=0.0);
    m1_p = m1_control_modes (x,y,phi=np.pi/2);

    # Make integer
    n = n.astype(int) - 1;

    dm = m1_s[...,n] * (p==0.0) + m1_p[...,n] * (p==0.5);
    return dm;

def invers (M, n=None, returnMatrix=False):
    '''
    Inverse by SVD,
    filtering modes above n
    '''
    # SVD
    u, s, vh = np.linalg.svd (M, full_matrices=True);
    # Filter modes
    if n is not None: s[n:] = 0;
    s[s!=0] = 1./s[s!=0];
    # Reconstruct
    vhs = (vh.T @ np.diag (s));
    if returnMatrix is True:
        out = vhs, u.T[:len(s),:];
    else:
        out = vhs @ u.T[:len(s),:];
    return out;
    
def toHstr(std,lbd=1.55):
    '''
    Return Strehl from Marechal approx with lbd at 1.55um
    '''
    return np.exp (-(std/lbd)**2);

class Controller:
    '''
    Controller parameters
    '''
    Ki = 0.3;
    nmodes = 1000;
    pmodes = 0;
    delay_wfs = 1;
    delay_dm = 1;
    pass;

def run_loop (pertu, IM, S2M, M2A, params, verb=100, offset=0, shift=0):
    '''
    Run the loop on pertu
    Controller parameters are in params
    gaina is the gain of individual actuator
    '''
    print ('run loop');

    nt = pertu.shape[-1];
    na,nm = M2A.shape;
    ns = int(pertu.shape[0]/2);

    print ('%i act, %i modes, %i:%i, Ki = %.2f'%(na,
            nm, params.pmodes,params.nmodes,params.Ki));

    # Check consistency
    if IM.shape != (ns*2,na):  raise(IOError('IM shape mismatch'));
    if S2M.shape != (nm,ns*2): raise(IOError('S2M shape mismatch'));
    if M2A.shape != (na,nm): raise(IOError('M2C shape mismatch'));

    # Phase screen and DM shape in WFS space
    corr = np.zeros ((2*ns,nt));
    ress = np.zeros ((2*ns,nt));
    cmdm = np.zeros ((nm,nt));
    resm = np.zeros ((nm,nt));
    dmps = np.zeros ((na,nt));
    resa = np.zeros ((na,nt));
    cmda = np.zeros ((na,nt));

    for t in range (3,nt):
        if t%verb == 0: print (t);

        # DM true position, including DM delay
        dmps[:,t] = cmda[:,t-params.delay_dm-1];
    
        # Measurements of residuals,
        # including camera delay
        corr[:,t] = IM @ dmps[:,t];
        ress[:,t] = pertu[:,t-params.delay_wfs] - corr[:,t-params.delay_wfs];

        # Residual in controlled space
        resm[:,t] = (S2M @ ress[:,t]);

        # Filter first and last modes
        resm[:params.pmodes,t] = 0.0;
        resm[params.nmodes:,t] = 0.0;

        # Controller with pur integrator
        cmdm[:,t] = cmdm[:,t-1] + params.Ki * resm[:,t] + shift;

        # Command in actuator space
        cmda[:,t] = (M2A @ cmdm[:,t]) + offset;

    return cmdm,dmps,ress,resm;

def rotation (re,im,angle):
    '''
    Return the output complex rotated by angle in deg.
    '''
    cang,sang = np.cos(angle*np.pi/180),np.sin(angle*np.pi/180);
    return cang * re + sang * im, cang * im - sang * re;

def rotate_shift (img, angle=0, shifts=0):
    """
    Rotate and shift the image
    ------
    Args:
        angle in deg
        shift in pixel
    """
    import cv2;

    shifts = np.ones(2) * shifts;
    shape  = img.shape;
    
    matrix = cv2.getRotationMatrix2D (shape/2, angle, 1);
    matrix[:,2] += shifts;
    
    return cv2.warpAffine (img,matrix,shape);
    

def zoom_clipped (img, zoom_factor=0):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    
    This is much faster than RectBivariateSpline or interp2d:
    x = np.linspace(-2,2,256);
    img =1.0*( (x[:,None]**2 + x[None,:]**2)<1);
    %timeit md.zoom_clipped(img,1.5);
    %timeit RectBivariateSpline(x,x,img)(x/1.5,x/1.5);

    ------
    Args:
        img : ndarray
            Image array
        zoom_factor : float
            amount of zoom as a ratio [1 to Inf)
    ------
    Returns:
        result: ndarray
           numpy ndarray of the same shape of the input img zoomed by the specified factor.          
    """
    import cv2;
    
    assert zoom_factor >= 1.0, "zoom_factor should be greater than 1.0"
    if zoom_factor == 1: return img;

    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    
    ### Crop only the part that will remain in the result (more efficient)
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]

    resize_height, resize_width = min(new_height, height), min(new_width, width)
    result = cv2.resize(cropped_img, (resize_width, resize_height))
    assert result.shape[0] == height and result.shape[1] == width
    return result

def savesim (basename, obj):
    '''
    Pickle save
    '''
    filename = tmp_path+'/'+basename+'_'+str(Time(Time.now(),format='isot'))[:-4]+'.pckl'
    print ('Save:');
    print (filename);
    f = open (filename, 'wb');
    pickle.dump (obj, f);
    f.close();
    print ('%.3f Mb'%(os.path.getsize(filename)*1e-6));
    return filename;

def loadsim (filename):
    '''
    Pickle load
    '''
    print ('Load:');
    print (filename);
    f = open (filename, 'rb');
    obj = pickle.load (f);
    f.close();
    return obj;

def zernike_rad (m, n, rho, crop=True):
    if ((n-m) % 2): return rho*0.0
    pre_fac = lambda k: (-1.0)**k * fac(n-k) / ( fac(k) * fac( (n+m)/2.0 - k ) * fac( (n-m)/2.0 - k ) )
    output = sum(pre_fac(k) * rho**(n-2.0*k) for k in range((n-m)//2+1))
    if crop: output = output * (rho<=1.0);
    return output;

def zernike (j, x, y, crop=True):
    '''
    Return the Zernike polynomial of number j
    estimated in cartesian coordinates x,y.
    j should be integer. x,y can be arrays.
    The returned zernike are from -1 to +1, that
    is normalised in ptp, not rms
    '''
    
    # Get the m,n from the index. Not sure about the
    # sorting, probably OSA/ANSI
    n = 0
    while (j > n):
        n += 1
        j -= n
    m = -n+2*j

# 	n = int(np.sqrt(2 * j - 1) + 0.5) - 1
# 	if n % 2:
# 		m = 2 * int((2 * (j + 1) - n * (n + 1)) // 4) - 1
# 	else:
# 		m = 2 * int((2 * j + 1 - n * (n + 1)) // 4)
# 	m = m * (-1)**(i % 2)    
    
    # Compute radial coordinates
    rho = np.sqrt(x**2 + y**2);
    phi = np.arctan2(y,x);

    # Compute Zernike
    if (m > 0): return zernike_rad(m, n, rho, crop=crop) * np.cos(m * phi)
    if (m < 0): return zernike_rad(-m, n, rho, crop=crop) * np.sin(-m * phi)
    return zernike_rad(0, n, rho, crop=crop)

def zernike_noll (j, x, y, crop=True):
    '''
    Return the Zernike polynomial of number j
    estimated in cartesian coordinates x,y.
    j should be integer. x,y can be arrays.
    The returned zernike are from -1 to +1, that
    is normalised in ptp, not rms
    '''
    
    n = int((-1.+np.sqrt(8*(j-1)+1))/2.)
    p = (j-(n*(n+1))/2.)
    k = n%2
    m = int((p+k)/2.)*2 - k

    if m!=0:
        if j%2==0:
            s=1
        else:
            s=-1
        m *= s
    
    # Compute radial coordinates
    rho = np.sqrt(x**2 + y**2);
    phi = np.arctan2(y,x);

    # Compute Zernike
    if (m > 0): return zernike_rad(m, n, rho, crop=crop) * np.cos(m * phi)
    if (m < 0): return zernike_rad(-m, n, rho, crop=crop) * np.sin(-m * phi)
    return zernike_rad(0, n, rho, crop=crop)


'''

'''
