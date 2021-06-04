import cvxpy as cvx
from pyCSalgos.sl0 import SmoothedL0
import numpy as np
import scipy as sc
import scipy.signal as scsig
import scipy.optimize as sco

def compressed_sensing(self, site=0, symmetric=False, method='L1', l=0.01, eps=1., show=False):
    """
    Computes the compressed sensing method.
    """
    # Gets the necessary Fourier parameters
    try:
        tmax = self.fourier_parameters[site]['t_max']
        n_times = self.fourier_parameters[site]['n_measurements']
        n0 = self.fourier_parameters[site]['n0']
        n1 = self.fourier_parameters[site]['n1']
        dt = self.fourier_parameters[site]['dt']
        center = self.fourier_parameters[site]['center_point']
        nz_range = self.fourier_parameters[site]['nonzero_range']
    except KeyError:
        print('set_fourier_parameters was not properly called.')
        raise

    ri = self.ri[site]
    Gt_sparse = self.green_function_time[site]

    # If A(omega) is known to be symmetric a priori, the problem is simplified by using a
    # transformation matrix A with cosines instead, with half the size.

    if symmetric:
        Gt_sparse = Gt_sparse.imag
        A = 2*np.cos(2.*np.pi*ri.reshape((-1,1))*np.arange(1,nz_range)[::-1]/n1) / (dt*n1)
        vx = cvx.Variable(nz_range-1)
    
    else:
        A = np.exp(2.*np.pi*1.j*ri.reshape((-1,1))*np.flip(np.r_[n1-nz_range+1:n1, 0:nz_range])/n1) / (dt*n1)
        vx = cvx.Variable(2*nz_range-1, complex=True)

    # show CS parameters
    print('A:', A.shape, '\tGw:', 2*nz_range, '\tM:', n_times, '\tRmax:', n0, '\ttmax:', tmax) if show else None

    # Uses `Smoothed L0` as in http://ee.sharif.edu/~SLzero/
    if method == 'SL0':
        sl0 = SmoothedL0(eps)
        Gw_sparse = sl0.solve(Gt_sparse, A)

    else:

        # Uses `LL1` method in Eq. (Llp) as in 10.1016/j.neucom.2013.03.017
        if method == 'L1':
            objective = cvx.Minimize(l*cvx.norm(vx, 1) + cvx.norm(A@vx - Gt_sparse, 2)/2)
            prob = cvx.Problem(objective)

        # Uses `LL1` method in Eq. (Nlp) as in 10.1016/j.neucom.2013.03.017
        elif method == 'NL1':
            constraints = [cvx.sum_squares(A@vx - Gt_sparse) <= eps]
            objective = cvx.Minimize(cvx.norm(vx, 1))
            prob = cvx.Problem(objective, constraints)

        # Uses `LL1` method in Eq. (Elp) as in 10.1016/j.neucom.2013.03.017
        elif method == 'EL1':
            constraints = [A@vx == Gt_sparse]
            objective = cvx.Minimize(cvx.norm(vx, 1))
            prob = cvx.Problem(objective, constraints)

        # Run convex optimization
        result = prob.solve(solver=cvx.SCS, verbose=True if show else False)
        assert prob.status == cvx.OPTIMAL or prob.status == cvx.OPTIMAL_INACCURATE, 'Solution not found!'
    
        # # reconstruct signal
        Gw_sparse = np.array(vx.value)
    
    if show:
        norm = np.sum(np.abs(Gw_sparse))
        eq = np.sum(np.abs(A@Gw_sparse - Gt_sparse)**2)/2
        print("Norm: {}, Deviation: {}, G(t):{}".format(norm,eq, np.sum(np.abs(Gt_sparse)**2)/2))

    Gw_sparse = np.squeeze(Gw_sparse)

    if symmetric:
        # Accounts for the lowest frequency and 0, both considered 0.
        Gw_sparse = 1j* np.concatenate(([0.], Gw_sparse, [0.], np.flip(Gw_sparse)))
    else:
        # Accounts for the lowest frequency, considered 0.
        Gw_sparse = np.concatenate(([0.+0.j], Gw_sparse))

    return Gw_sparse



def get_A(self, freqs=None, Gw=None, site=0, regularization=False, force_sym=False, show=False):
    """
    Convert compressed sensing array into an actual function.
    """

    if not regularization:
        A = sc.interpolate.interp1d(freqs, -1/np.pi * Gw.imag, kind='slinear')
    else:
        res = scsig.find_peaks(np.abs(Gw), height=3, width=1)
        w_int = freqs.shape[0]

        peaks = (res[1]['peak_heights'] * res[1]['widths']) * 2/w_int
        pos = ((res[1]['left_ips']+res[1]['right_ips'])/2)*(4*np.pi)/w_int - 2*np.pi
        pepos = peaks/pos
        n_peaks = peaks.shape[0]

        print("(Peak, frequency): {}".format(list(zip(peaks, pos))))
        print('Found',n_peaks,'peaks in Green\'s function.') if show else None

        new_peaks, new_pos = peaks, pos

        # Force symmetry
        if force_sym:
            new_peaks += new_peaks[::-1]
            new_pos -= new_pos[::-1]
            new_peaks /= 2
            new_pos /= 2

        A = lambda w: 1/np.pi * np.sum(new_peaks * Delta(w-new_pos))

    return A


def compute_spectral_function(self, site=0, method='L1', symmetric=False, force_sym=True, regularization=False, l=0.01, eps=1., show=False):

    nz = self.fourier_parameters[site]['nonzero_range']
    freqs = self.freqs[site]

    if method == 'fourier':
        assert self.fourier_parameters[site]['n_measurements']==self.fourier_parameters[site]['n0'], 'For the`fourier` method, the G(t) measurements must be uniform and evenly spaced (m = n0).'

        delta_t = self.fourier_parameters[site]['dt']
        cp = self.fourier_parameters[site]['center_point']

        tmp = sc.fft.fft(self.green_function_time[site])
        green_function_freq = delta_t * sc.fft.fftshift(tmp)[::-1]
        green_function_freq = green_function_freq[cp-nz:cp+nz]

    else:
        
        green_function_freq = compressed_sensing(self, site=site, symmetric=symmetric, method=method, l=l, eps=eps, show=show)
    
    A = get_A(self, freqs=freqs, Gw=green_function_freq, regularization=regularization, force_sym=force_sym, show=show)

    return green_function_freq, A
    

def Delta(x, a = 1e-1):
    return np.exp(-(x/a)**2)/(np.abs(a)*np.sqrt(np.pi))