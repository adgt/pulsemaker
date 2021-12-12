import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import traitlets as traitlets

def _wf_eq(wf_type, amp, sigma, cycles, width, t):
    samps = t.size
    if wf_type == 'Gaussian':
        sig = amp * np.exp(-0.5 * ((t - max(t) / 2) / sigma) ** 2)

    elif wf_type == 'Gaussian Derivative':
        sig = -(t - max(t) / 2) * np.exp(-0.5 * ((t - max(t) / 2) / sigma) ** 2)
        sig = amp * sig / max(sig)

    elif wf_type == 'Gaussian Sine':
        sig = -1 * amp * np.sin(2 * np.pi * cycles * t - np.pi)
        exp = np.exp(-0.5 * ((t - max(t) / 2) / sigma) ** 2)
        sig = sig * exp

    elif wf_type == 'Gaussian Cosine':
        sig = -1 * amp * np.cos(2 * np.pi * cycles * t - np.pi)
        exp = np.exp(-0.5 * ((t - max(t) / 2) / sigma) ** 2)
        sig = sig * exp
    
    elif wf_type == 'Gaussian Square':
        # Depending on what updates first (fig_out or width sliders) there is a
        # change width could be temporarily smaller than # of samples, creating an error
        # this shouldn't happen if there was only one .observe per widget, but since I am using
        # .interative_output(). To fix this, need to strip that function and create my own where
        # .observes() happen one by one (and only once) and call functions that update everything that
        # needs to be updated by that widget.
        # In the meantime, setting width=samps if width is smaller than samps is an easy fix.
        if width > samps: 
            width = samps      
        risefall = (samps - width)//2
        t1 = np.linspace(0,2*risefall,2*risefall)
        sig1 = amp * np.exp(-0.5 * ((t1 - max(t1) / 2) / sigma) ** 2)

        sig=sig1[:risefall+1]
        sig=np.append(sig,amp*np.ones(width))
        sig=np.append(sig,sig1[risefall+1:])
        
    elif wf_type == 'DRAG':
        pass  
    
    return sig

def _sig_gen(i_wf, q_wf, samples, i_amp, q_amp, i_width, q_width, i_cycles, q_cycles, i_sigma, q_sigma, beta):
    
    width = 1 # scaling in x. For sine/cos is equivalent to frequency
    
    t = np.linspace(0,samples,samples)
    
    if i_wf == 'DRAG' or q_wf == 'DRAG':
        i_wf = 'Gaussian'
        q_wf = 'Gaussian Derivative'
        q_amp = q_amp * beta

    i_sig = _wf_eq(i_wf,i_amp,i_sigma,i_cycles,i_width,t)
    q_sig = _wf_eq(q_wf,q_amp,q_sigma,q_cycles,q_width,t)

    return t, i_sig, q_sig

# Waveform plotting function
def _plot_wf(i_wf, q_wf, samples, i_amp, q_amp, i_width, q_width, i_cycles, q_cycles, i_sigma, q_sigma, beta):
    
    t, i_sig, q_sig = _sig_gen(i_wf, q_wf, samples, i_amp, q_amp, i_width, q_width, i_cycles, q_cycles, i_sigma, q_sigma, beta)
    
    _, axs = plt.subplots(2, sharex=True, figsize=(7,5), tight_layout=True)
    axs[1].set_xlabel('Samples (t/dt)')
    axs[0].set_ylabel('In-Phase Amplitude')
    axs[1].set_ylabel('Quadrature Amplitude')
    axs[0].set_xlim(0, samples)
    axs[0].set_ylim(-(abs(i_amp)+0.1),abs(i_amp)+0.1)
    axs[1].set_ylim(-(abs(q_amp)+0.1),abs(q_amp)+0.1)
    axs[0].step(t, i_sig, 'r')
    axs[1].step(t, q_sig, 'b')

    '''
    TODO: 
    1. Need to decide how to set the min res of y axes (set_ylim) based on backend amp resolution
    Also need to do yset_ticks so plot is not jumping around when changing amplitude
    2. To update quadrature max amplitude, need info about the in-phase signal. Can't seem to be able
    to get this info by using .interactive_output() and constructing the signals inside the passed function.
    Therefore, might need to have the signal construction function outside, and pass numpy arrays thru 
    interactive_output() as fixed widgets (might be better this way).
    '''

# Function that returns maximum allowable quadrature amplitude
def find_qamp_max(i_sign, q_sign):
    qamp_max = 1.0
    r_sign = np.sqrt(i_sign**2+q_sign**2) # Compute magnitude array of signal
    r_max_val = max(r_sign)               # Find maximum of the abs of magnitude 
                                          # (only positive, so already accounts for valleys)
    
    while r_max_val > 1:
        # reduce qamp_max until max of magnitude is less than 1
        
        r_max_ind = [i for i, x in enumerate(abs(r_sign)) if x == r_max_val] # indices of maxima in mag of signal
        
        iamp_max = max(abs(i_sign[r_max_ind])) # evaluate in-phase sig at indices of max of signal magnitude,
                                               # and find largest value
            
        qamp_max = np.sqrt(1-iamp_max**2)      # qamp max needed so norm of I and Q is not larger than 1
        
        # recalculate r_max_val
        r_sign = np.sqrt(i_sign**2+(qamp_max*q_sign)**2)
        r_max_val = max(r_sign)
        
    return qamp_max


class PulseDesigner(widgets.VBox, traitlets.HasTraits):
    pulse = traitlets.List([])

    def setup_config(self):

        # Parameters
        self.waveform_lst = ['Gaussian', 'Gaussian Derivative', 'Gaussian Sine', 'Gaussian Cosine', 'Gaussian Square','DRAG']  # Custom waveforms available

        '''TO DO: Find out what samples and amplitude resolution is for each backend'''
        self.amp_res_val = 0.01 # Amplitude resolution
#         self.samples_val = 1000  # Number of initial samples
        self.amp_val = 1
        self.samples_val = 2700
        self.sigma_val = 100

        self.drag_links = []

        #self.i_sig_arr = np.zeros(self.samples_val) # Array with in-phase signal amplitudes
        #self.q_sig_arr = np.zeros(self.samples_val) # Array with quadrature signal amplitudes

        ### Fixed parameters ###

        # fixed widget elements to manipulate in-phase and quadrature amplitude arrays
        #self.i_sig_fxd = widgets.fixed(value=self.i_sig_arr)
        #self.q_sig_fxd = widgets.fixed(value=self.q_sig_arr)

        # Maximum amplitude for quadrature signal. Needs to be interactive bc depends on in-phase amplitude
        #self.q_amp_max_fxd = widgets.fixed(value=0.0)

    def setup_custom_waveform_input(self):

        # In-Phase waveform components

        self.i_wf_dd = widgets.Dropdown(
            options=self.waveform_lst, 
            layout=widgets.Layout(width='auto'),
            description='In-Phase:',
        )

        self.i_sigma_sldr = widgets.IntSlider(
            value=self.sigma_val, 
            description='Sigma', 
            min=1, max=max(1000,self.sigma_val), step=1,
            continuous_update=False
        )

        # Slider for in-phase amplitude (from -1 to 1)
        self.i_amp_sldr = widgets.FloatSlider(
            value=self.amp_val, 
            description='Amplitude', 
            min=-1, max=1, step=self.amp_res_val,
            continuous_update=False,
        )

        self.i_cycles_slider = widgets.IntSlider(
            value=1,
            description='Periods',
            min=1, max=30, step=1,
            continuous_update=False
        )

        self.i_width_sldr = widgets.IntSlider(
            value=self.samples_val//2, 
            description='Width', 
            min=0, max=self.samples_val, step=2,
            continuous_update=False
        )

        self.beta_slider = widgets.FloatSlider(
            value=0, 
            description='Beta', 
            min=-1, max=1, step=0.05,
            continuous_update=False
        )

        self.i_gaussian_controls = [self.i_wf_dd, self.i_sigma_sldr, self.i_amp_sldr]
        self.i_gaussian_sine_cosine_controls = [self.i_wf_dd, self.i_sigma_sldr, self.i_amp_sldr, self.i_cycles_slider]
        self.i_gaussian_square_controls = [self.i_wf_dd, self.i_sigma_sldr, self.i_amp_sldr, self.i_width_sldr]
        self.i_drag_controls = [self.i_wf_dd, self.i_sigma_sldr, self.i_amp_sldr, self.beta_slider]

        #---------------------------------------------------------------------------------------------------------------------

        # Dropdown menu for type of quadrature wavefunction (enabled when custom waveform is selected)
        self.q_wf_dd = widgets.Dropdown(
            options=self.waveform_lst,
            layout=widgets.Layout(width='auto'),
            description='Quadrature:',
        )

        self.q_sigma_sldr = widgets.IntSlider(
            value=np.floor(self.sigma_val/2), 
            description='Sigma', 
            min=1, max=max(100,self.sigma_val), step=1,
            continuous_update=False
        )

        # Slider for quadrature amplitude (Starts from -1 to 1, but gets updated based on value of in-phase amp)
        # This is because Pulse only accepts steps with max amplitude of R(I,Q) = 1
        self.q_amp_sldr = widgets.FloatSlider(
            value=0, 
            description='Amplitude', 
            min=-1, max=1, step=self.amp_res_val,
            continuous_update=False,
        )

        self.q_cycles_slider = widgets.IntSlider(
            value=1,
            description='Periods',
            min=1, max=30, step=1,
            continuous_update=False
        )

        self.q_width_sldr = widgets.IntSlider(
            value=self.samples_val//2, 
            description='Width', 
            min=0, max=self.samples_val, step=2,
            continuous_update=False
        )

        self.q_gaussian_controls = [self.q_wf_dd, self.q_sigma_sldr, self.q_amp_sldr]
        self.q_gaussian_sine_cosine_controls = [self.q_wf_dd, self.q_sigma_sldr, self.q_amp_sldr, self.q_cycles_slider]
        self.q_gaussian_square_controls = [self.q_wf_dd, self.q_sigma_sldr, self.q_amp_sldr, self.q_width_sldr]
        self.q_drag_controls = [self.q_wf_dd, self.q_sigma_sldr, self.q_amp_sldr, self.beta_slider]


        # ------------------------------------------------------------------------------------

        self.samples_sldr = widgets.IntSlider(
            value=self.samples_val, 
            description='Samples:', 
            min=16, max=5120, 
            step=16,
            continuous_update=False            
        )
        
        #-------------------------------------------------------------------------------------------------------------------------------

        flex_layout = widgets.Layout(display='flex', flex_flow='row')
        self.in_phase_hbox = widgets.HBox(self.i_gaussian_controls, layout=flex_layout)
        self.quadrature_hbox = widgets.HBox(self.q_gaussian_controls, layout=flex_layout)

        self.designer = widgets.VBox([
            self.in_phase_hbox,
            self.quadrature_hbox,
            self.samples_sldr
        ])

    def setup_event_handlers(self):

        def update_inphase_controls(change):
            new_wf_type = change['new']
            old_wf_type = change['old']
            if new_wf_type == 'Gaussian Sine' or new_wf_type == 'Gaussian Cosine':
                self.in_phase_hbox.children = self.i_gaussian_sine_cosine_controls
            elif new_wf_type == 'Gaussian' or new_wf_type == 'Gaussian Derivative':
                self.in_phase_hbox.children = self.i_gaussian_controls
            elif new_wf_type == 'Gaussian Square':
                self.in_phase_hbox.children = self.i_gaussian_square_controls
            elif new_wf_type == 'DRAG':
                self.in_phase_hbox.children = self.i_drag_controls
                self.quadrature_hbox.children, self.q_wf_dd.value = self.q_drag_controls, 'DRAG'
                self.drag_links += [
                    widgets.link((self.i_sigma_sldr, 'value'), (self.q_sigma_sldr, 'value')),
                    widgets.link((self.i_amp_sldr, 'value'), (self.q_amp_sldr, 'value')),
                ]
            if old_wf_type == 'DRAG':
                if len(self.drag_links) > 0:
                    for link in self.drag_links:
                        link.unlink()
                    self.drag_links = []
                    self.q_wf_dd.value = 'Gaussian'
                

        self.i_wf_dd.observe(update_inphase_controls, 'value')


        def update_quadrature_controls(change):
            new_wf_type = change['new']
            old_wf_type = change['old']
            if new_wf_type == 'Gaussian Sine' or new_wf_type == 'Gaussian Cosine':
                self.quadrature_hbox.children = self.q_gaussian_sine_cosine_controls
            elif new_wf_type == 'Gaussian' or new_wf_type == 'Gaussian Derivative':
                self.quadrature_hbox.children = self.q_gaussian_controls
            elif new_wf_type == 'Gaussian Square':
                self.quadrature_hbox.children = self.q_gaussian_square_controls
            elif new_wf_type == 'DRAG':
                self.quadrature_hbox.children = self.q_drag_controls
                self.in_phase_hbox.children, self.i_wf_dd.value = self.i_drag_controls, 'DRAG'
                self.drag_links += [
                    widgets.link((self.q_sigma_sldr, 'value'), (self.i_sigma_sldr, 'value')),
                    widgets.link((self.q_amp_sldr, 'value'), (self.i_amp_sldr, 'value')),
                ]

                
            if old_wf_type == 'DRAG':
                if self.drag_links is not None and len(self.drag_links) > 0:
                    for link in self.drag_links:
                        link.unlink()
                    self.drag_links = []
                    self.i_wf_dd.value = 'Gaussian'


        self.q_wf_dd.observe(update_quadrature_controls, 'value')

        # Updates max allowable value of Quadrature amplitude
        def update_qamp_max(*args):

            # Here to calculate q_sig, I am passing amplitude of 1.0 because q_sig has to be normalized
            # so the q_amp_max value is calculated correctly.
            _, i_sig, q_sig = _sig_gen(self.i_wf_dd.value, self.q_wf_dd.value, 
                                    self.samples_sldr.value, self.i_amp_sldr.value, 1.0,
                                    self.i_width_sldr.value, self.q_width_sldr.value,
                                    self.i_cycles_slider.value, self.q_cycles_slider.value,
                                    self.i_sigma_sldr.value, self.q_sigma_sldr.value,
                                    self.beta_slider.value)
            beta = 1
            if self.i_wf_dd.value == 'DRAG' or self.q_wf_dd == 'DRAG':
                beta = self.beta_slider.value
            self.pulse = list(i_sig + 1j * beta * q_sig)
            q_amp_max = find_qamp_max(i_sig,q_sig)

            if q_amp_max < self.amp_res_val:
                self.q_amp_sldr.value = 0
                self.q_amp_sldr.disabled = True
            else:
                self.q_amp_sldr.max = q_amp_max
                self.q_amp_sldr.min = -q_amp_max
                self.q_amp_sldr.disabled = False

        self.i_wf_dd.observe(update_qamp_max, 'value')
        self.q_wf_dd.observe(update_qamp_max, 'value')
        self.samples_sldr.observe(update_qamp_max, 'value')
        self.i_amp_sldr.observe(update_qamp_max, 'value')
        self.q_amp_sldr.observe(update_qamp_max, 'value')

        def on_samples_changed(*args):
            self.samples_val = self.samples_sldr.value
            if self.i_wf_dd.value == 'Gaussian Square':
                self.i_width_sldr.min = 0
                self.i_width_sldr.max = self.samples_sldr.value
                if self.i_width_sldr.value > self.i_width_sldr.max:
                    self.i_width_sldr.value = self.i_width_sldr.max
                self.i_width_sldr.step=2
            elif self.i_wf_dd.value == 'Gaussian' or self.i_wf_dd.value == 'Gaussian Derivative':
                #self.i_sigma_sldr.value = self.samples_sldr.value//10
                self.i_sigma_sldr.min = 1
                self.i_sigma_sldr.max = self.samples_sldr.value
                if self.i_sigma_sldr.value > self.i_sigma_sldr.max:
                    self.i_sigma_sldr.value = self.i_sigma_sldr.max

            if self.q_wf_dd.value == 'Gaussian Square':
                self.q_width_sldr.min = 0
                self.q_width_sldr.max = self.samples_sldr.value
                if self.q_width_sldr.value > self.q_width_sldr.max:
                    self.q_width_sldr.value = self.q_width_sldr.max
                self.q_width_sldr.step=2
            elif self.q_wf_dd.value == 'Gaussian' or self.q_wf_dd.value == 'Gaussian Derivative':
                #self.q_sigma_sldr.value = self.samples_sldr.value//10
                self.q_sigma_sldr.min = 1
                self.q_sigma_sldr.max = self.samples_sldr.value
                if self.q_sigma_sldr.value > self.q_sigma_sldr.max:
                    self.q_sigma_sldr.value = self.q_sigma_sldr.max
                
        self.samples_sldr.observe(on_samples_changed, 'value')
        
        self.i_wf_dd.observe(self.update_pulse, 'value')
        self.q_wf_dd.observe(self.update_pulse, 'value')
        self.samples_sldr.observe(self.update_pulse, 'value')
        self.i_amp_sldr.observe(self.update_pulse, 'value')
        self.q_amp_sldr.observe(self.update_pulse, 'value')
        self.i_width_sldr.observe(self.update_pulse, 'value')
        self.q_width_sldr.observe(self.update_pulse, 'value')

    def plot_figure(self):
        self.fig_out = widgets.interactive_output(_plot_wf, 
        {
            'i_wf':self.i_wf_dd, 
            'q_wf':self.q_wf_dd,
            'samples':self.samples_sldr,
            'i_amp':self.i_amp_sldr,
            'q_amp':self.q_amp_sldr,
            'i_width':self.i_width_sldr,
            'q_width':self.q_width_sldr,
            'i_cycles': self.i_cycles_slider,
            'q_cycles': self.q_cycles_slider,
            'i_sigma': self.i_sigma_sldr,
            'q_sigma':self.q_sigma_sldr,
            'beta': self.beta_slider
        })
        
    def update_pulse(self, change):
        _, i_sig, q_sig = _sig_gen(self.i_wf_dd.value, self.q_wf_dd.value, 
                                self.samples_sldr.value, self.i_amp_sldr.value, self.q_amp_sldr.value,
                                self.i_width_sldr.value, self.q_width_sldr.value,
                                self.i_cycles_slider.value, self.q_cycles_slider.value,
                                self.i_sigma_sldr.value, self.q_sigma_sldr.value,
                                self.beta_slider.value)
        beta = 1
        if self.i_wf_dd.value == 'DRAG' or self.q_wf_dd == 'DRAG':
            beta = self.beta_slider.value
        self.pulse = list(i_sig + 1j * beta * q_sig)

    def __init__(self):
        self.setup_config()
        self.setup_custom_waveform_input()
        self.setup_event_handlers()
        self.plot_figure()

        super().__init__([
            self.designer,
            self.fig_out,
        ])
        
        self.update_pulse(None)