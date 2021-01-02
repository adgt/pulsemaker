import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

def _wf_eq(wf_type, amp, width, t):
    
    samps = t.size
    
    if wf_type == 'Rect':
        # Depending on what updates first (fig_out or width sliders) there is a
        # change width could be temporarily smaller than # of samples, creating an error
        # this shouldn't happen if there was only one .observe per widget, but since I am using
        # .interative_output(). To fix this, need to strip that function and create my own where
        # .observes() happen one by one (and only once) and call functions that update everything that
        # needs to be updated by that widget.
        # In the meantime, setting width=samps if width is smaller than samps is an easy fix.
        if width > samps: 
            width = samps      
        init_del = int(np.floor((samps-width)/2))
        fin_del = samps-width-init_del
        sig = np.zeros(init_del)
        sig = np.append(sig,amp*np.ones(width))
        sig = np.append(sig, np.zeros(fin_del))
        
    elif wf_type == 'Sine':
        sig = amp*np.sin(2*np.pi*width*t-np.pi)
        
    elif wf_type == 'Cosine':
        sig = amp*np.cos(2*np.pi*width*t-np.pi)
        
    elif wf_type == 'Gaussian':
        sig = amp*np.exp(-1/2*((t-max(t)/2)/width)**2)
        
    elif wf_type == 'Gaussian Derivative':
        sig = -(t-max(t)/2)*np.exp(-1/2*((t-max(t)/2)/width)**2)
        sig = amp*sig/max(sig)
        
    else:
        sig = np.zeros(samps)
    return sig

# function that generates waveforms depending on input selected
def _sig_gen(i_wf, q_wf, samples, i_amp, q_amp, i_width, q_width):
    
    width = 1 # scaling in x. For sine/cos is equivalent to frequency
    
    t = np.linspace(0,samples,samples)
    
    i_sig = _wf_eq(i_wf,i_amp,i_width,t)
    q_sig = _wf_eq(q_wf,q_amp,q_width,t)

    return t, i_sig, q_sig

# Waveform plotting function
def _plot_wf(i_wf, q_wf, samples, i_amp, q_amp, i_width, q_width):
    
    t, i_sig, q_sig = _sig_gen(i_wf, q_wf, samples, i_amp, q_amp, i_width, q_width)
    
    _, axs = plt.subplots(2, sharex=True, figsize=(12,5), tight_layout=True)
    axs[1].set_xlabel('time t/dt')
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
    '''
    TO DO: This function is NOT general enough! There are still cases where qamp is too larg
    Need to find the qmax for which the norm is < 0 throughout the whole samples span.
    Probably need to find MAX of ABS of I and Q, look for indices at those locations (instead of just Q),
    and then find q_max based on that.
    For this, it is then important to pass the correct q_sign signal (not just a scaled ver of it)
    '''
    
    q_max_val = max(abs(q_sign)) # Find maximum of the abs of quad signal (taking abs to account for minimum too)
    q_max_ind = [i for i, x in enumerate(abs(q_sign)) if x == q_max_val] # indices of maxima in abs quad signal
    
    iamp_max = max(abs(i_sign[q_max_ind]))   # evaluate in-phase sig at indices of quad max,
                                            # and find largest value
    
    qamp_max = np.sqrt(1-iamp_max**2)       # qamp max needed so norm of I and Q is not larger than 1
    
    return qamp_max


global my_pulse

class PulseDesigner(widgets.VBox):

    def setup_config(self):

        # Parameters
        self.backend_lst = ['Armonk', 'Almaden']                              # Backends that support Pulse
        self.input_lst = ['Custom Waveform', 'Import from Native', 'Import from Array']                                  # Waveform input options
        self.waveform_lst = ['Rect', 'Sine', 'Cosine', 'Gaussian', 
                        'Gaussian Derivative','Gaussian Square','DRAG']  # Custom waveforms available

        '''TO DO: Find out what samples and amplitude resolution is for each backend'''
        self.amp_res_val = 0.01 # Amplitude resolution
        self.samples_val = 640  # Number of initial samples

        self.i_sig_arr = np.zeros(self.samples_val) # Array with in-phase signal amplitudes
        self.q_sig_arr = np.zeros(self.samples_val) # Array with quadrature signal amplitudes

        ### Fixed parameters ###

        # fixed widget elements to manipulate in-phase and quadrature amplitude arrays
        self.i_sig_fxd = widgets.fixed(value=self.i_sig_arr)
        self.q_sig_fxd = widgets.fixed(value=self.q_sig_arr)

        # Maximum amplitude for quadrature signal. Needs to be interactive bc depends on in-phase amplitude
        self.q_amp_max_fxd = widgets.fixed(value=0.0)

    def setup_custom_waveform_input(self):
        self.i_wf_dd = widgets.Dropdown(
            options=self.waveform_lst, 
            layout=widgets.Layout(width='auto'),
            description='In-Phase:',
        )

        # Dropdown menu for type of quadrature wavefunction (enabled when custom waveform is selected)
        self.q_wf_dd = widgets.Dropdown(
            options=self.waveform_lst,
            layout=widgets.Layout(width='auto'),
            description='Quadrature:',
        )

        # Slider for in-phase amplitude (from -1 to 1)
        self.samples_sldr = widgets.IntSlider(
            value=self.samples_val, 
            description='Samples:', 
            min=10, max=2*self.samples_val, step=2,
            continuous_update=False
            
        )

        self.samples_hbox = widgets.HBox([self.samples_sldr], layout=widgets.Layout(display='flex', flex_flow='column', align_items='center'))

        # Slider for In-phase frequency,width,sigma (Gets updated depending on type of signal)
        self.i_width_sldr = widgets.IntSlider(
            value=np.floor(self.samples_val/2), 
            description='I Width', 
            min=1, max=100, step=1,
            continuous_update=False
        )

        # Slider for quadrature frequency,width,sigma (Gets updated depending on type of signal)
        self.q_width_sldr = widgets.IntSlider(
            value=np.floor(self.samples_val/2), 
            description='Q Width', 
            min=1, max=100, step=1,
            continuous_update=False
        )

        width_hbox = widgets.HBox([self.i_width_sldr, self.q_width_sldr])

        # Slider for in-phase amplitude (from -1 to 1)
        self.i_amp_sldr = widgets.FloatSlider(
            value=1, 
            description='I Amplitude', 
            min=-1, max=1, step=self.amp_res_val,
            continuous_update=False,
        )

        # Slider for quadrature amplitude (Starts from -1 to 1, but gets updated based on value of in-phase amp)
        # This is because Pulse only accepts steps with max amplitude of R(I,Q) = 1
        self.q_amp_sldr = widgets.FloatSlider(
            value=0, 
            description='Q Amplitude', 
            min=-1, max=1, step=self.amp_res_val,
            continuous_update=False,
        )

        amp_hbox = widgets.HBox([self.i_amp_sldr, self.q_amp_sldr])

        self.custom_wf_hbox = widgets.HBox([
            widgets.VBox([
                widgets.Label("Waveform: "), self.i_wf_dd, self.q_wf_dd
            ], layout=widgets.Layout(width='30%')),
            widgets.VBox([
                self.samples_hbox,
                width_hbox,
                amp_hbox
            ], layout=widgets.Layout(width='70%'))
        ])

        self.custom_wf_controls = [self.i_wf_dd, self.q_wf_dd, self.samples_sldr, self.i_width_sldr, self.q_width_sldr, self.i_amp_sldr, self.q_amp_sldr]

    

    def setup_event_handlers(self):

        # Change number of min/max samples based on backend selected
        '''TO DO: Right now, min/max/value samples below were manually plugged in, 
        but need to get them from actual backends.'''
        # Updates max allowable value of Quadrature amplitude
        def update_qamp_max(*args):

            # Here to calculate q_sig I am passing an arbitrary amplitude (10*amp_res_val) because q_sig is only needed
            # to find the indices of the maxima of the currently-selected function. If I pass the current
            # q_amp (q_amp_sldr.value), there is a change it will be zero and no correct maxima are found
            t, i_sig, q_sig = _sig_gen(self.i_wf_dd.value, self.q_wf_dd.value, 
                                    self.samples_sldr.value, self.i_amp_sldr.value, 10*self.amp_res_val,
                                    self.i_width_sldr.value, self.q_width_sldr.value)

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

        def update_i_width_sldr(*args):
            if self.i_wf_dd.value == 'Sine' or self.i_wf_dd.value == 'Cosine':
                self.i_width_sldr.description='I Periods'
                self.i_width_sldr.value = 1
                self.i_width_sldr.min = 1
                self.i_width_sldr.max = 10
            elif self.i_wf_dd.value == 'Rect':
                self.i_width_sldr.description='I Width'
                self.i_width_sldr.value = np.floor(self.samples_sldr.value/2)
                self.i_width_sldr.min = 1
                self.i_width_sldr.max = self.samples_sldr.value
            else:
                self.i_width_sldr.description='I Sigma'
                self.i_width_sldr.value = np.floor(self.samples_sldr.value/10)
                self.i_width_sldr.min = 1
                self.i_width_sldr.max = self.samples_sldr.value
                
        self.i_wf_dd.observe(update_i_width_sldr, 'value')
        self.samples_sldr.observe(update_i_width_sldr, 'value')

        # Updates quadrature width/sigma/frequency slider
        def update_q_width_sldr(*args):
            if self.q_wf_dd.value == 'Sine' or self.q_wf_dd.value == 'Cosine':
                self.q_width_sldr.description='Q Periods'
                self.q_width_sldr.value = 1
                self.q_width_sldr.min = 1
                self.q_width_sldr.max = 10
            elif self.q_wf_dd.value == 'Rect':
                self.q_width_sldr.description='Q Width'
                self.q_width_sldr.value = np.floor(self.samples_sldr.value/2)
                self.q_width_sldr.min = 1
                self.q_width_sldr.max = self.samples_sldr.value
            else:
                self.q_width_sldr.description='Q Sigma'
                self.q_width_sldr.value = np.floor(self.samples_sldr.value/10)
                self.q_width_sldr.min = 1
                self.q_width_sldr.max = self.samples_sldr.value


        self.q_wf_dd.observe(update_q_width_sldr, 'value')
        self.samples_sldr.observe(update_q_width_sldr, 'value')

    def plot_figure(self):
        self.fig_out = widgets.interactive_output(
            _plot_wf, 
            {'i_wf':self.i_wf_dd, 
            'q_wf':self.q_wf_dd,
            'samples':self.samples_sldr,
            'i_amp':self.i_amp_sldr,
            'q_amp':self.q_amp_sldr,
            'i_width':self.i_width_sldr,
            'q_width':self.q_width_sldr
            }
        )

    def __init__(self):

        self.setup_config()
        self.setup_custom_waveform_input()
        self.setup_event_handlers()
        self.plot_figure()

        # Button to save pulse to array
        self.save_btn = widgets.Button(
            description='Save to my_pulse',
            icon='check',
            button_style='info',
            disabled=False,
            layout=widgets.Layout(width='150px')
        )
        def on_button_clicked(b):
            t, i_sig, q_sig = _sig_gen(self.i_wf_dd.value, self.q_wf_dd.value, 
                                    self.samples_sldr.value, self.i_amp_sldr.value, self.q_amp_sldr.value,
                                    self.i_width_sldr.value, self.q_width_sldr.value)
            
            my_pulse = i_sig + 1j*q_sig

        self.save_btn.on_click(on_button_clicked)

        super().__init__([
            self.custom_wf_hbox,
            self.fig_out,
            self.save_btn,
        ])

