import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'


# Fuction that evaluates waveform equation depending on waveform type selection
def wf_eq(wf_type, amp, width, t):
    
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
def sig_gen(wf_in, i_wf, q_wf, samples, i_amp, q_amp, i_width, q_width):
    
    width = 1 # scaling in x. For sine/cos is equivalent to frequency
    
    t = np.linspace(0,samples,samples)
    
    if wf_in == 'Custom Waveform':
        i_sig = wf_eq(i_wf,i_amp,i_width,t)
        q_sig = wf_eq(q_wf,q_amp,q_width,t)
    else:
        i_sig = np.zeros(t.size)
        q_sig = np.zeros(t.size)

    return t, i_sig, q_sig

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

# Waveform plotting function
def plot_wf(wf_in, i_wf, q_wf, samples, i_amp, q_amp, i_width, q_width):
    
    t, i_sig, q_sig = sig_gen(wf_in, i_wf, q_wf, samples, i_amp, q_amp, i_width, q_width)
    
    fig, axs = plt.subplots(2, sharex=True)
    axs[1].set_xlabel('time t/dt')
    axs[0].set_ylabel('In-Phase Amplitude')
    axs[1].set_ylabel('Quadrature Amplitude')
    axs[0].set_xlim(0,samples)
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

# Parameters
backend_lst = ['Armonk', 'Almaden']                              # Backends that support Pulse
input_lst = ['Import from Native', 'Import from Array', 
             'Custom Waveform']                                  # Waveform input options
waveform_lst = ['Rect', 'Sine', 'Cosine', 'Gaussian', 
                'Gaussian Derivative','Gaussian Square','DRAG']  # Custom waveforms available

'''TO DO: Find out what samples and amplitude resolution is for each backend'''
amp_res_val = 0.01 # Amplitude resolution
samples_val = 640  # Number of initial samples

i_sig_arr = np.zeros(samples_val)                  # Array with in-phase signal amplitudes
q_sig_arr = np.zeros(samples_val)                  # Array with quadrature signal amplitudes


### UI Left Panel (I/O) ###

# Dropdown menu for backend
backend_dd = widgets.Dropdown(options=backend_lst, 
                              layout=widgets.Layout(width='auto'),
                              continuous_update=False,
                              disabled=False)

# Dropdown menu for waveform input
wf_in_dd = widgets.Dropdown(options=input_lst, 
                            layout=widgets.Layout(width='auto'),
                            continuous_update=False,
                            disabled=False)

# Dropdown menu for type of in-phase wavefunction (enabled when custom waveform is selected)
i_wf_dd = widgets.Dropdown(options=waveform_lst, 
                          layout=widgets.Layout(width='auto'),
                          description='In-Phase:',
                          continuous_update=False,
                          disabled=True)

# Dropdown menu for type of quadrature wavefunction (enabled when custom waveform is selected)
q_wf_dd = widgets.Dropdown(options=waveform_lst,
                          layout=widgets.Layout(width='auto'),
                          description='Quadrature:',
                          continuous_update=False,
                          disabled=True)

# Button to save pulse to array
save_btn = widgets.Button(description='Save',
                          icon='check',
                          button_style='', # 'success', 'info', 'warning', 'danger' or ''
                          disabled=False)


# Combines all dropdown menus in a left panel
left_panel = widgets.VBox([widgets.Label("Backend:"), backend_dd,
                           widgets.Label("Input:"), wf_in_dd,
                           widgets.Label("Waveform:"), i_wf_dd, q_wf_dd,
                           widgets.Label("Output:"), 
                           widgets.HBox([widgets.Label("Save to \'my_pulse\" "),save_btn])])


### UI Top Pannel Waveform Control (enabled when custom waveform is selected) ###

# Slider to select number of samples
'''TO DO: Need to check what the valid resolution is for pulse amplitudes for each backend, 
dont think there is a restriction though.'''
samples_sldr = widgets.IntSlider(value=samples_val, 
                                 description='Samples:', 
                                 min=10, max=2*samples_val, step=2,
                                 continuous_update=False,
                                 disabled=True)

# Slider for In-phase frequency,width,sigma (Get's updated depending on type of signal)
i_width_sldr = widgets.IntSlider(value=np.floor(samples_val/2), 
                                 description='I Width', 
                                 min=1, max=100, step=1,
                                 continuous_update=False,
                                 disabled=True)

# Slider for quadrature frequency,width,sigma (Get's updated depending on type of signal)
q_width_sldr = widgets.IntSlider(value=np.floor(samples_val/2), 
                                 description='Q Width', 
                                 min=1, max=100, step=1,
                                 continuous_update=False,
                                 disabled=True)


'''TO DO: maybe create dictionary containing different width options for ease of selection
Need to decide if I need a "delay/centering" option.'''

# Slider for in-phase amplitude (from -1 to 1)
i_amp_sldr = widgets.FloatSlider(value=1, 
                                 description='I Amplitude', 
                                 min=-1, max=1, step=amp_res_val,
                                 continuous_update=False,
                                 disabled=True)

# Slider for quadrature amplitude (Starts from -1 to 1, but gets updated based on value of in-phase amp)
# This is because Pulse only accepts steps with max amplitude of R(I,Q) = 1
q_amp_sldr = widgets.FloatSlider(value=0, 
                                 description='Q Amplitude', 
                                 min=-1, max=1, step=amp_res_val,
                                 continuous_update=False,
                                 disabled=True)


# Combines time-related sliders in a box
time_panel = widgets.HBox([i_width_sldr,q_width_sldr])

# Combines amplitude sliders in a box
amp_panel = widgets.HBox([i_amp_sldr,q_amp_sldr])



# Top panel with all waveform control sliders
top_panel = widgets.VBox([samples_sldr,time_panel,amp_panel])

### Fixed parameters ###

# fixed widget elements to manipulate in-phase and quadrature amplitude arrays
i_sig_fxd = widgets.fixed(value=i_sig_arr)
q_sig_fxd = widgets.fixed(value=q_sig_arr)

# Maximum amplitude for quadrature signal. Needs to be interactive bc depends on in-phase amplitude
q_amp_max_fxd = widgets.fixed(value=0.0)

### UI interactions ###

# Enable waveform editing when 'Custom Waveform' is selected as input
def update_wf_dd_disabled(*args):
    if wf_in_dd.value == 'Custom Waveform':
        i_wf_dd.disabled = False
        q_wf_dd.disabled = False
        samples_sldr.disabled = False
        i_amp_sldr.disabled = False
        q_amp_sldr.disabled = False
        i_width_sldr.disabled = False
        q_width_sldr.disabled = False
    else:
        i_wf_dd.disabled = True
        q_wf_dd.disabled = True
        samples_sldr.disabled = True
        i_amp_sldr.disabled = True
        q_amp_sldr.disabled = True
        i_width_sldr.disabled = True
        q_width_sldr.disabled = True
        
wf_in_dd.observe(update_wf_dd_disabled, 'value')

# Change number of min/max samples based on backend selected
'''TO DO: Right now, min/max/value samples below were manually plugged in, 
but need to get them from actual backends.'''
def update_samples(*args):
    if backend_dd.value == 'Almaden':
        samples_sldr.min = 10
        samples_sldr.value = 160
        samples_sldr.max = 320
    if backend_dd.value == 'Armonk':
        samples_sldr.min = 10
        samples_sldr.value = 640
        samples_sldr.max = 1280        

backend_dd.observe(update_samples, 'value')


# Updates max allowable value of Quadrature amplitude
def update_qamp_max(*args):
    if wf_in_dd.value == 'Custom Waveform':

        # Here to calculate q_sig I am passing an arbitrary amplitude (10*amp_res_val) because q_sig is only needed
        # to find the indices of the maxima of the currently-selected function. If I pass the current
        # q_amp (q_amp_sldr.value), there is a change it will be zero and no correct maxima are found
        t, i_sig, q_sig = sig_gen(wf_in_dd.value, i_wf_dd.value, q_wf_dd.value, 
                                  samples_sldr.value, i_amp_sldr.value, 10*amp_res_val,
                                  i_width_sldr.value, q_width_sldr.value)

        q_amp_max = find_qamp_max(i_sig,q_sig)

        if q_amp_max < amp_res_val:
            q_amp_sldr.value = 0
            q_amp_sldr.disabled = True
        else:
            q_amp_sldr.max = q_amp_max
            q_amp_sldr.min = -q_amp_max
            q_amp_sldr.disabled = False


wf_in_dd.observe(update_qamp_max, 'value')
i_wf_dd.observe(update_qamp_max, 'value')
q_wf_dd.observe(update_qamp_max, 'value')
samples_sldr.observe(update_qamp_max, 'value')
i_amp_sldr.observe(update_qamp_max, 'value')
q_amp_sldr.observe(update_qamp_max, 'value')


# Updates in-phase width/sigma/frequency slider
def update_i_width_sldr(*args):
    if i_wf_dd.value == 'Sine' or i_wf_dd.value == 'Cosine':
        i_width_sldr.description='I Periods'
        i_width_sldr.value = 1
        i_width_sldr.min = 1
        i_width_sldr.max = 10
    elif i_wf_dd.value == 'Rect':
        i_width_sldr.description='I Width'
        i_width_sldr.value = np.floor(samples_sldr.value/2)
        i_width_sldr.min = 1
        i_width_sldr.max = samples_sldr.value
    else:
        i_width_sldr.description='I Sigma'
        i_width_sldr.value = np.floor(samples_sldr.value/10)
        i_width_sldr.min = 1
        i_width_sldr.max = samples_sldr.value
        
i_wf_dd.observe(update_i_width_sldr, 'value')
samples_sldr.observe(update_i_width_sldr, 'value')

# Updates quadrature width/sigma/frequency slider
def update_q_width_sldr(*args):
    if q_wf_dd.value == 'Sine' or q_wf_dd.value == 'Cosine':
        q_width_sldr.description='Q Periods'
        q_width_sldr.value = 1
        q_width_sldr.min = 1
        q_width_sldr.max = 10
    elif q_wf_dd.value == 'Rect':
        q_width_sldr.description='Q Width'
        q_width_sldr.value = np.floor(samples_sldr.value/2)
        q_width_sldr.min = 1
        q_width_sldr.max = samples_sldr.value
    else:
        q_width_sldr.description='Q Sigma'
        q_width_sldr.value = np.floor(samples_sldr.value/10)
        q_width_sldr.min = 1
        q_width_sldr.max = samples_sldr.value


q_wf_dd.observe(update_q_width_sldr, 'value')
samples_sldr.observe(update_q_width_sldr, 'value')


def on_button_clicked(b):
    if wf_in_dd.value == 'Custom Waveform':
        t, i_sig, q_sig = sig_gen(wf_in_dd.value, i_wf_dd.value, q_wf_dd.value, 
                                  samples_sldr.value, i_amp_sldr.value, q_amp_sldr.value,
                                  i_width_sldr.value, q_width_sldr.value)
        
        b.my_pulse = i_sig + 1j*q_sig

save_btn.on_click(on_button_clicked)

# Interactive figure
fig_out = widgets.interactive_output(plot_wf, 
                                     {'wf_in':wf_in_dd, 
                                      'i_wf':i_wf_dd, 
                                      'q_wf':q_wf_dd,
                                      'samples':samples_sldr,
                                      'i_amp':i_amp_sldr,
                                      'q_amp':q_amp_sldr,
                                      'i_width':i_width_sldr,
                                      'q_width':q_width_sldr})
fig_out.layout.height = '350px'

# Combines top sliders and figure in right panel
right_panel = widgets.VBox([top_panel,fig_out])



class PulseEditor(widgets.VBox):

    def __init__(self):
        super().__init__([widgets.HBox([left_panel, right_panel])])

