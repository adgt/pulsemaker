import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

class ScheduleEditor:
    def __init__(self):

        ### Initialize ###
        ''' 
        I thoughtlists were a better option for our data structuring bc they can be easily manipulated. 
        After some reasearch, dictonaries seem to be a more natural choice bc the can be easily indexed by channel and
        merge/replace can be easily done with the in-place operator |=

        self._pulses = [[str,np.array([])],]        # List of pulses for internal use. 
                                                    # Format: [['chan_a',waveform0],['chan_b',waveform1],...] waveformx is a numpy array with pulse data
        self._phases = [[str,np.array([[],[]])],]   # List of phase-shift values for internal use. 
                                                    # Format: [['chan_a',phaseshift0],['chan_b',phaseshift1],...] phaseshiftx is a numpy array with times,phases
        self._freqs = [[str,np.array([[],[]])],]    # List of frequency values for internal use. 
                                                    # Format: [['chan_a',freqval0],['chan_b',freqval1],...] freqvalx is a numpy array with times,frequencies
        '''

        self._pulses = {}   # List of pulses for internal use. 
                            # Format: {d0:waveform00, u0:waveform01, d1:waveform11 ...} waveformx is a numpy array with pulse data
        self._phases = {}   # List of phase-shift values for internal use. 
                            # Format: {d0:phaseshift00, u0:phaseshift01, d1:phaseshift11 ...} phaseshiftx is a list with [time,phase] elems
        self._freqs = {}    # List of frequency values for internal use. 
                            # Format: {d0:freqval00, u0:freqval01, d1:freqval11 ...} freqvalx is a list with [time,phase] elems

        self._current_qubit = 'q0' 
        self._current_chann = ['d0','d0','d0']  # list of current channel selections for [phase,freq,pulse]
        self._current_sample = 0
        self._current_phase = 0
        self._current_freq = 0

        self.schedule = tuple()  # Final schedule (currently uses Qiskit's data structuring) 

        backend_input_lst = ['Armonk', 'Almaden', 'Casablanca']
        backend_qnum_lst = [1, 20, 20]
        backend_qubit_lst = ['q0']      # NOTE: must be updated as a function of selected backend
        backend_chan_lst = ['d0','u0','d1']       # NOTE: must be updated as a function of selected backend (includes d and u)
        backend_cmap_lst = ['q0 <-> q1',
                            'q1 <-> q2']  # NOTE: This is a dummy coupling map. Need to import actual map from backend
                                          # This will be used for 2 qubit gates, therefore this list will replace the
                                          # backend_qubit_lst in the dd widget when 2-qubit gates are selected (e.g. cx)

        schedule_input_lst = ['From Native Gate', 'Custom Schedule', 'From Variable']
        nativegate_input_lst = ['X','Y','Z','H','CX','CZ']
        pulse_input_lst = []

        # Function to Update schedule plot
        '''
        def plot_schedule():

            if not len(self._pulses):
                i_sig = np.array([0])
                q_sig = np.array([0])
            else:
                i_sig = np.real(my_schedule)
                q_sig = np.imag(my_schedule)
                
            samps = i_sig.size
            t = np.linspace(0,samps,samps)
            
            fig, axs = plt.subplots(figsize=(7.5,3))
            axs.set_xlabel('time t/dt')
            axs.set_ylabel('Amplitude')
            axs.set_xlim(0,samps)
            axs.set_ylim(-1.1,1.1)
            axs.step(t, i_sig, 'r')
            axs.fill_between(t, i_sig, color='r', alpha=0.2, step='pre')
            axs.step(t, q_sig, 'b')
            axs.fill_between(t, q_sig, color='b', alpha=0.2, step='pre')
        '''

        # Waveform plotting function (NOTE: TO BE DELETED)
        def plot_wf(nativegate):
            samples = 16
            t = np.linspace(0,samples,samples)
            
            if nativegate == 'X':
                i_sig = np.sin(2*np.pi*t)
                q_sig = 0.001*np.cos(2*np.pi*t)
            elif nativegate == 'Z':
                i_sig = np.cos(2*np.pi*t)
                q_sig = 0.001*np.sin(2*np.pi*t)
            else:
                i_sig = np.ones(t.size)
                q_sig = np.ones(t.size)

            i_amp = np.max(i_sig)
            q_amp = np.max(q_sig)
            fig, (ax0,ax1) = plt.subplots(1,2, figsize=(8, 4), sharex=True)
            #fig, (ax0,ax1) = plt.subplots(1,2, sharex=True)
            ax0.set_xlabel('time t/dt')
            ax1.set_xlabel('time t/dt')
            ax0.set_ylabel('In-Phase Amplitude')
            ax1.set_ylabel('Quadrature Amplitude')
            ax0.set_xlim(0,samples)
            ax1.set_xlim(0,samples)
            ax0.set_ylim(-(abs(i_amp)+0.1),abs(i_amp)+0.1)
            ax1.set_ylim(-(abs(q_amp)+0.1),abs(q_amp)+0.1)
            ax0.step(t, i_sig, 'r')
            ax1.step(t, q_sig, 'b')
            fig.tight_layout(pad=3.0)

        ### User Interface Components ###

        # Dropdown menu for backend
        backend_input_dd = widgets.Dropdown(options=backend_input_lst, 
                                            layout=widgets.Layout(width='auto'),
                                            continuous_update=False,
                                            disabled=False)

        # Dropdown menu for waveform input
        schedule_input_dd = widgets.Dropdown(options=schedule_input_lst, 
                                             layout=widgets.Layout(width='auto'),
                                             continuous_update=False,
                                             disabled=False)

        # Dropdown menu for native gate selection
        nativegate_input_dd = widgets.Dropdown(options=nativegate_input_lst, 
                                               layout=widgets.Layout(width='160px'),
                                               description='From Gate:',
                                               continuous_update=False,
                                               disabled=False)

        # Dropdown menu for qubit selection to native gate schedule to main schedule
        nativegate_qubit_dd = widgets.Dropdown(options=backend_qubit_lst, 
                                               layout=widgets.Layout(width='160px'),
                                               description='Append to:',
                                               continuous_update=False,
                                               disabled=False)

        # Button to append native gate schedule to main schedule
        nativegate_append_btn = widgets.Button(description='Append',
                                               icon='',
                                               button_style='', # 'success', 'info', 'warning', 'danger' or ''
                                               layout=widgets.Layout(width='80px'),
                                               disabled=False)
        nativegate_append_btn.name = 'nativegate_btn'

        # Box for nativegate-related widgets:
        nativegate_pannel = widgets.VBox([nativegate_input_dd,
                                          widgets.HBox([nativegate_qubit_dd,nativegate_append_btn])])

        # Floating Textbox for Phase Shift input
        '''TODO: Need to be able to pass multiples of np.pi or pi'''
        shiftphase_input_fltxt = widgets.BoundedFloatText(value=0.0, min=0.0, max=2*np.pi, step=0.001,
                                                           layout=widgets.Layout(width='245px'),
                                                           description='Phase [1/(2π)]:',
                                                           disabled=False)

        # Dropdown menu for channel selection to append Phase Shift schedule
        shiftphase_chan_dd = widgets.Dropdown(options=backend_chan_lst, 
                                              layout=widgets.Layout(width='160px'),
                                              description='Append to:',
                                              continuous_update=False,
                                              disabled=False)

        # Button to append Phase Shift to schedule
        shiftphase_append_btn = widgets.Button(description='Append',
                                               icon='',
                                               button_style='',
                                               layout=widgets.Layout(width='80px'),
                                               disabled=False)
        shiftphase_append_btn.name = 'shiftphase_btn'

        # Box for PhaseShift-related widgets:
        shiftphase_pannel = widgets.VBox([shiftphase_input_fltxt,
                                          widgets.HBox([shiftphase_chan_dd,shiftphase_append_btn])])

        # Floating Textbox for Frequency value input
        shiftfreq_input_fltxt = widgets.BoundedFloatText(value=0.0, min=0.0, max=2*np.pi, step=0.001,
                                                          layout=widgets.Layout(width='245px'),
                                                          description='Freq [Hz]:',
                                                          disabled=False)

         # Dropdown menu for channel selection to append Frequency value schedule
        shiftfreq_chan_dd = widgets.Dropdown(options=backend_chan_lst, 
                                             layout=widgets.Layout(width='160px'),
                                             description='Append to:',
                                             continuous_update=False,
                                             disabled=False)

        # Button to append Frequency value to schedule
        shiftfreq_append_btn = widgets.Button(description='Append',
                                              icon='',
                                              button_style='',
                                              layout=widgets.Layout(width='80px'),
                                              disabled=False)
        shiftfreq_append_btn.name = 'shiftfreq_btn'

        # Box for FreqValue-related widgets:
        shiftfreq_pannel = widgets.VBox([shiftfreq_input_fltxt,
                                          widgets.HBox([shiftfreq_chan_dd,shiftfreq_append_btn])])

        # Dropdown menu for pulse array selection
        pulse_input_dd = widgets.Dropdown(options=pulse_input_lst, 
                                          layout=widgets.Layout(width='245px'),
                                          description='Pulse:',
                                          continuous_update=False,
                                          disabled=False)

         # Dropdown menu for channel selection to append pulse to schedule
        pulse_chan_dd = widgets.Dropdown(options=backend_chan_lst, 
                                         layout=widgets.Layout(width='160px'),
                                         description='Append to:',
                                         continuous_update=False,
                                         disabled=False)

        # Button to append pulse to schedule
        pulse_append_btn = widgets.Button(description='Append',
                                          icon='',
                                          button_style='',
                                          layout=widgets.Layout(width='80px'),
                                          disabled=False)
        pulse_append_btn.name = 'pulse_btn'

        # Box for pulse-related widgets:
        pulse_pannel = widgets.VBox([pulse_input_dd,
                                          widgets.HBox([pulse_chan_dd,pulse_append_btn])])

        # Combines all dropdown menus in a left panel
        left_panel = widgets.VBox([widgets.Label("Backend:"), backend_input_dd,
                                   widgets.Label("Input:"), schedule_input_dd,
                                   widgets.Label("Schedule:"),
                                   nativegate_pannel, shiftphase_pannel, shiftfreq_pannel,pulse_pannel])

        
        ### Widget Interactions ###

        # Update Schedule when append buttons are pressed
        def update_schedule(b):
            if b.name == 'nativegate_btn':
                print('Phases:',self._phases)
                print('Frequencies:',self._freqs)
                print('Pulses:',self._pulses)

            elif b.name == 'shiftphase_btn':
                phase = [self._current_sample, self._current_phase]

                if self._current_chann[0] in self._phases.keys():
                    # Check if channel is already present in _phases to append/replace new data
                    # Else, add channel to _phases.
                    phase_array = self._phases[self._current_chann[0]]

                    if phase_array[-1][0] == self._current_sample:
                        # If sample number hasn't changed, replace PhaseShift value
                        # Else, append new [time,PhaseShift] item to _phases
                        phase_array[-1] = phase
                    else:
                        phase_array += [phase]
                else:
                    phase_array = [phase]

                self._phases[self._current_chann[0]] = phase_array

            elif b.name == 'shiftfreq_btn':
                freq = [self._current_sample, self._current_freq]

                if self._current_chann[1] in self._freqs.keys():
                    # Check if channel is already present in _freqs to append/replace new data
                    # Else, add channel to _freqs.
                    freq_array = self._freqs[self._current_chann[1]]

                    if freq_array[-1][0] == self._current_sample:
                        # If sample number hasn't changed, replace Frequency value
                        # Else, append new [time,FreqValue] item to _freqs
                        freq_array[-1] = freq
                    else:
                        freq_array += [freq]
                else:
                    freq_array = [freq]

                self._freqs[self._current_chann[1]] = freq_array

        append_btns = [nativegate_append_btn,shiftphase_append_btn,shiftfreq_append_btn,pulse_append_btn]
        for btn in append_btns:
            btn.on_click(update_schedule)
        
        # Update current Channels based on dropdown menus

        def update_channels(*args):
            self._current_chann = [shiftphase_chan_dd.value, shiftfreq_chan_dd.value, pulse_chan_dd.value]

        chan_dds = [shiftphase_chan_dd, shiftfreq_chan_dd, pulse_chan_dd]
        for chan in chan_dds:
            chan.observe(update_channels, 'value')


        # TEST FIG OUT (NOTE: TO BE DELETED)
        wf_fig_out = widgets.interactive_output(plot_wf, {'nativegate':nativegate_input_dd})


        schedule_editor = widgets.HBox([left_panel, wf_fig_out])

        self._editor = schedule_editor
    
    def _ipython_display_(self):
        display(self._editor)