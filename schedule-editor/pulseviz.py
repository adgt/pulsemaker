import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display


# Function to Update schedule plot
def plot_sch(phases,freqs,pulses,samples):
    '''
    TODO: Need to generalize this function to strip each channel and create a separate subplot in the following order:
          d0,u0,d1,u1,...dn,un. Channels should share the x-axis, which means that if one channel has less samples than
          another, then padding (0 values) need to be added so that all vectors match the length of the largest array
    '''

    # plot pulses
    '''NOTE: Only plotting d0 right now. Need to generalize for all channels'''
    
    if not 'd0' in pulses.keys():
        pulse_arr = np.array([0])
    else:
        pulse_arr = pulses.get('d0')

    i_sig = np.real(pulse_arr)
    q_sig = np.imag(pulse_arr)
        
    samps = i_sig.size
    t = np.linspace(0,samps,samps)
    
    fig, axs = plt.subplots(figsize=(8.5,3))
    axs.set_xlabel('time t/dt')
    axs.set_ylabel('Amplitude')
    axs.set_xlim(0,samps)
    axs.set_ylim(-1.1,1.1)
    axs.step(t, i_sig, 'r')
    axs.fill_between(t, i_sig, color='r', alpha=0.2, step='pre')
    axs.step(t, q_sig, 'b')
    axs.fill_between(t, q_sig, color='b', alpha=0.2, step='pre')

    if 'd0' in phases.keys():
        phases_lst = phases['d0']
        for time in phases_lst:
            axs.text(x=time[0], y=0, s=r'$\circlearrowleft$',
                     fontsize=14, color='purple',
                     ha='center', va='center')

    if 'd0' in freqs.keys():
        freqs_lst = freqs['d0']
        for time in freqs_lst:
            axs.text(x=time[0], y=0, s=r'$\downarrow$',
                     fontsize=14, color='forestgreen',
                     ha='center', va='bottom')

    '''
    ### NOTE: DELETE BELOW. JUST FOR DEBUGGING ###
    print('Phases:',phases)
    print('Frequencies:',freqs)
    print('Pulses:',pulses)
    print('Samples:',samples)
    ### ### ### ### ### ### ### ###
    '''

class ScheduleEditor:
    def __init__(self):

        ### Initialize ###
        ''' 
        I thought lists were a better option for our data structuring bc they can be easily manipulated. 
        After some reasearch, dictonaries seem to be a more natural choice bc the can be easily indexed by channel and
        merge/replace can be easily done.

        self.pulses = [[str,np.array([])],]        # List of pulses. 
                                                    # Format: [['chan_a',waveform0],['chan_b',waveform1],...] waveformx is a numpy array with pulse data
        self.phases = [[str,np.array([[],[]])],]   # List of phase-shift values. 
                                                    # Format: [['chan_a',phaseshift0],['chan_b',phaseshift1],...] phaseshiftx is a numpy array with times,phases
        self.freqs = [[str,np.array([[],[]])],]    # List of frequency values for 
                                                    # Format: [['chan_a',freqval0],['chan_b',freqval1],...] freqvalx is a numpy array with times,frequencies
        '''

        self.pulses = {}   # List of pulses. 
                           # Format: {d0:waveform00, u0:waveform01, d1:waveform11 ...} waveformx is a numpy array with pulse data
        self.phases = {}   # List of phase-shift values. 
                           # Format: {d0:phaseshift00, u0:phaseshift01, d1:phaseshift11 ...} phaseshiftx is a list with [time,phase] elems
        self.freqs = {}    # List of frequency values. 
                           # Format: {d0:freqval00, u0:freqval01, d1:freqval11 ...} freqvalx is a list with [time,phase] elems
        self.samples = 0   # Number of samples in schedule


        self._current_qubit = 'q0' 
        self._current_chann = ['d0','d0','d0']  # list of current channel selections for [phase,freq,pulse]
        self._current_phase = 0
        self._current_freq = 0
        self._current_pulse = np.array([])
        self.dummy_pulse  = np.array([])   # DELETE after figuring out how to pass pulses from pulse-editor

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
        shiftfreq_input_fltxt = widgets.BoundedFloatText(value=0.0, min=0.0, max=5.5, step=0.001,
                                                          layout=widgets.Layout(width='245px'),
                                                          description='Freq [GHz]:',
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
                pass

            elif b.name == 'shiftphase_btn':
                phase = [self.samples, self._current_phase]

                if self._current_chann[0] in self.phases.keys():
                    # Check if channel is already present in phases to append/replace new data
                    # Else, add channel to phases.
                    phase_array = self.phases[self._current_chann[0]]

                    if phase_array[-1][0] == self.samples:
                        # If sample number hasn't changed, replace PhaseShift value
                        # Else, append new [time,PhaseShift] item to phases
                        phase_array[-1] = phase
                    else:
                        phase_array += [phase]
                else:
                    phase_array = [phase]

                self.phases[self._current_chann[0]] = phase_array

            elif b.name == 'shiftfreq_btn':
                freq = [self.samples, self._current_freq]

                if self._current_chann[1] in self.freqs.keys():
                    # Check if channel is already present in freqs to append/replace new data
                    # Else, add channel to freqs.
                    freq_array = self.freqs[self._current_chann[1]]

                    if freq_array[-1][0] == self.samples:
                        # If sample number hasn't changed, replace Frequency value
                        # Else, append new [time,FreqValue] item to freqs
                        freq_array[-1] = freq
                    else:
                        freq_array += [freq]
                else:
                    freq_array = [freq]

                self.freqs[self._current_chann[1]] = freq_array

            elif b.name == 'pulse_btn':
                '''
                TODO: Need to add padding option (add dotted line of where schedule stands on each channel?)
                         Best way might be to add padding for visualization in the plot_sch function, but keep
                         arrays for each channel true to what the user is adding. The challenge is then keeping
                         track of the "current_sample" for each channel individually, but might be as simple as
                         always checking the length of the pulse array for each specific chan.
                '''


                pulse = self.dummy_pulse 

                if self._current_chann[2] in self.pulses.keys():
                    # Check if channel is already present in pulses to append new data
                    # Else, add channel to pulses.
                    pulse_array = np.append(self.pulses[self._current_chann[2]],pulse)
                else:
                    pulse_array = pulse

                self.pulses[self._current_chann[2]] = pulse_array

                self.samples = len(pulse_array)

            self.update()

        append_btns = [nativegate_append_btn,shiftphase_append_btn,shiftfreq_append_btn,pulse_append_btn]
        for btn in append_btns:
            btn.on_click(update_schedule)
        
        # Update current Channels based on values of dropdown menus
        def update_channels(*args):
            self._current_chann = [shiftphase_chan_dd.value, shiftfreq_chan_dd.value, pulse_chan_dd.value]

        chan_dds = [shiftphase_chan_dd, shiftfreq_chan_dd, pulse_chan_dd]
        for chan in chan_dds:
            chan.observe(update_channels, 'value')

        # Plot schedule when outputs change
        '''
        plot_sch_out = widgets.interactive_output(plot_sch,
                                                  {'phases':widgets.fixed(self.phases),
                                                   'freqs':widgets.fixed(self.freqs),
                                                   'pulses':widgets.fixed(self.pulses),
                                                   'samples':widgets.fixed(self.samples)})
        '''

        plot_sch_out = widgets.interactive(plot_sch,
                                           phases=widgets.fixed(self.phases),
                                           freqs=widgets.fixed(self.freqs),
                                           pulses=widgets.fixed(self.pulses),
                                           samples=widgets.fixed(self.samples))

        self._plot_sch = plot_sch_out

        # TEST FIG OUT (NOTE: TO BE DELETED)
        #wf_fig_out = widgets.interactive_output(plot_wf, {'nativegate':nativegate_input_dd})

        schedule_editor = widgets.HBox([left_panel, plot_sch_out])

        self._editor = schedule_editor
    
    def update(self):
        self._plot_sch.update()

    def _ipython_display_(self):
        display(self._editor)