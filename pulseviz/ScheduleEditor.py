import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import ipywidgets as widgets
from IPython.display import display

# Function to generate qubit, channel and coupling map lists based on selected backend
def qiskit_backend_config(backend_name, backend_input_lst, backend_qnum_lst):

    try:
        from qiskit.test.mock import FakeArmonk, FakeAlmaden

    except ImportError:
        pass

    '''TODO: might be able to do this in a better way without having to do conditional checks for each backend name.'''
    if backend_name == 'Armonk':
        backend = FakeArmonk()
    elif backend_name == 'Almaden':
        backend = FakeAlmaden()
    else:
        '''TODO: There is no FakeCasablanca, so using FakeAlmaden for it right now'''
        backend = FakeAlmaden()
    
    '''
    TODO: When selecting back a backend that has less number of qubits than what has already been added to the schedule,
            the extra channels need to be cleared out so that the user doesn't run into issues when trying to run the schedule in the
            desired backend. Qiskit will throw an error if schedule has instructions in channels not available in that backend.
    '''
    
    backend_indx = backend_input_lst.index(backend_name)
    num_qbs = backend_qnum_lst[backend_indx]

    # generate qubit_lst and chan_lst
    qubit_lst = []
    chan_lst = []
    for qb_num in range(num_qbs):
        qubit_lst.append('q'+str(qb_num))
        chan_lst.append('d'+str(qb_num))
        '''
        TODO: Right now, I am just associating one u channel per qubit. This is NOT correct. Each qubit-qubit interaction has 
                several control channels associated with it. Therefore, the right way to populate the available u channels is to
                get the coupling map for the backend, and extract the u channels associated with each element in that coupling map
                Currently trying to figure out how to do this. 
        '''
        if num_qbs > 1:
            chan_lst.append('u'+str(qb_num))

    # generate gate list
    gate_lst = ['X','Y','Z','H','ID','SX','RZ','CX']
    if num_qbs == 1:
        gate_lst.pop()

    # generate coupling map list
    cmap_lst = backend.configuration().coupling_map
    '''
    TODO: this conditional below should be removed once there is a FakeCasablanca. right now just "mocking" the coupling map by taking
          the cmap of the first 7 qubits of Almaden"
    '''
    if backend_name == 'Casablanca':
        cmap_aux = []
        for qb_pair in cmap_lst:
            if qb_pair[0] <= 6 and qb_pair[1] <= 6:
                cmap_aux.append(qb_pair)
        cmap_lst = cmap_aux

    return qubit_lst, chan_lst, cmap_lst, gate_lst


# Function to build qiskit schedule based on selected gate and backend
def qiskit_gate_to_sched(backend_name, gate_name, qubits, num_qbs):
    
    try:
        from qiskit import QuantumCircuit
        from qiskit import transpile, schedule as build_schedule
        from qiskit.test.mock import FakeArmonk, FakeAlmaden

    except ImportError:
        pass
    
    '''
    TODO: gates are currently applied only to q0. Need to get the selected qubit number as a parameter,
          and pass it the circuit below.
          DONE for single qubit gates
    '''
    qa = int(qubits[0][1:])
    qb = int(qubits[1][1:])

    circuit = QuantumCircuit(num_qbs)
    if gate_name == 'X':
        circuit.x(qa)
    if gate_name == 'Y':
        circuit.y(qa)
    if gate_name == 'Z':
        circuit.z(qa)
    if gate_name == 'H':
        circuit.h(qa)
    if gate_name == 'ID':
        circuit.id(qa)
    if gate_name == 'SX':
        circuit.sx(qa)
    if gate_name == 'RZ':
        '''
        TODO: currently, rx gate applies a pi/2 rotation, but need to make this general so 
              that the angle value can be passed by the user
        '''
        circuit.rz(np.pi/2,0)
   
    if gate_name == 'CX':
        '''TODO: Add CX gate. Currently commented out  bc only have support for single qubits, so qa always equals qb.'''
        circuit.cx(qa,qb)
    

    if backend_name == 'Armonk':
        backend = FakeArmonk()
    elif backend_name == 'Almaden':
        backend = FakeAlmaden()
    else:
        '''TODO: There is no FakeCasablanca, so using FakeAlmaden for it right now'''
        backend = FakeAlmaden()
    
    trans_circuit = transpile(circuit, backend)
    return build_schedule(trans_circuit, backend)


# Function to translate qiskit-schedule to scheduler-editor format
def qiskit_to_schedviz(qiskit_sch, current_samples):
    try:
        from qiskit.pulse import Play, SetFrequency, ShiftPhase
        from qiskit.pulse.channels import DriveChannel, ControlChannel

    except ImportError:
        pass
    
    phases = {}
    freqs = {}
    pulses = {}

    for start_time, instruction in qiskit_sch.instructions:

        if isinstance(instruction.channel, DriveChannel):
            chan = 'd'+str(instruction.channel.index)
        elif isinstance(instruction.channel, ControlChannel):
            chan = 'u'+str(instruction.channel.index)
        else:
            '''TODO: Might need to add support for other channels, like measure, acquire, etc'''
            pass
        
        if chan in current_samples:
            chan_sample = current_samples[chan]
        else: 
            chan_sample = 0

        if isinstance(instruction, Play):
            '''TODO: For Play instructions (pulses), 
            Need to check that the last sample of each pulse matches the first sample of the next.
            In general this could not be the case because eventually we will add "Delay" Instructions that could be 
            padding the pulse data in between consecutive Play instructions, so would need to fill in that missing data.
            From what I've seen, when building schedules from native gates, no Delay Instructions are used, but
            this might change in the future'''
            pulse = np.array(instruction.pulse.samples)

            if chan in pulses.keys():
                pulses[chan] = np.append(pulses[chan],pulse)
            else:
                pulses[chan] = pulse

        elif isinstance(instruction, ShiftPhase):
            phase = [chan_sample+start_time, instruction.phase]

            if chan in phases.keys():
                # Check if channel is already present in phases to append/replace new data
                # Else, add channel to phases.
                phase_array = phases[chan]
                phase_array += [phase]         
            else:
                phase_array = [phase]

            phases[chan] = phase_array

        '''TODO: isinstance(instruction, ShiftFrequency). 
                 Haven't done it yet bc haven't seen this instruction in any of the native gates.'''

    return phases, freqs, pulses

def plot_pulse_schedule(phases, freqs, pulses, samples):
    # Function to draw/update schedule plot
    def _plot(phases,freqs,pulses,samples):
        
        # Check channels in phase and frequency dicts but not on pulse
        # Create one-elem pulse array on those chans to avoid matplotlib error for missing data
        phase_chans = set(phases.keys())
        freq_chans = set(freqs.keys())
        pulses_srt = pulses.copy()

        for chan in (set.union(phase_chans,freq_chans)):
            if chan not in pulses_srt:
                pulses_srt[chan]=np.array([0])


        labels = ['a','d','m','u'] # labels for different channels:
                                   # a: acquire, d: drive, m: measure, u: x-correlation

        num_chans = max(len(pulses_srt), 1)
        gs = gridspec.GridSpec(num_chans, 1)
        ax = []

        ''' 
        To sort pulse dictionary, channel index values are calculated as follows:
        indx[0][0] stores the type of channel: a, d, m, u. By using the index value of the 'labels' list, 
        we know the position the channel holds within a given qubit. indx[0][1] stores the qubit value, so 
        by multiplying by the length of the 'labels' list, we know where the qubit sits wrt the others.
        by adding the two, we know where a given qubit channel should sit wrt to other channels.
        ''' 
        pulses_srt = sorted(pulses_srt.items(), 
                        key=lambda indx: (labels.index(indx[0][0])+int(indx[0][1])*len(labels)))


        fig = plt.subplots(figsize=(9,5))

        for chan_num, chan in enumerate(pulses_srt):

            # plot pulses
            if chan_num == 0:
                ax.append(plt.subplot(gs[chan_num]))
            else: 
                ax.append(plt.subplot(gs[chan_num], sharex=ax[0]))
            if chan_num < num_chans - 1:
                plt.setp(ax[chan_num].get_xticklabels(), visible=False)

            # NOTE: Axis settings. Still need to decide how they should look like
            ax[chan_num].text(0,0, chan[0], horizontalalignment='center',verticalalignment='center', fontweight='bold')
            ax[chan_num].tick_params(axis='y', which='major', labelsize=7)
            #ax[chan_num].tick_params(axis="y",direction="in", pad=-22)
            #ax[chan_num].get_yaxis().set_ticks([])
            #ax[chan_num].set_ylabel(chan+'  ', rotation=0, fontweight='bold')

            i_sig = np.real(chan[1])
            q_sig = np.imag(chan[1])
            samps = i_sig.size
            t = np.linspace(0,samps,samps)

            if chan[0][0] == 'd':
                ax[chan_num].step(t, i_sig, 'r')
                ax[chan_num].fill_between(t, i_sig, color='r', alpha=0.2, step='pre')
                ax[chan_num].step(t, q_sig, 'b')
                ax[chan_num].fill_between(t, q_sig, color='b', alpha=0.2, step='pre')
            else:
            #elif chan[0][0] == 'u': 
                '''
                TODO: Here I'm using an else statement to display anything that isn't a drive 
                    channel 'd' as if it was a control channel 'u'. If support for 'a' 'm' channels is added
                    need to make this into an 'elif chan[0][0] == 'u':' and also change colors of those channels too
                '''
                ax[chan_num].step(t, i_sig, 'y')
                ax[chan_num].fill_between(t, i_sig, color='y', alpha=0.2, step='pre')
                ax[chan_num].step(t, q_sig, 'orange')
                ax[chan_num].fill_between(t, q_sig, color='orange', alpha=0.2, step='pre')

            # plot phases
            if chan[0] in phases:
                phases_lst = phases[chan[0]]
                for time in phases_lst:           
                    ax[chan_num].text(x=time[0], y=0, s=r'$\circlearrowleft$',
                                      fontsize=14, color='purple',
                                      ha='center', va='center')


            # plot frequencies
            if chan[0] in freqs:
                freqs_lst = freqs[chan[0]]
                for time in freqs_lst:
                    ax[chan_num].text(x=time[0], y=0, s=r'$\downarrow$',
                            fontsize=14, color='forestgreen',
                            ha='center', va='bottom')

        plt.subplots_adjust(hspace=.0)

        '''
        ### NOTE: DELETE BELOW. JUST FOR DEBUGGING ###
        print('Phases:',phases)
        print('Frequencies:',freqs)
        print('Pulses:',pulses)
        print('Samples:',samples)
        ### ### ### ### ### ### ### ###
        '''

    return widgets.interactive(_plot,
                              phases=widgets.fixed(phases),
                              freqs=widgets.fixed(freqs),
                              pulses=widgets.fixed(pulses),
                              samples=widgets.fixed(samples))

class ScheduleEditor(widgets.VBox):
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

        self.pulses = {}   # Dictionary of pulses. 
                           # Format: {d0:waveform00, u0:waveform01, d1:waveform11 ...} waveformx is a numpy array with pulse data
        self.phases = {}   # Dictionary of phase-shift values. 
                           # Format: {d0:phaseshift00, u0:phaseshift01, d1:phaseshift11 ...} phaseshiftx is a list with [time,phase] elems
        self.freqs = {}    # Dictionary of frequency values. 
                           # Format: {d0:freqval00, u0:freqval01, d1:freqval11 ...} freqvalx is a list with [time,phase] elems
        self.samples = {}  # Dictionary of samples in schedule
                           # Format: {d0:sampsval00, u0:samps01, d1:samps11 ...} sampsvalx is the value of last sample on channel x

        
        self._current_qubits = ['q0','q0']      # qubit(s) currently selected. If a single-qubit gate is selected, then 
                                                # both elements in the list take the same value ['qx','qx'] (qx is currently selecte qubit)
                                                # If two-qubit gate (CX) is selected: ['qx','qy'] where CX is applied from qx to qy.
        self._current_chann = 'd0'              # list of current channel selections for phase,freq,pulse based on selected op type
        self._current_phase = 0
        self._current_freq = 0
        self._current_pulse = np.array([])
        self.dummy_pulse  = np.array([])        # DELETE after figuring out how to pass pulses from pulse-editor

        self.schedule = tuple()                 # Final schedule (currently will use Qiskit's data structuring) 

        backend_input_lst = ['Armonk', 'Almaden', 'Casablanca']
        backend_qnum_lst = [1, 20, 7]
        
        '''
        TODO: Since I already have a function that returns the params below based on selected backend (qiskit_backend_config),
              might just call it here rather than doing this explicit initialization
        '''
        self._backend_qubit_lst = ['q0']  # Updated as a function of selected backend
        self._backend_chan_lst = ['d0']   # Updated as a function of selected backend (includes d and u)
        self._backend_cmap_lst = None     # Updated as a function of selected backend
        self._backend_cmap_nms = None     # Maps back_cmap_lst elems to strings that can be displayed in the dropdown menu
                                          # e.g. [2,3] maps to ['q2 -> q3']
        nativegate_input_lst = ['X','Y','Z','H','ID','SX','RZ','CX']

        '''
        TODO: Might not need a schedule_input_lst dropdown menu. I had one in the Pulse editor to "gray out" certain 
              functionalities depending on what was selected. But for the schedule editor, there is no reason to do this. So I can
              remove both schedule_input_lst and schedule_input_dd
        '''    
        schedule_input_lst = ['From Native Gate', 'Custom Schedule', 'From Array']
        pulse_input_lst = []

        ### User Interface (UI) Components ###

        # Dropdown menu for backend
        backend_input_dd = widgets.Dropdown(options=backend_input_lst, 
#                                             layout=widgets.Layout(width='auto'),
                                            continuous_update=False,
                                            disabled=False)

        # Dropdown menu for waveform input
        schedule_input_dd = widgets.Dropdown(options=schedule_input_lst, 
#                                              layout=widgets.Layout(width='auto'),
                                             continuous_update=False,
                                             disabled=False)

        # Dropdown menu for native gate selection
        nativegate_input_dd = widgets.Dropdown(options=nativegate_input_lst[0:len(nativegate_input_lst)-1], 
                                               layout=widgets.Layout(width='160px'),
                                               continuous_update=False,
                                               disabled=False)

        # Floating Textbox for Phase Shift input
        '''TODO: Need to be able to pass multiples of np.pi or pi'''
        shiftphase_input_fltxt = widgets.BoundedFloatText(value=0.0, min=0.0, max=1.0, step=0.001,
                                                           layout=widgets.Layout(width='160px'),
                                                           disabled=False)

        # Floating Textbox for Frequency value input
        shiftfreq_input_fltxt = widgets.BoundedFloatText(value=4.75, min=4.0, max=5.5, step=0.001,
                                                          layout=widgets.Layout(width='160px'),
                                                          disabled=False)
        
        # Dropdown menu for pulse array selection
        pulse_input_dd = widgets.Dropdown(options=pulse_input_lst, 
                                          layout=widgets.Layout(width='160px'),
                                          continuous_update=False,
                                          disabled=False)


         # Dropdown menu for channel selection to append to schedule
        append_to_dd = widgets.Dropdown(options=self._backend_chan_lst, 
                                             layout=widgets.Layout(width='160px'),
                                             description='Append to:',
                                             continuous_update=False,
                                             disabled=False)

        # Button to append selected value to schedule
        append_to_btn = widgets.Button(description='Append',
                                              layout=widgets.Layout(width='80px'),
                                              disabled=False)

        def toggle_append_type(change):
            append_type = change['owner'].options.index(change['new'])
            nativegate_input_dd.disabled = True
            shiftphase_input_fltxt.disabled = True
            shiftfreq_input_fltxt.disabled = True
            pulse_input_dd.disabled = True
            
            if append_type == 0:
                nativegate_input_dd.disabled = False
                append_to_dd.options = self._backend_qubit_lst
            elif append_type == 1:
                shiftphase_input_fltxt.disabled = False
                append_to_dd.options = self._backend_chan_lst
            elif append_type == 2:
                shiftfreq_input_fltxt.disabled = False
                append_to_dd.options = self._backend_chan_lst            
            elif append_type == 3:
                pulse_input_dd.disabled = False
                append_to_dd.options = self._backend_chan_lst

        append_type_radio = widgets.RadioButtons(
            options=['From Gate:', 'Phase [1/(2π)]:', 'Freq [GHz]:', 'Pulse:'],
           layout=widgets.Layout(width='max-content', display='flex', flex='1 0 auto'), # If the items' names are long
        )
        append_type_radio.observe(toggle_append_type, names='value');
        toggle_append_type({'new': append_type_radio.value, 'owner': append_type_radio}) # call it once to initialize

        append_type_panel = widgets.VBox([widgets.HBox([append_type_radio,
                                              widgets.VBox([nativegate_input_dd, 
                                                            shiftphase_input_fltxt,
                                                            shiftfreq_input_fltxt,
                                                            pulse_input_dd])]),
                                          widgets.HBox([append_to_dd,
                                                       append_to_btn],
                                                      layout=widgets.Layout(justify_content='space-between', margin='40px 0px 0px 0px'))])

        clear_btn = widgets.Button(description='Clear',
                                   layout=widgets.Layout(width='auto', height='auto'))
                                   
        # Combines all dropdown menus in a left panel
        left_panel = widgets.VBox([widgets.Label("Backend:"), backend_input_dd,
                                   widgets.Label("Input:"), schedule_input_dd,
                                   widgets.HBox([widgets.Label("Schedule:"), clear_btn],
                                                 layout=widgets.Layout(justify_content='space-between')),
                                   append_type_panel])

        
        ### Widget Interactions ###
        def clear_data(b):
            self.pulses.clear()    
            self.phases.clear()        
            self.freqs.clear()        
            self.samples.clear()
            self.update()
        
        clear_btn.on_click(clear_data)

        # Update dropdown options for gates and channels based on selected backend
        def update_dd_options(*args):
            current_backend = backend_input_dd.value
            self._backend_qubit_lst, self._backend_chan_lst, self._backend_cmap_lst, nativegate_input_lst = qiskit_backend_config(current_backend, backend_input_lst, backend_qnum_lst)

            self._backend_cmap_nms = []
            if self._backend_cmap_lst is not None:
                for qb_pair in self._backend_cmap_lst:
                    self._backend_cmap_nms.append('q'+str(qb_pair[0])+'->q'+str(qb_pair[1]))
            else:
                self._backend_cmap_nms = None

#             print(self._backend_cmap_nms)

            nativegate_input_dd.options = nativegate_input_lst
            if not nativegate_input_dd.disabled:
                append_to_dd.options = self._backend_qubit_lst
            elif not shiftphase_input_fltxt.disabled:
                append_to_dd.options = self._backend_chan_lst
            elif not shiftfreq_input_fltxt.disabled:
                append_to_dd.options = self._backend_chan_lst  
            elif not pulse_input_dd.disabled:
                append_to_dd.options = self._backend_chan_lst

        backend_input_dd.observe(update_dd_options, 'value')          

        # Update Schedule when append button is pressed
        def update_schedule(b):
            '''NOTE: Not sure if I need _current_phase, _current_freq or if I should just use 
                     the values from the widgets directly:'''
            self._current_phase = 2*np.pi*shiftphase_input_fltxt.value
            self._current_freq = 1e9*shiftfreq_input_fltxt.value

            if not nativegate_input_dd.disabled:
                '''TODO: Will need to add conditionals here depending on the provider. Right now, everythin is qiskit
                         so only calling qiskit_gate_to_sched() function to make translations'''
                
                backend_indx = backend_input_lst.index(backend_input_dd.value)
                num_qbs = backend_qnum_lst[backend_indx]
                
                qiskit_gate_sch = qiskit_gate_to_sched(backend_input_dd.value, nativegate_input_dd.value, self._current_qubits, num_qbs)
                phases, freqs, pulses = qiskit_to_schedviz(qiskit_gate_sch, self.samples)

                for chan, pulse in pulses.items():
                    if chan in self.pulses.keys():
                        # Check if channel is already present in pulses to append new data
                        # Else, add channel to pulses.
                        pulse_array = np.append(self.pulses[chan],pulse)
                    else:
                        pulse_array = pulse
                    self.pulses[chan] = pulse_array
                    self.samples[chan] = len(pulse_array)

                for chan, phase in phases.items():
                    if chan in self.phases.keys():
                        # Check if channel is already present in pulses to append new data
                        # Else, add channel to pulses.
                        '''TODO: Need to be careful here. Gotta check if there is phaseshift overlap bw last elem in self.phases
                                 and the first elem being appended from the native gate to. The new one should replace old one'''
                        phase_array = self.phases[chan] + phase
                    else:
                        phase_array = phase
                    self.phases[chan] = phase_array

                '''NOTE: Don't have a for loop for freq.items() here bc pulses from native gates in qiskit don't contain
                         that type of instruction. Might need to add later for other backends?'''

            elif not shiftphase_input_fltxt.disabled:
                if self._current_chann in self.samples.keys():
                    current_sample = self.samples[self._current_chann]
                else:
                    current_sample = 0

                phase = [current_sample, self._current_phase]

                if self._current_chann in self.phases.keys():
                    # Check if channel is already present in phases to append/replace new data
                    # Else, add channel to phases.
                    
                    phase_array = self.phases[self._current_chann]

                    if phase_array[-1][0] == current_sample:
                        # If sample number hasn't changed, replace PhaseShift value
                        # Else, append new [time,PhaseShift] item to phases
                        phase_array[-1] = phase
                    else:
                        phase_array += [phase]
                else:
                    phase_array = [phase]

                self.phases[self._current_chann] = phase_array

            elif not shiftfreq_input_fltxt.disabled:
                if self._current_chann in self.samples.keys():
                    current_sample = self.samples[self._current_chann]
                else:
                    current_sample = 0

                freq = [current_sample, self._current_freq]

                if self._current_chann in self.freqs.keys():
                    # Check if channel is already present in freqs to append/replace new data
                    # Else, add channel to freqs.
                    freq_array = self.freqs[self._current_chann]

                    if freq_array[-1][0] == current_sample:
                        # If sample number hasn't changed, replace Frequency value
                        # Else, append new [time,FreqValue] item to freqs
                        freq_array[-1] = freq
                    else:
                        freq_array += [freq]
                else:
                    freq_array = [freq]

                self.freqs[self._current_chann] = freq_array

            elif not pulse_input_dd.disabled:
                '''
                TODO: Need to add padding option (add dotted line of where schedule stands on each channel?)
                         Best way might be to add padding for visualization in the plot_sch function, but keep
                         arrays for each channel true to what the user is adding. The challenge is then keeping
                         track of the "current_sample" for each channel individually, but might be as simple as
                         always checking the length of the pulse array for each specific chan.
                '''
                pulse = self.dummy_pulse 

                if self._current_chann in self.pulses.keys():
                    # Check if channel is already present in pulses to append new data
                    # Else, add channel to pulses.
                    pulse_array = np.append(self.pulses[self._current_chann],pulse)
                else:
                    pulse_array = pulse

                self.pulses[self._current_chann] = pulse_array
                self.samples[self._current_chann] = len(pulse_array)

            self.update()

        append_to_btn.on_click(update_schedule)
        
        '''
        NOTE: Don't know if I really need the update_channels() and update_qubits() functions. I might be OK just working with
              the .value items of each widget instead of saving them in the self._current_chann & self._current_qbuit lists.
              In particular, the update_qubits function might be redundant. Could do the same inside the qiskit_gate_to_sched() function
              by just passing append_to_dd.value directly.
              Only reason to have them might be to support backends from other companies? (e.g. Rigetti)
        '''

        # Update current Channels based on values of dropdown menus
        '''NOTE: can I use a labmda function here to avoid the extra definition?'''
        def update_channels(*args):
            self._current_chann = append_to_dd.value

        append_to_dd.observe(update_channels, 'value')

        # Update self._current_qubits based on value of the dropdown menu
        def update_curr_qubits(*args):
            if not nativegate_input_dd.disabled:
                if nativegate_input_dd.value == 'CX':
                    qubits_indx = self._backend_cmap_nms.index(append_to_dd.value)
                    qb_pair = self._backend_cmap_lst[qubits_indx]
                    self._current_qubits = ['q'+str(qb) for qb in qb_pair]

                else: 
                    self._current_qubits = [append_to_dd.value, append_to_dd.value]
#                 print(self._current_qubits)

        append_to_dd.observe(update_curr_qubits, 'value')

        # Update dropdown options for qubit selection based on single-qubit or two-qubit gate
        def update_dd_qubits(*args):
            if nativegate_input_dd.value == 'CX':
                append_to_dd.options = self._backend_cmap_nms
            else:
                append_to_dd.options = self._backend_qubit_lst

        nativegate_input_dd.observe(update_dd_qubits, 'value')

        # Plot schedule when outputs change
        self._plot = plot_pulse_schedule(self.phases, self.freqs, self.pulses, self.samples)

        # TEST FIG OUT (NOTE: TO BE DELETED)
        #wf_fig_out = widgets.interactive_output(plot_wf, {'nativegate':nativegate_input_dd})

        schedule_editor = widgets.HBox([left_panel, self._plot])
        super().__init__([schedule_editor])

        self._editor = schedule_editor
    
    def update(self):
        self._plot.update()