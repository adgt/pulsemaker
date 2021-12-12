from enum import IntEnum
import ipywidgets as widgets
import traitlets as traitlets
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

def get_qiskit_backend(backend_name):
    try:
        from qiskit.test.mock import FakeArmonk, FakeAlmaden, FakeAthens

    except ImportError:
        pass

    '''TODO: might be able to do this in a better way without having to do conditional checks for each backend name.'''
    if backend_name == 'Armonk':
        backend = FakeArmonk()
    elif backend_name == 'Almaden':
        backend = FakeAlmaden()
    elif backend_name == 'Athens':
        backend = FakeAthens()
    else:
        '''TODO: There is no FakeCasablanca, so using FakeAlmaden for it right now'''
        backend = FakeAlmaden()
    return backend

def qiskit_backend_config(backend_name, backend_input_lst, backend_qnum_lst):
    '''Generate qubit, channel and coupling map lists based on selected backend'''

    backend = get_qiskit_backend(backend_name)
    
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
    gate_lst = ['X','Y','Z','H','SX','RZ','CX']
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


def qiskit_gate_to_sched(backend_name, gate_name, qubits, num_qbs):
    '''Build qiskit schedule based on selected gate and backend'''

    try:
        from qiskit import QuantumCircuit
        from qiskit import transpile, schedule as build_schedule
        from qiskit.test.mock import FakeArmonk, FakeAlmaden, FakeAthens

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
    
    backend = get_qiskit_backend(backend_name)
    
    trans_circuit = transpile(circuit, backend)
    return build_schedule(trans_circuit, backend)

def qiskit_circuit_qasm_to_sched(backend_name, circuit_qasm):    
    '''Build qiskit schedule based on selected gate and backend'''

    try:
        from qiskit import QuantumCircuit
        from qiskit.qasm import QasmError
        from qiskit import transpile, schedule as build_schedule
        from qiskit.test.mock import FakeArmonk, FakeAlmaden, FakeAthens

        backend = get_qiskit_backend(backend_name)

        try:
            circuit = QuantumCircuit.from_qasm_str(circuit_qasm)
        except QasmError:
            circuit = QuantumCircuit(1)
        trans_circuit = transpile(circuit, backend)
        return build_schedule(trans_circuit, backend)

    except ImportError:
        pass    

class InstructionType(IntEnum):
    # Keep these in sync with AppendType
    PAD = 0
    PHASE = 1
    FREQ = 2
    PULSE = 3
    
    def __str__(self):
        if self == self.PAD:
            return "Pad"
        elif self == self.PHASE:
            return "Phase"
        elif self == self.FREQ:
            return "Freq"
        elif self == self.PULSE:
            return "Pulse"

def get_max_sample_count(instructions, channel=None):
    '''Get the max sample count for either a specific channel or all channels (default)'''

    max_sample_count = 0

    if channel is None:
        for channel_instructions in instructions.values():
            sample_count = 0
            for t, data in channel_instructions:
                if t is InstructionType.PULSE:
                    sample_count += len(data)
                elif t is InstructionType.PAD:
                    sample_count += data
            if sample_count > max_sample_count:
                max_sample_count = sample_count
    else:
        channel_instructions = instructions[channel]
        for t, data in channel_instructions:
            if t is InstructionType.PULSE:
                max_sample_count += len(data)
    
    return max_sample_count

def qiskit_to_pulsemaker(qiskit_sch):
    '''Translate qiskit-schedule to pulsemaker format'''

    try:
        from qiskit.pulse import Play, SetFrequency, ShiftPhase
        from qiskit.pulse.channels import DriveChannel, ControlChannel

    except ImportError:
        return {}
    
    instructions = {}

    for start_time, instruction in qiskit_sch.instructions:

        if isinstance(instruction.channel, DriveChannel):
            chan = 'd'+str(instruction.channel.index)
        elif isinstance(instruction.channel, ControlChannel):
            chan = 'u'+str(instruction.channel.index)
        else:
            '''TODO: Might need to add support for other channels, like measure, acquire, etc'''
            pass
        
        payload = None
        if isinstance(instruction, Play):
            payload = (InstructionType.PULSE, np.array(instruction.pulse.get_waveform().samples))
            
        elif isinstance(instruction, ShiftPhase):
            payload = (InstructionType.PHASE, instruction.phase)

        elif isinstance(instruction, SetFrequency):
            payload = (InstructionType.FREQ, instruction.frequency)

        instructions[chan] = instructions.get(chan, []) + [payload]

    return instructions

def pulsemaker_to_qiskit(instructions):
    '''Translate pulsemaker to qiskit-schedule format'''

    try:
        from qiskit import pulse
        from qiskit.pulse import Schedule, Play, DriveChannel, ControlChannel, Waveform, ShiftPhase, SetFrequency
    except ImportError:
        return Schedule()

    schedule = Schedule()
    for c, data in instructions.items():
        channel_type = c[0]
        channel_index = int(c[1:])
        if channel_type == 'd':
            channel = DriveChannel(channel_index)
        elif channel_type == 'u':
            channel = ControlChannel(channel_index)
        
        current_sample_count = 0
        for type, payload in data:
            if type == InstructionType.PULSE:
                sample_count = len(payload)
                if sample_count == 1:
                    continue
                schedule |= Play(Waveform(payload), channel).shift(current_sample_count)
                current_sample_count += sample_count
            elif type == InstructionType.PHASE:
                schedule |= ShiftPhase(payload, channel).shift(current_sample_count)
            elif type == InstructionType.FREQ:
                schedule |= SetFrequency(payload, channel).shift(current_sample_count)
                    
    return schedule

def simulate(qiskit_schedule, backend_name):
    if qiskit_schedule.duration == 0:
        return {'0': 1}
    if backend_name == 'Armonk':
        return run_with_measure(qiskit_schedule, backend_name, 2).get_counts()
    else:
        print("Only FakeArmonk is supported for simulation currently because other backends are too slow")
        return {'0': 1}
    
def run_with_measure(qiskit_schedule, backend_name, meas_level=1):
    try:
        from qiskit import providers, assemble
        from qiskit.pulse import DriveChannel, SetFrequency
        from qiskit.pulse.macros import measure
        from qiskit.result.result import Result
        
        if qiskit_schedule.duration == 0:
            return Result(backend_name, None, None, None, None, None)
        if backend_name == 'Armonk':
            backend = get_qiskit_backend(backend_name)
            pulse_sim = providers.aer.PulseSimulator.from_backend(backend)
            pulse_qobj = assemble(qiskit_schedule, backend=pulse_sim)
            measure_qubits = []
            for channel in qiskit_schedule.channels:
                if isinstance(channel, DriveChannel):
                    measure_qubits.append(channel.index)
            frequency = None
            for start_time, instruction in qiskit_schedule.instructions:
                if isinstance(instruction, SetFrequency):
                    frequency = {instruction.channel: instruction.frequency}
                    break

            def strip_frequencies(instruction):
                if isinstance(instruction[1], SetFrequency):
                    return False                
                return True

            # Setting frequences isn't supported on simulators, so instead we use `schedule_los` to set a single frequency
            # and subsequently strip any SetFrequency instructions from the schedule.
            qiskit_schedule = qiskit_schedule.filter(strip_frequencies)
            qiskit_schedule += measure(measure_qubits, pulse_sim) << qiskit_schedule.duration
            pulse_qobj = assemble(qiskit_schedule, backend=pulse_sim, meas_level=meas_level, schedule_los=frequency)
            job = pulse_sim.run(pulse_qobj)
            return job.result()
        else:
            print("Only FakeArmonk is supported for simulation currently because other backends are too slow")
            return Result(backend_name, None, None, None, None, None)
    except ImportError:
        pass

def plot_pulse_schedule(instructions):
    '''Function to draw/update schedule plot'''
    def _plot(instructions):        
        sorted_instructions = instructions.copy()
        
        labels = ['a','d','m','u'] # labels for different channels:
                                   # a: acquire, d: drive, m: measure, u: x-correlation

        num_chans = max(len(sorted_instructions), 1)
        gs = gridspec.GridSpec(num_chans, 1)
        ax = []

        ''' 
        To sort pulse dictionary, channel index values are calculated as follows:
        indx[0][0] stores the type of channel: a, d, m, u. By using the index value of the 'labels' list, 
        we know the position the channel holds within a given qubit. indx[0][1] stores the qubit value, so 
        by multiplying by the length of the 'labels' list, we know where the qubit sits wrt the others.
        by adding the two, we know where a given qubit channel should sit wrt to other channels.
        ''' 
        sorted_instructions = sorted(sorted_instructions.items(), 
                        key=lambda indx: (labels.index(indx[0][0])+int(indx[0][1])*len(labels)))


        fig = plt.subplots(figsize=(9,5))        

        for chan_num, chan_data in enumerate(sorted_instructions):
            if chan_num == 0:
                ax.append(plt.subplot(gs[chan_num]))
                ax[0].set_xlabel('Samples (t/dt)')
            else: 
                ax.append(plt.subplot(gs[chan_num], sharex=ax[0]))
            if chan_num < num_chans - 1:
                plt.setp(ax[chan_num].get_xticklabels(), visible=False)

            ''' TODO: Axis settings. Still need to decide how they should look like '''
            ###ax[chan_num].text(0,0, chan[0], horizontalalignment='center',verticalalignment='center', fontweight='bold')
            ax[chan_num].tick_params(axis='y', which='major', labelsize=7)
            #ax[chan_num].tick_params(axis="y",direction="in", pad=-22)
            #ax[chan_num].get_yaxis().set_ticks([])
            ax[chan_num].set_ylabel(chan_data[0]+'  ', rotation=0, fontweight='bold')

            instructions = chan_data[1]
            current_plot_sample = 0
            
            for t, data in instructions:
                if t == InstructionType.PULSE or t == InstructionType.PAD:
                    # Construct real/imag signals to be plotted
                    if t == InstructionType.PAD:
                        i_sig = np.zeros(data)
                        q_sig = np.zeros(data)
                    else:
                        i_sig = np.real(data)
                        q_sig = np.imag(data)

                    samps = i_sig.size
                    t = current_plot_sample + np.linspace(0,samps,samps)

                    if chan_data[0][0] == 'd':
                        ax[chan_num].step(t, i_sig, 'r')
                        ax[chan_num].fill_between(t, i_sig, color='r', alpha=0.2, step='pre')
                        ax[chan_num].step(t, q_sig, 'b')
                        ax[chan_num].fill_between(t, q_sig, color='b', alpha=0.2, step='pre')
                    else:
                        '''
                        TODO: Here I'm using an else statement to display anything that isn't a drive 
                            channel 'd' as if it was a control channel 'u'. If support for 'a' 'm' channels is added
                            need to make this into an 'elif chan[0][0] == 'u':' and also change colors of those channels too
                        '''
                        ax[chan_num].step(t, i_sig, 'y')
                        ax[chan_num].fill_between(t, i_sig, color='y', alpha=0.2, step='pre')
                        ax[chan_num].step(t, q_sig, 'orange')
                        ax[chan_num].fill_between(t, q_sig, color='orange', alpha=0.2, step='pre')

                    current_plot_sample += len(i_sig)

                elif InstructionType.PHASE:
                    ax[chan_num].text(x=current_plot_sample, y=0, s=r'$\circlearrowleft$',
                                      fontsize=14, color='purple',
                                      ha='center', va='center')

                elif InstructionType.FREQ:
                    ax[chan_num].text(x=current_plot_sample, y=0, s=r'$\downarrow$',
                                      fontsize=14, color='forestgreen',
                                      ha='center', va='bottom')

        plt.subplots_adjust(hspace=.0)

    return widgets.interactive(_plot, instructions=widgets.fixed(instructions))

from qiskit.pulse import Schedule
from qiskit.result.result import Result
class ScheduleDesigner(widgets.VBox):
    class AppendType(IntEnum):
        GATE = 0
        PHASE = 1
        FREQ = 2
        PULSE = 3
        CIRCUIT = 4
        
        def __str__(self):
            if self == self.GATE:
                return "From Gate"
            elif self == self.PHASE:
                return "Phase [2π]"
            elif self == self.FREQ:
                return "Freq [GHz]"
            elif self == self.PULSE:
                return "From Pulse"
            elif self == self.CIRCUIT:
                return "From Circuit"
            
    schedule = traitlets.Instance(Schedule)
    result = traitlets.Instance(Result)
    circuit_qasm = traitlets.Unicode()
        
    def __init__(self):

        ### Initialize ###  
        self.instructions = {}     # Dictionary of instructions (i.e. pulses, phases, and frequencies)
                                   #  Format: {d0:instructions_d0, u0:instructions_01, d1:instructions_d1, ...} 
                                   #    instructions_xx is a list of (InstructionType, data) tuples
                                   #    data is either a numpy array for pulse data, a phase, or a frequency value
        self.custom_pulse  = []    # Custom pulse
                                   #  Format: numpy array with the pulse data
        
        self._current_qubits = ['q0','q0']      # qubit(s) currently selected. If a single-qubit gate is selected, then 
                                                # both elements in the list take the same value ['qx','qx'] (qx is currently selecte qubit)
                                                # If two-qubit gate (CX) is selected: ['qx','qy'] where CX is applied from qx to qy.

        self.schedule = Schedule()  # Final schedule (currently will use Qiskit's data structuring) 
        
        self.edit_item = None       # Editing mode of an existing instruction
        phase_multiplier = 2*np.pi
        freq_multiplier = 1e9

        def get_collated_schedule():
            '''combined list of pulse schedule in format (w/o payload): (time, channel, type)'''
            schedule = []        
            for channel, data in self.instructions.items():
                sample_count = 0
                for i, (type, payload) in enumerate(data):
                    schedule += [(i, sample_count, channel, type)]
                    if type == InstructionType.PULSE:
                        sample_count += len(payload)
                    elif type == InstructionType.PAD:
                        sample_count += payload
            schedule.sort(key=lambda x: x[0])
            return schedule

        backend_input_lst = ['Armonk', 'Almaden', 'Athens', 'Casablanca']
        backend_qnum_lst = [1, 20, 5, 7]
        
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
        
        ### User Interface (UI) Components ###

        # Dropdown menu for backend
        backend_input_dd = widgets.Dropdown(options=backend_input_lst, 
                                            layout=widgets.Layout(width='100px'),
                                            continuous_update=False,
                                            disabled=False)
        
        backend_autosim = widgets.Checkbox(value=False,
                                           description='',
                                           layout=widgets.Layout(width='100px', align_self='flex-start'),
                                           disabled=False)
        backend_autosim.style.description_width = "0"
        self._backend_autosim = backend_autosim

        # Dropdown menu for native gate selection
        nativegate_input_dd = widgets.Dropdown(options=nativegate_input_lst[0:len(nativegate_input_lst)-1], 
                                               layout=widgets.Layout(width='160px'),
                                               continuous_update=False,
                                               disabled=False)

        # Floating Textbox for Phase Shift input
        '''TODO: Need to be able to pass multiples of np.pi or pi'''
        shiftphase_input_fltxt = widgets.BoundedFloatText(value=0.0, min=-1.0, max=1.0, step=0.001,
                                                           layout=widgets.Layout(width='160px'),
                                                           disabled=False)

        # Floating Textbox for Frequency value input
        shiftfreq_input_fltxt = widgets.BoundedFloatText(value=4.0, min=4.0, max=5.5, step=0.001,
                                                          layout=widgets.Layout(width='160px'),
                                                          disabled=False)
        
        apply_btn = widgets.Button(description='Apply',
                                              layout=widgets.Layout(width='80px'),
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
        
        append_input_panel = widgets.HBox([nativegate_input_dd, 
                                    shiftphase_input_fltxt,
                                    shiftfreq_input_fltxt])

        def toggle_append_type(change):
            append_type = change['owner'].options.index(change['new'])

            append_to_dd.layout.visibility = 'visible'
            append_to_btn.description = 'Append'

            if append_type == self.AppendType.GATE:
                nativegate_input_dd.value = nativegate_input_lst[0]
                append_input_panel.children = [nativegate_input_dd]
                append_to_dd.options = self._backend_qubit_lst
            elif append_type == self.AppendType.PHASE:
                if self.edit_item and self.edit_item[0] == InstructionType.PHASE:
                    append_input_panel.children = [shiftphase_input_fltxt, apply_btn]
                else:
                    append_input_panel.children = [shiftphase_input_fltxt]
                append_to_dd.options = self._backend_chan_lst
            elif append_type == self.AppendType.FREQ:
                if self.edit_item and self.edit_item[0] == InstructionType.FREQ:
                    append_input_panel.children = [shiftfreq_input_fltxt, apply_btn]
                else:                    
                    append_input_panel.children = [shiftfreq_input_fltxt]
                append_to_dd.options = self._backend_chan_lst            
            elif append_type == self.AppendType.PULSE:
                append_input_panel.children = []
                append_to_dd.options = self._backend_chan_lst
            elif append_type == self.AppendType.CIRCUIT:
                append_input_panel.children = []
                append_to_dd.layout.visibility = 'hidden'
                append_to_btn.description = 'Replace'
                pass

        append_type_select = widgets.ToggleButtons(
            options=list(self.AppendType),
#             layout=widgets.Layout(width='max-content', display='flex', flex='1 0 auto'), # If the items' names are long
        )
        append_type_select.style.button_width = "100px"
        append_type_select.observe(toggle_append_type, names='value')
        # NOTE: Another option than the code below is to cycle the .value to trigger a call
        toggle_append_type({'new': append_type_select.value, 'owner': append_type_select}) # call it once to initialize
                
        append_panel = widgets.VBox([append_type_select,
                                     append_input_panel])
                                           
        # Combines all dropdown menus in a left panel
        input_panel = widgets.HBox([widgets.VBox([
                                                widgets.HBox([widgets.Label("Backend:"), backend_input_dd]), 
                                                widgets.HBox([widgets.Label("Auto-run:"), backend_autosim])],
                                                layout=widgets.Layout(justify_content='flex-start')),
                                            append_panel],
                                            layout=widgets.Layout(justify_content='space-between'))
        
        ### Widget Interactions ###
        
        def update_schedule():
            schedule = get_collated_schedule()
            if hasattr(self, '_schedule_list'):
                index = self._schedule_list.index
                self._schedule_list.options = [f"{channel}: {instruction}" for (index, time, channel, instruction) in schedule]
                if index is not None and len(self._schedule_list.options) > 0:
                    self._schedule_list.index = min(len(self._schedule_list.options) - 1, index)

            self.schedule = pulsemaker_to_qiskit(self.instructions)
            
            if self._backend_autosim.value == True:
                self.result = run_with_measure(self.schedule, self._current_backend)

            self.update()

        # Clear schedule
        def clear_data(*args):
            self.instructions.clear()
            self.schedule = Schedule()
            update_schedule()

        clear_btn = widgets.Button(description='Clear', layout=widgets.Layout(align_self='flex-start', width='auto', height='auto'))
        clear_btn.on_click(clear_data)

        # Update dropdown options for gates and channels based on selected backend
        def update_dd_options(*args):
            self._current_backend = backend_input_dd.value
            self._backend_qubit_lst, self._backend_chan_lst, self._backend_cmap_lst, \
            nativegate_input_lst = qiskit_backend_config(self._current_backend, backend_input_lst, backend_qnum_lst)

            self._backend_cmap_nms = []
            if self._backend_cmap_lst is not None:
                for qb_pair in self._backend_cmap_lst:
                    self._backend_cmap_nms.append('q'+str(qb_pair[0])+'->q'+str(qb_pair[1]))
            else:
                self._backend_cmap_nms = None

            # print(self._backend_cmap_nms)

            append_type = append_type_select.options.index(append_type_select.value)
                                    
            nativegate_input_dd.options = nativegate_input_lst
            nativegate_input_dd.value = nativegate_input_lst[0]
            if append_type == self.AppendType.GATE:
                append_to_dd.options = self._backend_qubit_lst
            else:
                append_to_dd.options = self._backend_chan_lst
                
            clear_data(*args)
            
        backend_input_dd.observe(update_dd_options, 'value')          
        update_dd_options({'new': backend_input_dd.value, 'owner': backend_input_dd}) # call it once to initialize

        def append_to_schedule(*args):
            def pad_to_max_sample():
                max_sample_count = get_max_sample_count(self.instructions)
                for chan in self.instructions.keys():
                    channel_max_sample_count = get_max_sample_count(self.instructions, chan)
                    if channel_max_sample_count < max_sample_count:
                        self.instructions[chan] += [InstructionType.PAD, max_sample_count - channel_max_sample_count]

            current_chan = append_to_dd.value
            current_phase = phase_multiplier * shiftphase_input_fltxt.value
            current_freq = freq_multiplier * shiftfreq_input_fltxt.value
            
            append_type = append_type_select.options.index(append_type_select.value)
                        
            if append_type == self.AppendType.GATE:
                '''TODO: Will need to add conditionals here depending on the provider. Right now, everythin is qiskit
                         so only calling qiskit_gate_to_sched() function to make translations'''

                backend_indx = backend_input_lst.index(backend_input_dd.value)
                num_qbs = backend_qnum_lst[backend_indx]

                qiskit_gate_sch = qiskit_gate_to_sched(backend_input_dd.value, nativegate_input_dd.value, self._current_qubits, num_qbs)

                # Pad before gate is applied
                pad_to_max_sample()

                instructions = qiskit_to_pulsemaker(qiskit_gate_sch)

                for chan in instructions.keys():
                    self.instructions[chan] = self.instructions.get(chan, []) + instructions[chan]
                
                # Pad after gate schedule is appended to full schedule
                pad_to_max_sample()

            elif append_type == self.AppendType.PHASE:
                # Check if the last phase that was set is the same phase
                same_phase = False
                for type, data in reversed(self.instructions):
                    if type == InstructionType.PHASE:
                        if data == current_phase:
                            same_phase = True
                        break

                if not same_phase:
                    phase = (InstructionType.PHASE, current_phase)
                    self.instructions[current_chan] = self.instructions.get(current_chan, []) + [phase]

            elif append_type == self.AppendType.FREQ:
                # Check if the last frequency that was set is the same frequency
                same_freq = False
                for type, payload in reversed(self.instructions):
                    if type == InstructionType.FREQ:
                        if payload == current_freq:
                            same_freq = True
                        break

                if not same_freq:
                    freq = (InstructionType.FREQ, current_freq)
                    self.instructions[current_chan] = self.instructions.get(current_chan, []) + [freq]

            elif append_type == self.AppendType.PULSE:
                current_pulse = self.custom_pulse
                pulse = (InstructionType.PULSE, current_pulse)
                self.instructions[current_chan] = self.instructions.get(current_chan, []) + [pulse]

            elif append_type == self.AppendType.CIRCUIT:
                clear_data(*args)
                qiskit_gate_sch = qiskit_circuit_qasm_to_sched(self._current_backend, self.circuit_qasm)
                instructions = qiskit_to_pulsemaker(qiskit_gate_sch)
                for chan in instructions.keys():
                    self.instructions[chan] = self.instructions.get(chan, []) + instructions[chan]
            
            update_schedule()
                
        append_to_btn.on_click(append_to_schedule)
        
        '''
        NOTE: Don't know if I really need the update_qubits() functions. I might be OK just working with
              the .value items of each widget instead of saving them in the self._current_chann & self._current_qbuit lists.
              In particular, the update_qubits function might be redundant. Could do the same inside the qiskit_gate_to_sched() function
              by just passing append_to_dd.value directly.
              Only reason to have them might be to support backends from other companies? (e.g. Rigetti)
        '''
        # Update self._current_qubits based on value of the dropdown menu
        def update_curr_qubits(*args):
            append_type = append_type_select.options.index(append_type_select.value)
                        
            if append_type == self.AppendType.GATE:
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

        def select_schedule_item(button):
            schedule = get_collated_schedule()
            index = self._schedule_list.index
            if index is None:
                schedule_item_edit_btn.disabled = True
                return

            (item_index, time, channel, type) = schedule[index]

            if not (type == InstructionType.PHASE or type == InstructionType.FREQ):
                self.edit_item = None
                schedule_item_edit_btn.disabled = True
            else:
                schedule_item_edit_btn.disabled = False

        def edit_schedule_item(button):
            schedule = get_collated_schedule()
            index = self._schedule_list.index
            (item_index, time, channel, type) = schedule[index]

            instructions = self.instructions[channel]
            edit_item = instructions[item_index]
            if self.edit_item == edit_item:
                self.edit_item = None
                append_type = append_type_select.value                    
                append_type_select.value = 0 # set to something else to force a refresh
                append_type_select.value = append_type                                        
            else:
                self.edit_item = edit_item
                if type == InstructionType.PHASE:
                    append_type_select.value = 0 # set to something else to force a refresh
                    append_type_select.value = self.AppendType.PHASE
                    shiftphase_input_fltxt.value = self.edit_item[1] / phase_multiplier
                elif type == InstructionType.FREQ:
                    append_type_select.value = 0 # set to something else to force a refresh
                    append_type_select.value = self.AppendType.FREQ
                    shiftfreq_input_fltxt.value = self.edit_item[1] / freq_multiplier
                    
        def on_freq_updated(change):
            # Auto-update value when autosim is turned on
            if self._backend_autosim.value == True and self.edit_item:
                update_edited_item(apply_btn)
        shiftfreq_input_fltxt.observe(on_freq_updated)

        def update_edited_item(button):
            edit_item = self.edit_item
            if edit_item:
                type = edit_item[0]
                index = -1
                instructions = None
                for channel, data in self.instructions.items():
                    item_index = data.index(edit_item)
                    if item_index >= 0:
                        index = item_index
                        instructions = self.instructions[channel]
                        break

                if index >= 0:
                    if type == InstructionType.PHASE:
                        payload = (InstructionType.PHASE, phase_multiplier * shiftphase_input_fltxt.value)
                    elif type == InstructionType.FREQ:                
                        payload = (InstructionType.FREQ, freq_multiplier * shiftfreq_input_fltxt.value)

                    instructions[index] = payload
                    self.edit_item = payload
            update_schedule()

        def delete_schedule_item(*args):
            schedule = get_collated_schedule()
            index = self._schedule_list.index
            (item_index, time, channel, type) = schedule[index]
  
            instructions = self.instructions[channel]
            del instructions[item_index]

            if len(instructions) == 0:
                del self.instructions[channel]

            del schedule[index]
            update_schedule()

        def move_schedule_item(button):
            up = button.tooltip == "Up"

            schedule = get_collated_schedule()
            index = self._schedule_list.index            
            (item_index, time, channel, type) = schedule[index]

            instructions = self.instructions[channel]
            if up and item_index - 1 >= 0:
                previous = instructions[item_index - 1]
                instructions[item_index - 1] = instructions[item_index]
                instructions[item_index] = previous
            elif not up and item_index + 1 < len(instructions):
                next = instructions[item_index + 1]
                instructions[item_index + 1] = instructions[item_index]
                instructions[item_index] = next

            if up:
                self._schedule_list.index = max(index - 1, 0)
            else:
                self._schedule_list.index = min(index + 1, len(self._schedule_list.options) - 1)                
            update_schedule()

        apply_btn.on_click(update_edited_item)

        self._schedule_list = widgets.Select(
            description='',
            disabled=False,
            layout=widgets.Layout(width='100px', align_items='stretch')
        )        
        self._schedule_list.observe(select_schedule_item, names='value')
        schedule_item_edit_btn = widgets.Button(description='📝', tooltip='Edit', layout=widgets.Layout(width='40px'))
        schedule_item_up_btn = widgets.Button(description='🔼', tooltip='Up', layout=widgets.Layout(width='40px'))
        schedule_item_down_btn = widgets.Button(description='🔽', tooltip='Down', layout=widgets.Layout(width='40px'))
        schedule_item_del_btn = widgets.Button(description='❌', layout=widgets.Layout(width='40px'))
        schedule_item_edit_btn.on_click(edit_schedule_item)
        schedule_item_up_btn.on_click(move_schedule_item)
        schedule_item_down_btn.on_click(move_schedule_item)
        schedule_item_del_btn.on_click(delete_schedule_item)
        
        # Plot schedule when outputs change
        self._plot = plot_pulse_schedule(self.instructions)
        self._plot_panel = widgets.HBox([self._plot, self._schedule_list, 
                                widgets.VBox([schedule_item_edit_btn,
                                              schedule_item_up_btn, 
                                              schedule_item_down_btn, 
                                              schedule_item_del_btn])]) # allow for extending

        schedule_editor = widgets.VBox([input_panel, 
                                        widgets.HBox([widgets.Label("Schedule:"), clear_btn, widgets.HBox(layout=widgets.Layout(flex='1 0 auto')),
                                                      widgets.HBox([append_to_dd, append_to_btn], layout=widgets.Layout(align_self='flex-end'))],
                                                    layout=widgets.Layout(justify_content='space-between')),
                                        self._plot_panel])
        super().__init__([schedule_editor], layout=widgets.Layout(width='900px'))

        self._editor = schedule_editor
        
    # Helper method to link PulseDesigner with ScheduleDesigner
    def update_custom_pulse(self, change):
        self.custom_pulse = change['new']
    
    def update(self):
        if hasattr(self, '_plot'):
            self._plot.update()