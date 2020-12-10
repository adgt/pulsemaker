import ipywidgets as widgets
import qonduit.visualization.circuit as cvis
import qonduit.visualization.state as svis
import qiskit.quantum_info as qi

class Simple:
    def __init__(self, circuit):
        def add_gate(b):
            if b.description == 'H0':
                circuit.h(0)
            if b.description == 'H1':
                circuit.h(1)
            if b.description == 'X0':
                circuit.x(0)
            if b.description == 'X1':
                circuit.x(1)
            self.update()
        
        gates = ['H0', 'H1', 'X0', 'X1']
        buttons = [widgets.Button(description=g) for g in gates]
        for b in buttons:
            b.on_click(add_gate)
                   
        cv = cvis.draw(circuit)
        state = qi.Statevector.from_instruction(circuit.to_instruction())        
        sv = svis.plot_bloch_multivector(state)
                
        tab = widgets.Tab()
        tab.children = [widgets.VBox([widgets.HBox(buttons), cv, sv])]
        tab.set_title(0, 'Circuit Dashboard')
        
        self._circuit = circuit
        self._tab = tab
        self._cv = cv
        self._sv = sv
    
    def update(self):
        self._cv.update()
        self._sv.kwargs_widgets[0].value = qi.Statevector.from_instruction(self._circuit.to_instruction())        
    
    def _ipython_display_(self):        
        display(self._tab)   
        