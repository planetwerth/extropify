class ComplexSystem:
    def __init__(self, energy, extropy, entropy, innovation_rate=0.1, ethical_factor=1.0):
        """
        Initializes a complex system with core properties such as energy, extropy (organization potential), 
        and entropy (disorder). The system also has parameters for innovation and ethical influence.
        
        :param energy: The available energy within the system.
        :param extropy: The measure of the system's potential for growth and organized development.
        :param entropy: The measure of disorder and decay within the system.
        :param innovation_rate: Rate at which external innovation influences the system's energy and extropy.
        :param ethical_factor: The degree to which ethical decisions influence the system's extropy.
        """
        self.energy = energy
        self.extropy = extropy
        self.entropy = entropy
        self.innovation_rate = innovation_rate
        self.ethical_factor = ethical_factor
        self.extropic_flux = 0
        self.system_state = "Stable"
        self.external_innovation = 0
        self.ethical_intervention = 0

    def compute_extropy(self):
        """
        Calculates the system's extropy based on the balance between energy, entropy, and feedback loops. 
        This dynamic calculation reflects how order (extropy) and disorder (entropy) interact, influenced 
        by both internal forces and external ethical interventions.
        """
        self.extropic_flux = -self.compute_energy_gradient() + self.entropy_feedback()
        self.extropy += self.extropic_flux + self.ethical_intervention
        return self.extropy

    def compute_energy_gradient(self):
        """
        Models how energy is distributed across the system, considering the spatial dynamics 
        and subsystem interactions, with a nonlinear relationship to entropy.
        """
        return self.energy * (1 - self.entropy / (self.entropy + 1))

    def entropy_feedback(self):
        """
        Computes the feedback of entropy into the system. This feedback dampens entropy when the system's 
        extropy is high, supporting growth and stability in the face of disorder.
        """
        return -self.entropy * (1 - self.extropy / (self.extropy + 1))

    def apply_external_innovation(self, external_energy):
        """
        Introduces external innovation or energy into the system. The effect of external inputs scales 
        with the system's current state of extropy, amplifying or moderating the impact based on its level.
        """
        innovation_effect = self.innovation_rate * external_energy * (self.extropy / (self.extropy + 1))
        self.external_innovation = innovation_effect
        self.energy += innovation_effect

    def apply_ethical_intervention(self, ethical_choice):
        """
        Ethical decisions influence the system's extropy in an ongoing, non-binary manner. 
        Interventions act as continuous feedback, promoting growth in alignment with ethical values.
        """
        if ethical_choice:
            self.ethical_intervention = self.ethical_factor * (self.extropy / (self.extropy + 1))
        else:
            self.ethical_intervention = 0

    def simulate_cycle(self):
        """
        Simulates the system's cyclical evolution, taking into account the dynamic balance between extropy, 
        entropy, and external factors. This method determines whether the system is growing, stable, or collapsing.
        """
        self.compute_extropy()
        if self.extropy > self.entropy:
            self.system_state = "Growing"
        else:
            self.system_state = "Collapsing"
            self.entropy += 0.05  # Gradual entropy increase in collapse

    def stabilize_system(self):
        """
        Stabilizes the system during periods of collapse through ethical interventions. When extropy falls 
        below entropy, external innovation can restore balance and support continued growth.
        """
        if self.system_state == "Collapsing":
            self.apply_ethical_intervention(True)
            self.system_state = "Stable"
        else:
            self.apply_external_innovation(20)  # External innovation ensures continued growth
