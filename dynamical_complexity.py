import numpy as np

class SmallWorldModularNetwork:
  """
  Representation of a single small-world modular 
  network that seperates the representation of inhibitory
  and excitatory neurons
  """

  def __init__(self, M, EN, IN):
    """
    Initialise a small-world modular network

    Inputs:
    M  -- Number of modules of excitatory neurons

    EN -- Number of excitatory neurons per module

    IN -- Number of inhibitory neurons
    """
    pass
  
  def rewire(self, p):
    """
    Rewire the modular network

    Inputs:
    p --  Probability that an excitatory neuron will be
          rewired as an edge between communities
    """
    pass

  def setExcitatoryParameters(self, a, b, c, d):
    """
    Set parameters for the excitatory neurons. Names are the the same as 
    in Izhikevich's original paper. All inputs must be np.arrays of size 
    (M * EN), where M is the number of modules and EN is number of excitatory 
    neurons per module
    """
    pass

  def setInhibitoryParameters(self, a, b, c, d):
    """
    Set parameters for the inhibitory neurons. Names are the the same as 
    in Izhikevich's original paper. All inputs must be np.arrays of size 
    IN, where IN is the number of inhibitory neurons per module
    """
    pass

  def setExcitatoryToExcitatoryWeights(self, W):
    """
    Set the weights of all excitatory to excitatory neuron
    connections

    Inputs:
    W --  np.array or np.matrix of size (M * EN)-by-(M * EN)
          where M is number of modules and EN is number of 
          excitatory neurons per module
    """
    pass

  def setExcitatoryToInhibitoryWeights(self, W):
    """
    Set the weights of all excitatory to inhibitory neuron
    connections

    Inputs:
    W --  np.array or np.matrix of size (M * EN)-by-IN
          where M is number of modules, EN is number of excitatory 
          neurons per module, and IN is the number of inhibitory neurons
    """
    pass

  def setInhibitoryToExcitatoryWeights(self, W):
    """
    Set the weights of all inhibitory to excitatorty neuron
    connections

    Inputs:
    W --  np.array or np.matrix of size IN-by-(M * EN)
          where M is number of modules, EN is number of excitatory 
          neurons per module, and IN is the number of inhibitory neurons
    """
    pass

  def setInhibitoryToInhibitoryWeights(self, W):
    """
    Set the weights of all inhibitory to inhibitory neuron
    connections

    Inputs:
    W --  np.array or np.matrix of size IN-by-IN
          where IN is the number of inhibitory neurons
    """
    pass

  def setExcitatoryToExcitatoryDelays(self, D):
    """
    Set the delays of all excitatory to excitatory neuron
    connections

    Inputs:
    D --  np.array or np.matrix of size (M * EN)-by-(M * EN)
          where M is number of modules and EN is number of 
          excitatory neurons per module
    """
    pass

  def setExcitatoryToInhibitoryDelays(self, D):
    """
    Set the delays of all excitatory to inhibitory neuron
    connections

    Inputs:
    D --  np.array or np.matrix of size (M * EN)-by-IN
          where M is number of modules, EN is number of excitatory 
          neurons per module, and IN is the number of inhibitory neurons
    """
    pass

  def setInhibitoryToExcitatoryDelays(self, D):
    """
    Set the delays of all inhibitory to excitatorty neuron
    connections

    Inputs:
    D --  np.array or np.matrix of size IN-by-(M * EN)
          where M is number of modules, EN is number of excitatory 
          neurons per module, and IN is the number of inhibitory neurons
    """
    pass

  def setInhibitoryToInhibitoryDelays(self, D):
    """
    Set the delays of all inhibitory to inhibitory neuron
    connections

    Inputs:
    D --  np.array or np.matrix of size IN-by-IN
          where IN is the number of inhibitory neurons
    """
    pass

  def setExcitatoryCurrent(self, I):
    """
    Set any additional background current for excitatory neurons

    Inputs:
    I --  np.array of size (M * EN) with each element representing 
          additional current to that neuron for the NEXT update only
    """
    pass

  def setInhibitoryCurrent(self, I):
    """
    Set any additional background current for inhibitory neurons

    Inputs:
    I --  np.array of size IN with each element representing additional 
          current to that neuron for the NEXT update only
    """
    pass

  def update(self):
    """
    Simulate one millisecond of network activity (Euler method used)

    Returns the indices of the excitatory neurons that fired this 
    millisecond. Neurons are ordered so that consecutive indices
    represent modules.
    """
    pass
