import numpy as np
import numpy.random as rn
from iznetwork import IzNetwork

class SmallWorldModularNetworkBuilder:
  """
  Builder of a small-world modular network that separates 
  the representation of inhibitory and excitatory neurons
  """

  def __init__(self, M, EN, IN):
    """
    Initialise a small-world modular network

    Inputs:
    M  -- Number of modules of excitatory neurons

    EN -- Number of excitatory neurons per module

    IN -- Number of inhibitory neurons
    """
    self._M = M
    self._EN = EN
    self._IN = IN
    self._N = M * EN


  def setExcitatoryParameters(self, a, b, c, d):
    """
    Set parameters for the excitatory neurons. Names are the the same as 
    in Izhikevich's original paper. All inputs must be np.arrays of size 
    (M * EN), where M is the number of modules and EN is number of excitatory 
    neurons per module
    """
    if (len(a), len(b), len(c), len(d)) != (self._N, self._N, self._N, self._N):
      raise Exception('Excitatory parameter vectors must be of size N.')
    
    self._exa = a
    self._exb = b
    self._exc = c
    self._exd = d


  def setInhibitoryParameters(self, a, b, c, d):
    """
    Set parameters for the inhibitory neurons. Names are the the same as 
    in Izhikevich's original paper. All inputs must be np.arrays of size 
    IN, where IN is the number of inhibitory neurons per module
    """
    if (len(a), len(b), len(c), len(d)) != (self._IN, self._IN, self._IN, self._IN):
      raise Exception('Inhibitory parameter vectors must be of size N.')
    
    self._ina = a
    self._inb = b
    self._inc = c
    self._ind = d


  def setExcitatoryToExcitatoryWeights(self, W):
    """
    Set the weights of all excitatory to excitatory neuron
    connections

    Inputs:
    W --  np.array or np.matrix of size (M * EN)-by-(M * EN)
          where M is number of modules and EN is number of 
          excitatory neurons per module
    """
    if W.shape != (self._N, self._N):
      raise Exception('Excitatory to excitatory Weight matrix must be (M * EN)-by-(M * EN).')
    self._exToExW = np.array(W)


  def setExcitatoryToInhibitoryWeights(self, W):
    """
    Set the weights of all excitatory to inhibitory neuron
    connections

    Inputs:
    W --  np.array or np.matrix of size (M * EN)-by-IN
          where M is number of modules, EN is number of excitatory 
          neurons per module, and IN is the number of inhibitory neurons
    """
    if W.shape != (self._N, self._IN):
      raise Exception('Excitatory to inhibitory weight matrix must be (M * EN)-by-IN.')
    self._exToInW = np.array(W)


  def setInhibitoryToExcitatoryWeights(self, W):
    """
    Set the weights of all inhibitory to excitatorty neuron
    connections

    Inputs:
    W --  np.array or np.matrix of size IN-by-(M * EN)
          where M is number of modules, EN is number of excitatory 
          neurons per module, and IN is the number of inhibitory neurons
    """
    if W.shape != (self._IN, self._N):
      raise Exception('Inhibitory to excitatory weight matrix must be IN-by-(M * EN).')
    self._inToExW = np.array(W)


  def setInhibitoryToInhibitoryWeights(self, W):
    """
    Set the weights of all inhibitory to inhibitory neuron
    connections

    Inputs:
    W --  np.array or np.matrix of size IN-by-IN
          where IN is the number of inhibitory neurons
    """
    if W.shape != (self._IN, self._IN):
      raise Exception('Inhibitory to inhibitory weight matrix must be IN-by-IN.')
    self._inToInW = np.array(W)


  def setExcitatoryToExcitatoryDelays(self, D):
    """
    Set the delays of all excitatory to excitatory neuron
    connections

    Inputs:
    D --  np.array or np.matrix of size (M * EN)-by-(M * EN)
          where M is number of modules and EN is number of 
          excitatory neurons per module
    """
    if D.shape != (self._N, self._N):
      raise Exception('Excitatory to excitatory delay matrix must be (M * EN)-by-(M * EN).')

    if not np.issubdtype(D.dtype, np.integer):
      raise Exception('Delays must be integer numbers.')

    if (D < 0.5).any():
      raise Exception('Delays must be strictly positive.')

    self._exToExD = D


  def setExcitatoryToInhibitoryDelays(self, D):
    """
    Set the delays of all excitatory to inhibitory neuron
    connections

    Inputs:
    D --  np.array or np.matrix of size (M * EN)-by-IN
          where M is number of modules, EN is number of excitatory 
          neurons per module, and IN is the number of inhibitory neurons
    """
    if D.shape != (self._N, self._IN):
      raise Exception('Excitatory to inhibitory delay matrix must be (M * EN)-by-IN.')

    if not np.issubdtype(D.dtype, np.integer):
      raise Exception('Delays must be integer numbers.')

    if (D < 0.5).any():
      raise Exception('Delays must be strictly positive.')

    self._exToInD = D


  def setInhibitoryToExcitatoryDelays(self, D):
    """
    Set the delays of all inhibitory to excitatorty neuron
    connections

    Inputs:
    D --  np.array or np.matrix of size IN-by-(M * EN)
          where M is number of modules, EN is number of excitatory 
          neurons per module, and IN is the number of inhibitory neurons
    """
    if D.shape != (self._IN, self._N):
      raise Exception('Inhibitory to excitatory delay matrix must be IN-by-(M * EN).')

    if not np.issubdtype(D.dtype, np.integer):
      raise Exception('Delays must be integer numbers.')

    if (D < 0.5).any():
      raise Exception('Delays must be strictly positive.')

    self._inToExD = D


  def setInhibitoryToInhibitoryDelays(self, D):
    """
    Set the delays of all inhibitory to inhibitory neuron
    connections

    Inputs:
    D --  np.array or np.matrix of size IN-by-IN
          where IN is the number of inhibitory neurons
    """
    if D.shape != (self._IN, self._IN):
      raise Exception('Inhibitory to inhibitory delay matrix must be IN-by-IN.')

    if not np.issubdtype(D.dtype, np.integer):
      raise Exception('Delays must be integer numbers.')

    if (D < 0.5).any():
      raise Exception('Delays must be strictly positive.')

    self._inToInD = D


  def buildAndRewireNetwork(self, p, Dmax):
    """
    Builds the network of Izhikevich neurons and with the excitatory
    neurons rewired with probability p

    Rewiring is only applied to the output IzNetwork (so does not change
    the previously set parameters of this builder class)

    Inputs:
    p --  Probability that an excitatory neuron will be
          rewired as an edge between communities

    Dmax  -- Maximum delay in all the synapses in the network, in ms. Any
             longer delay will result in failing to deliver spikes.

    Returns:
    IzNetwork -- A network with the set parameters
    """
    exToExW = self._exToExW.copy()

    # Rewire intra-community excitatory edges with probability p
    for m in range(self._M):
      for i in range(self._EN):
        for e in range(self._EN):
          if rn.rand() < p:
            targetM = (m + rn.randint(0, self._M)) % self._M # New module (excluding current)
            targetN = rn.randint(0, self._EN)

            # Compute current and target indices
            currIndex = (m * self._EN) + i
            oldTargetIndex = (m * self._EN) + e
            newTargetIndex = (targetM * self._EN) + targetN

            # Rewire
            exToExW[currIndex, newTargetIndex] = exToExW[currIndex, oldTargetIndex]
            exToExW[currIndex, oldTargetIndex] = 0

    # Combine weights and delays into single network as follows:
    #  | ExToEx ExToIn |
    #  | InToEx InToIn | 
    W = np.block([[self._exToExW, self._exToInW], [self._inToExW, self._inToInW]])
    D = np.block([[self._exToExD, self._exToInD], [self._inToExD, self._inToInD]])
    
    # Combine parameters
    a = np.vstack((self._exa, self._ina))
    b = np.vstack((self._exb, self._inb))
    c = np.vstack((self._exc, self._inc))
    d = np.vstack((self._exd, self._ind))

    # Build network
    network = IzNetwork((self._M * self._EN) + self._IN, Dmax)
    network.setWeights(W)
    network.setDelays(D)
    network.setParameters(a, b, c, d)

    return network
  


### Experiment-specific network construction

M = 8
EN = 100
IN = 200


## Excitatory to excitatory weights
exToExW = np.zeros(((M * EN), (M * EN)))
for m in range(M):
  connections = set()

  # 1000 random edges per module
  while len(connections) < 1000:
    base = rn.randint(0, 100)
    target = rn.randint(0, 100)

    # Prevent symmetric connections
    if base != target and (target, base) not in connections:
      connections.add((base, target))

  # Create edges with weight 1 and sf 17
  for (base, target) in connections:
    exToExW[(m * EN) + base, (m * EN) + target] = 17


## Excitatory to inhibitory weights
exToInW = np.zeros(((M * EN), IN))
for i in range(IN):

  # Focal - 4 excitatory all from same module
  baseM = rn.randint(0, 8)
  baseNs = rn.randint(0, 100, size=4)

  # Weight 0-1, sf of 50
  exToInW[(baseM * EN) + baseNs, i] = rn.uniform(0, 50)


## Inhibitory to excitatory weights
inToExW = rn.uniform(-2, 0, size=(IN, (M * EN)))


## Inhibitory to inhibitory weights
inToInW = rn.uniform(-1, 0, size=(IN, IN))


## Excitatory to excitatory delays (rand)
exToExD = rn.randint(1, 21, size=((M * EN), (M * EN)))


## Excitatory to inhibitory delays
exToInD = np.ones(((M * EN), IN))


## Inhibitory to excitatory weights
inToExD = np.ones((IN, (M * EN)))


## Inhibitory to inhibitory delays
inToInD = np.ones((IN, IN))


## Excitatory parameters
exa = 0.02 * np.ones(M * EN)
exb = 0.2 * np.ones(M * EN)
exc = -65 * np.ones(M * EN)
exd = 8 * np.ones(M * EN)


## Inhibitory parameters
ina = 0.02 * np.ones(IN)
inb = 0.25 * np.ones(IN)
inc = -65 * np.ones(IN)
ind = 2 * np.ones(IN)


## Construct network builder
builder = SmallWorldModularNetworkBuilder(M, EN, IN)
builder.setExcitatoryToExcitatoryWeights(exToExW)
builder.setExcitatoryToInhibitoryWeights(exToInW)
builder.setInhibitoryToExcitatoryWeights(inToExW)
builder.setInhibitoryToInhibitoryWeights(inToInW)
builder.setExcitatoryToExcitatoryDelays(exToExD)
builder.setExcitatoryToInhibitoryDelays(exToInD)
builder.setInhibitoryToExcitatoryDelays(inToExD)
builder.setInhibitoryToInhibitoryDelays(inToInD)
builder.setExcitatoryParameters(exa, exb, exc, exd)
builder.setExcitatoryParameters(ina, inb, inc, ind)


## Network with p=0.1
network1 = builder.buildAndRewireNetwork(0.1, 100)