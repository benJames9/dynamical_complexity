import numpy as np
import numpy.random as rn
from iznetwork import IzNetwork
import matplotlib.pyplot as plt

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
        for j in range(self._EN):
          currIndex = (m * self._EN) + i
          oldTargetIndex = (m * self._EN) + j

          # Check that there is an edge to rewire
          if exToExW[currIndex, oldTargetIndex] == 0:
            continue

          if rn.rand() < p:
            targetM = (m + rn.randint(0, self._M)) % self._M # New module (excluding current)
            targetN = rn.randint(0, self._EN)

            newTargetIndex = (targetM * self._EN) + targetN

            # Rewire
            exToExW[currIndex, newTargetIndex] = exToExW[currIndex, oldTargetIndex]
            exToExW[currIndex, oldTargetIndex] = 0

    # Combine weights and delays into single network as follows:
    #  | ExToEx ExToIn |
    #  | InToEx InToIn | 
    W = np.block([[exToExW, self._exToInW], [self._inToExW, self._inToInW]])
    D = np.block([[self._exToExD, self._exToInD], [self._inToExD, self._inToInD]])
    
    # Combine parameters
    a = np.concatenate((self._exa, self._ina))
    b = np.concatenate((self._exb, self._inb))
    c = np.concatenate((self._exc, self._inc))
    d = np.concatenate((self._exd, self._ind))

    # Build network
    network = IzNetwork((self._M * self._EN) + self._IN, Dmax)
    network.setWeights(W)
    network.setDelays(D)
    network.setParameters(a, b, c, d)

    self.generate_matrix_connectivity_plot(p, exToExW)

    return network
  
  def generate_matrix_connectivity_plot(self, p, weights):
    """
    Generate a plot of the connection matrix

    Inputs:
    p       -- Rewiring probability
    weights -- Connection matrix
    """
    plt.figure(figsize=(12, 4))
    plt.imshow(weights, cmap="hot")
    plt.title(f"Connection Matrix, p={p}")
    plt.colorbar(label="Connection weight")
    # plt.savefig(f"img/connectivity_p_{p}.svg")
    plt.show()


def generate_plots(network, p):
  """
  Generate a raster plot and mean firing rate plot of the network

  Inputs:
  network -- IzNetwork object
  p       -- Rewiring probability
  """
  milliseconds = 1000
  firing_counts = np.zeros((M * EN, milliseconds))

  # Run simulation for 1000 ms
  for t in range(milliseconds):
    poisson_spikes = rn.poisson(0.01, network._N)
    extra_current = 15.0 * (poisson_spikes > 0)
    network.setCurrent(extra_current)

    fired_neurons = network.update()

    for n in fired_neurons:
      if n < M * EN:
        firing_counts[n, t] = 1
    
  generate_raster_plot(firing_counts)
  generate_mean_firing_rate_plot(firing_counts, p)


def generate_raster_plot(firings):
  """
  Generate a raster plot of the network

  Inputs:
  firings -- List of tuples (time, neuron index) of when a neuron fired
  """
  plt.figure(figsize=(12, 4))
  neurons, times = np.nonzero(firings)
  plt.scatter(times, neurons, c="blue")
  plt.title(f"Neuron Firings, p={p}")
  plt.xlabel("Time (ms)")
  plt.ylabel("Neuron index")
  # plt.savefig(f"img/firing_p_{p}.svg")
  plt.show()


def generate_mean_firing_rate_plot(firing_counts, p):
  """
  Generate a mean firing rate plot of the network

  Inputs:
  firing_counts -- 2D np.array of size (M * EN)-by-1000
                   where M is number of modules and EN is number of 
                   excitatory neurons per module
  p             -- Rewiring probability
  """
  window_size = 50
  shift = 20
  module_firing_rates = []

  for module in range(M):
    module_indices = np.arange(module * EN, (module + 1) * EN)
    module_firing_rate = []
    for t in range(0, 1000 - window_size + 1, shift):
      window = firing_counts[module_indices, t:t + window_size]
      module_firing_rate.append(np.count_nonzero(window) / window_size)
    module_firing_rates.append(module_firing_rate)

  # Plot mean firing rates
  time_periods = np.arange(0, 1000 - window_size + 1, shift)

  plt.figure(figsize=(12, 4))
  for module, rates in enumerate(module_firing_rates):
      plt.plot(time_periods, rates, label=f"Module {module}")
  plt.title(f"Mean Firing Rates, p={p}")
  plt.xlabel("Time (ms)")
  plt.ylabel("Mean Firing Rate")
  plt.legend()
  # plt.savefig(f"img/mean_p_{p}.svg")
  plt.show()

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
    source = rn.randint(0, EN)
    target = rn.randint(0, EN)

    # Prevent symmetric connections
    if source != target and (target, source) not in connections:
      connections.add((source, target))

  # Create edges with weight 1 and sf 17
  for (source, target) in connections:
    exToExW[(m * EN) + source, (m * EN) + target] = 17


## Excitatory to inhibitory weights
exToInW = np.zeros(((M * EN), IN))
connected_inhibitory = set()
for i in range(0, M * EN - 4, 4):
  targetN = rn.randint(0, IN)
  while targetN in connected_inhibitory:
    targetN = rn.randint(0, IN)

  connected_inhibitory.add(targetN)

  # Weight 0-1, sf of 50
  exToInW[i:i+4, targetN] = rn.uniform(0, 50)


## Inhibitory to excitatory weights
inToExW = rn.uniform(-2, 0, size=(IN, (M * EN)))

## Inhibitory to inhibitory weights
inToInW = rn.uniform(-1, 0, size=(IN, IN))
for i in range(IN):
  inToInW[i, i] = 0

## Excitatory to excitatory delays (rand)
exToExD = rn.randint(1, 21, size=((M * EN), (M * EN)))

## Excitatory to inhibitory delays
exToInD = np.ones(((M * EN), IN), dtype=np.int32)

## Inhibitory to excitatory weights
inToExD = np.ones((IN, (M * EN)), dtype=np.int32)

## Inhibitory to inhibitory delays
inToInD = np.ones((IN, IN), dtype=np.int32)

## Excitatory parameters
exa = 0.02 * np.ones(M * EN, dtype=np.int32)
exb = 0.2 * np.ones(M * EN, dtype=np.int32)
exc = -65 * np.ones(M * EN, dtype=np.int32)
exd = 8 * np.ones(M * EN, dtype=np.int32)

## Inhibitory parameters
ina = 0.02 * np.ones(IN, dtype=np.int32)
inb = 0.25 * np.ones(IN, dtype=np.int32)
inc = -65 * np.ones(IN, dtype=np.int32)
ind = 2 * np.ones(IN, dtype=np.int32)

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
builder.setInhibitoryParameters(ina, inb, inc, ind)

## Generate plots for different rewiring probabilities
ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
for p in ps:
  network = builder.buildAndRewireNetwork(p, 20)
  generate_plots(network, p)
