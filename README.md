# NN

## With connections

  n0, i0
  n0, i1
  n1, i0
  n1, i1
  n1, i2
  n1, i3
  n2, i3
  n3, i2
  n3, i3

[
  [
    [0, 1],
    [0, 1, 2, 3],
    [3],
    [2, 3]
  ],
  [
    [0, 1],
    [0, 1, 2, 3],
  ]
]






[
  Nx.tensor([1,1]),
  Nx.tensor([1,1,1,1]),
  Nx.tensor([1]),
  Nx.tensor([1,1])
]


[                  # layers
    [              # layer_0
        [          # neuron_0
          x1,      # connection_0
          x2,      # connection_1
          ...
        ],
        [ ... ]    # neuron_1
    ],
    [ ... ]        # layer_1
]
