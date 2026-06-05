<h1>Don't try to make sense of what's here; these are just my chaotic experiments. </h1>
# NN

## TODO

- [x] Activation by layer
- [x] Map output. A function to map output to real results.
- [ ] Add weight decay (L2 regularization) | penalty = 0.001 * sum(weights^2) | cost = mse + penalty


## Notes

Too small → Underfitting (can't learn the pattern)
Too large → Overfitting (memorizes, doesn't generalize)
Just right → Good generalization

# For Hidden Layer Size (number of neurons)
## Heuristic #1: Between input and output size
Rule: hidden_size between input_size and output_size
Typical: (input_size + output_size) / 2 to (input_size + output_size)

## Heuristic #2: Based on training examples
Rule: hidden_size ≤ training_examples / (input_size + output_size)

## Heuristic #3: Problem complexity (most important)
Different problems need different capacities:
Problem	Decision boundary	Min hidden neurons
AND/OR	Linear (1 line)	0 (no hidden layer needed)
XOR	2 lines (1 corner)	2
3-input XOR	4 lines (2 corners?)	3-4
Circle pattern	Curved boundary	5-10
For Hidden Layer Count (number of layers)

# General rule:
Most problems: 1-2 hidden layers is enough
Very complex problems (image/speech): 3+ layers (deep learning)

When to add more layers:

    2 layers: When problem has hierarchical features (e.g., shapes → objects → faces)

    1 layer: For XOR, AND, OR, most simple classification

Why not always more layers?

    Harder to train (vanishing gradients)

    More prone to overfitting

    Slower computation

# Start with 1 hidden layer
hidden_size = (input_size + output_size) * 2/3

# Test and adjust
if underfitting (high training error):
    increase hidden_size or add layer

if overfitting (low training error, high test error):
    decrease hidden_size or add regularization

# Use known baselines
For standard problems, known architectures work:
Problem	Architecture
XOR (2-input)	2-2-1
3-input XOR	3-3-1 or 3-4-1
MNIST digits	784-128-64-10
Iris flowers	4-8-3
The XOR-specific logic

```elixir
def suggest_hidden_size(inputs, outputs, examples) do
  # Rule of thumb
  size1 = (inputs + outputs) * 2 / 3
  size2 = examples / (inputs + outputs)

  # Take average, round up
  max(round((size1 + size2) / 2), 2)
end
```


## Pure garbage
```
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


model
|> connect("input[0]", "layer[0][0]")
|> connect("input[1]", "layer[0][0]")
|> connect("input[0]", "layer[0][1]")
|> connect("input[1]", "layer[0][1]")
|> connect("layer[0][0]", "output[0]")
|> connect("layer[0][1]", "output[0]")



```






0  [0,0,0] → 0 → [0,0]
1  [0,0,1] → 1 → [0,1]
2  [0,1,0] → 1 → [0,1]
3  [0,1,1] → 2 → [1,0]
4  [1,0,0] → 1 → [0,1]
5  [1,0,1] → 2 → [1,0]
6  [1,1,0] → 2 → [1,0]
7  [1,1,1] → 3 → [1,1]


  3 _|                                    .
     |
     |
  2 _|               .          .     .
     |
     |
  1 _|     .     .         .
     |
     |
     |____________________________________________
           '     '    '    '    '     '    '    '
           1     2    3    4    5     6    7


inputs = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
outputs = [[0,0],[0,1],[0,1],[1,0],[0,1],[1,0],[1,0],[1,1]]



shape: {3,7,2}, eps: 0.01, rate: 0.1, act_func: :tanh
Iteration: 1000 - Cost: 0.014857055619359016
[
  {"input", "expected", "rounded out", "raw out"},
  {[0, 0, 0], [0, 0], [0, 0], [-0.001660881913267076, 0.0664716437458992]},
  {[0, 0, 1], [0, 1], [0, 1], [1.6644597053527832e-4, 0.8873333930969238]},
  {[0, 1, 0], [0, 1], [0, 1], [0.0013499698834493756, 0.9907898306846619]},
  {[0, 1, 1], [1, 0], [1, 0], [0.9757646322250366, 0.03246882185339928]},
  {[1, 0, 0], [0, 1], [0, 1], [0.0011084522120654583, 0.7835304141044617]},
  {[1, 0, 1], [1, 0], [1, 0], [0.9935481548309326, -0.014273389242589474]},
  {[1, 1, 0], [1, 0], [1, 0], [0.9775427579879761, 0.07937335222959518]},
  {[1, 1, 1], [1, 1], [1, 1], [0.9971345663070679, 0.8118522763252258]}
]


shape: {3,8,2}, eps: 0.01, rate: 0.1, act_func: :tanh
Iteration: 1000 - Cost: 0.0028977347537875175
[
  {"input", "expected", "rounded out", "raw out"},
  {[0, 0, 0], [0, 0], [0, 0], [-0.0015668781707063317, 5.911588086746633e-4]},
  {[0, 0, 1], [0, 1], [0, 1], [2.4975091218948364e-4, 0.9225526452064514]},
  {[0, 1, 0], [0, 1], [0, 1], [0.001324835349805653, 0.9779666662216187]},
  {[0, 1, 1], [1, 0], [1, 0], [0.9743190407752991, 0.017042208462953568]},
  {[1, 0, 0], [0, 1], [0, 1], [0.001040544011630118, 0.9557743072509766]},
  {[1, 0, 1], [1, 0], [1, 0], [0.992577314376831, 0.01221289113163948]},
  {[1, 1, 0], [1, 0], [1, 0], [0.9785308837890625, 0.010893122293055058]},
  {[1, 1, 1], [1, 1], [1, 1], [0.9962378144264221, 0.8894790410995483]}
]

shape: {3,9,2}, eps: 0.01, rate: 0.1, act_func: :tanh
Iteration: 1000 - Cost: 0.06315294653177261
[
  {"input", "expected", "rounded out", "raw out"},
  {[0, 0, 0], [0, 0], [0, 0], [-0.0012884729076176882, 0.08228442817926407]},
  {[0, 0, 1], [0, 1], [0, 1], [1.1223554611206055e-4, 0.796014130115509]},
  {[0, 1, 0], [0, 1], [0, 1], [0.001183330430649221, 0.6905784606933594]},
  {[0, 1, 1], [1, 0], [1, 0], [0.9798218011856079, 0.2578057646751404]},
  {[1, 0, 0], [0, 1], [0, 1], [0.0013806810602545738, 0.9775758385658264]},
  {[1, 0, 1], [1, 0], [1, 0], [0.9881641268730164, 0.08895070850849152]},
  {[1, 1, 0], [1, 0], [1, 0], [0.9757696390151978, 0.2618613541126251]},
  {[1, 1, 1], [1, 1], [1, 1], [0.9985682368278503, 0.6755024790763855]}
]


------------------------------------------
shape: {3,[6],2}, eps: 0.01, rate: 0.1, act_func: :tanh, output_act_func: :sigmoid
Iteration: 1000 - Cost: 0.12887956202030182
AVG ERROR: 0.05821955370811338
[
  {"input", "expected", "rounded out", "raw out"},
  {[1, 1, 1], [1, 1], [1, 0], [0.9868037700653076, 0.3284628093242645]},
  {[1, 1, 0], [1, 0], [1, 0], [0.9565959572792053, 0.2860051393508911]},
  {[1, 0, 1], [1, 0], [1, 0], [0.9569188952445984, 0.3277142643928528]},
  {[1, 0, 0], [0, 1], [0, 1], [0.004707690794020891, 0.7728898525238037]},
  {[0, 1, 1], [1, 0], [1, 0], [0.9579609036445618, 0.4767155647277832]},
  {[0, 1, 0], [0, 1], [0, 1], [0.0047858464531600475, 0.9427762627601624]},
  {[0, 0, 1], [0, 1], [0, 1], [0.004723916761577129, 0.942499577999115]},
  {[0, 0, 0], [0, 0], [0, 0], [9.03384352568537e-5, 0.013740736991167068]}
]
------------------------------------------


shape: {3,[8],2}, eps: 0.01, rate: 0.2, act_func: :tanh, output_act_func: :sigmoid
Iteration: 1000 - Cost: 0.007565782405436039
AVG ERROR: 0.003599793410444363
[
  {"input", "expected", "rounded out", "raw out"},
  {[1, 1, 1], [1, 1], [1, 1], [0.9999605417251587, 0.8416318893432617]},
  {[1, 1, 0], [1, 0], [1, 0], [0.973404049873352, 0.04058432951569557]},
  {[1, 0, 1], [1, 0], [1, 0], [0.9745596051216125, 0.022257059812545776]},
  {[1, 0, 0], [0, 1], [0, 1], [0.0015930724330246449, 0.8882744908332825]},
  {[0, 1, 1], [1, 0], [1, 0], [0.9801992177963257, 0.03148745372891426]},
  {[0, 1, 0], [0, 1], [0, 1], [0.0012637110194191337, 0.9285372495651245]},
  {[0, 0, 1], [0, 1], [0, 1], [0.0014661728637292981, 0.8999255895614624]},
  {[0, 0, 0], [0, 0], [0, 0], [1.2737310726151918e-6, 0.004999437369406223]}
]

------------------------------------------
shape: {3,[8],2}, eps: 0.01, rate: 0.2, act_func: :tanh, output_act_func: :tanh
Iteration: 1000 - Cost: 0.0010032568825408816

AVG ERROR: 4.91358207796111e-4
[
  {"input", "expected", "rounded out", "raw out"},
  {[1, 1, 1], [1, 1], [1, 1], [0.9977027773857117, 0.9378135204315186]},
  {[1, 1, 0], [1, 0], [1, 0], [0.9864873886108398, 0.004593131598085165]},
  {[1, 0, 1], [1, 0], [1, 0], [0.9954954981803894, 0.003261897247284651]},
  {[1, 0, 0], [0, 1], [0, 1], [3.8700547884218395e-4, 0.9649709463119507]},
  {[0, 1, 1], [1, 0], [1, 0], [0.9833454489707947, 0.003392161801457405]},
  {[0, 1, 0], [0, 1], [0, 1], [4.345066554378718e-4, 0.9912692904472351]},
  {[0, 0, 1], [0, 1], [0, 1], [-1.3308599591255188e-4, 0.9535260796546936]},
  {[0, 0, 0], [0, 0], [0, 0], [-6.408243207260966e-4, 0.001394554041326046]}
]
------------------------------------------

inputs = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
outputs = [[0,0,0,1],[0,0,1,0],[0,0,1,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[0,1,0,0],[1,0,0,0]]


shape: {3,[5],1}, eps: 0.001, rate: 0.5, act_func: :tanh, output_act_func: :sigmoid
Iteration: 1000 - Cost: 4.549452569335699e-6
AVG ERROR: 4.00898538588379e-6
[
  {"input", "expected", "rounded out", "raw out"},
  {[1, 1, 1], [4], [4], [0.9947529435157776]},
  {[1, 1, 0], [3], [3], [0.6596224904060364]},
  {[1, 0, 1], [3], [3], [0.6599181294441223]},
  {[1, 0, 0], [1], [1], [0.330190509557724]},
  {[0, 1, 1], [3], [3], [0.6598495244979858]},
  {[0, 1, 0], [1], [1], [0.32996949553489685]},
  {[0, 0, 1], [1], [1], [0.3303910195827484]},
  {[0, 0, 0], [0], [0], [0.002044085180386901]}
]


NEXT EXPERIMENTS
================

# Output layer: sigmoid (range [0,1] matches binary)
# Hidden layers: tanh (better gradients)

-----------------------------------------

# one output with 4 classes (softmax) → [0,1,2,3] as 4 categories

-----------------------------------------

3-6-2 with:
- Hidden: tanh
- Output: sigmoid (change your output activation)
- learning_rate: 0.1 for 2000 iterations


Problem 2: Majority Vote (5 inputs)
===================================

Input (5 bits) → Output (1 bit)
[0,0,0,0,0] → 0
[0,0,0,0,1] → 0
[0,0,0,1,1] → 0
[0,0,1,1,1] → 1
[0,1,1,1,1] → 1
[1,1,1,1,1] → 1
[1,0,0,0,0] → 0
[1,1,0,0,0] → 0
[1,1,1,0,0] → 1
[1,0,1,0,1] → 1

inputs = [[0,0,0,0,0],[0,0,0,0,1],[0,0,0,1,1],[0,0,1,1,1],[0,1,1,1,1],[1,1,1,1,1],[1,0,0,0,0],[1,1,0,0,0],[1,1,1,0,0],[1,0,1,0,1]]
outputs = [[0],[0],[0],[1],[1],[1],[0],[0],[1],[1]]
outputs = [[-1],[-1],[-1],[1],[1],[1],[-1],[-1],[1],[1]]


shape: {5,3,1}, eps: 0.01, rate: 0.5, act_func: :tanh, output_act_func: :sigmoid
outputs = [[0],[0],[0],[1],[1],[1],[0],[0],[1],[1]]
Iteration: 1000 - Cost: 5.426945164799691e-4

AVG ERROR: 5.405313451774419e-4
[
  {"input", "expected", "rounded out", "raw out"},
  {[1, 0, 1, 0, 1], [1], [1], [0.9710056185722351]},
  {[1, 1, 1, 0, 0], [1], [1], [0.9762733578681946]},
  {[1, 1, 0, 0, 0], [0], [0], [0.020968714728951454]},
  {[1, 0, 0, 0, 0], [0], [0], [0.021344909444451332]},
  {[1, 1, 1, 1, 1], [1], [1], [0.9764326214790344]},
  {[0, 1, 1, 1, 1], [1], [1], [0.9740235209465027]},
  {[0, 0, 1, 1, 1], [1], [1], [0.9792242050170898]},
  {[0, 0, 0, 1, 1], [0], [0], [0.02281954698264599]},
  {[0, 0, 0, 0, 1], [0], [0], [0.021976429969072342]},
  {[0, 0, 0, 0, 0], [0], [0], [0.020996807143092155]}
]

-----------------------------------------
shape: {5,3,1}, eps: 0.01, rate: 0.5, act_func: :tanh, output_act_func: :sigmoid
outputs = [[-1],[-1],[-1],[1],[1],[1],[-1],[-1],[1],[1]]
Iteration: 1000 - Cost: 2.6628762134350838e-5

AVG ERROR: 2.6602804518915944e-5
[
  {"input", "expected", "rounded out", "raw out"},
  {[1, 0, 1, 0, 1], [1], [1], [0.9916399717330933]},
  {[1, 1, 1, 0, 0], [1], [1], [0.9952752590179443]},
  {[1, 1, 0, 0, 0], [0], [0], [-0.9924569725990295]},
  {[1, 0, 0, 0, 0], [0], [0], [-0.997585654258728]},
  {[1, 1, 1, 1, 1], [1], [1], [0.9965998530387878]},
  {[0, 1, 1, 1, 1], [1], [1], [0.9967386722564697]},
  {[0, 0, 1, 1, 1], [1], [1], [0.996033251285553]},
  {[0, 0, 0, 1, 1], [0], [0], [-0.9925805926322937]},
  {[0, 0, 0, 0, 1], [0], [0], [-0.9962409734725952]},
  {[0, 0, 0, 0, 0], [0], [0], [-0.9980054497718811]}
]


t = [0,1]
inputs = for a <- t, b <- t, c <- t, d <- t, e <- t, do: [a,b,c,d,e]
outputs = inputs |> Enum.map(&Enum.sum/1) |> Enum.map(&(&1 >= 3 && [1] || [-1]))

outputs = outputs |> Enum.map(fn [v] -> v == -1 && [0] || [1] end)
outputs = outputs |> Enum.map(fn [v] -> v == 0 && [-1] || [1] end)

inputs = inputs |> Enum.shuffle() |> Enum.take(16)
outputs = inputs |> Enum.map(&Enum.sum/1) |> Enum.map(&(&1 >= 3 && [1] || [-1]))

[ 
  [ // first_layers_list
    [ // grads_list
      {ws, b},
      {ws, b},
      {ws, b},
       ...
    ],
    [ ... ]
  ]
  ,
  [ // layers_list
    [ // grads_list
      {ws, b},
      {ws, b},
      {ws, b},
       ...
    ],
    [ ... ]
  ]
  ,
  ...
]

[0,0] → 0
[0,1] → 0
[1,0] → 0
[1,1] → 1

inputs = [[0,0],[0,1],[1,0],[1,1]]
outputs = [[0],[0],[0],[1]]


target: #Nx.Tensor<
  f32[3]
  [0.0, 1.0, 0.0]
>
target: #Nx.Tensor<
  f32[3]
  [0.0, 1.0, 0.0]
>
target: #Nx.Tensor<
  f32[3]
  [0.0, 0.0, 1.0]
>
target: #Nx.Tensor<
  f32[3]
  [0.0, 0.0, 1.0]
>
target: #Nx.Tensor<
  f32[3]
  [0.0, 0.0, 1.0]
>

a: #Nx.Tensor<
  f32[3]
  [0.58726996, 0.33228046, 0.08044958]
>
a: #Nx.Tensor<
  f32[3]
  [0.5689425, 0.35199645, 0.07906104]
>
a: #Nx.Tensor<
  f32[3]
  [0.60393524, 0.3280162, 0.06804856]
>
a: #Nx.Tensor<
  f32[3]
  [0.59608454, 0.3346589, 0.06925657]
>
a: #Nx.Tensor<
  f32[3]
  [0.62234396, 0.31381255, 0.063843444]
>

