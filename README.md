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


{features, labels} = Scidata.Iris.download()
indices = 0..149 |> Enum.shuffle()
train_indices = Enum.take(indices, 120)
test_indices = Enum.drop(indices, 120)


## separo train of test
x_train = train_indices |> Enum.map(fn i -> Enum.at(features, i) end)
x_test = test_indices |> Enum.map(fn i -> Enum.at(features, i) end)
y_train = train_indices |> Enum.map(fn i -> Enum.at(labels, i) end)
y_test = test_indices |> Enum.map(fn i -> Enum.at(labels, i) end)

## normalize inputs
max = Nx.concatenate([Nx.tensor(x_train), Nx.tensor(x_test)]) |> Nx.reduce_max(axes: [0])
min = Nx.concatenate([Nx.tensor(x_train), Nx.tensor(x_test)]) |> Nx.reduce_min(axes: [0])

x_train = x_train |> Enum.map(fn t -> Nx.divide(Nx.subtract(Nx.tensor(t), min), Nx.subtract(max, min)) end)
x_test = x_test |> Enum.map(fn t -> Nx.divide(Nx.subtract(Nx.tensor(t), min), Nx.subtract(max, min)) end)


## convert outputs for softmax
y_train = y_train |> Enum.map(fn n -> Integer.to_string(n,2) |> String.pad_leading(3, "0") |> String.split("", trim: true) |> Enum.map(&String.to_integer/1) end)
y_test = y_test |> Enum.map(fn n -> Integer.to_string(n,2) |> String.pad_leading(3, "0") |> String.split("", trim: true) |> Enum.map(&String.to_integer/1) end)

## convert outputs for sigmoid
y_train = y_train |> Enum.map(fn n -> [n/3+1/6] end)
y_test = y_test |> Enum.map(fn n -> [n/3+1/6] end)

## convert outputs for tanh
y_train = y_train |> Enum.map(fn n -> [2*(n/3+1/6)-1] end)
y_test = y_test |> Enum.map(fn n -> [2*(n/3+1/6)-1] end)

## model sigmoid
model = NN4.build_model({4,12,1}, rate: 0.3, act_func: :tanh, output_act_func: :sigmoid, batch_mode: true, map_output_func: fn n -> trunc(n*3) end)

## model tanh
model = NN4.build_model({4,12,1}, rate: 0.3, act_func: :tanh, output_act_func: :tanh, batch_mode: true, map_output_func: fn n -> trunc(3*(n+1)/2) end)

model2 = NN4.train_model(model, x_train, y_train, 1000)
NN4.test(model2, x_test, y_test)



Test 1:
Iteration: 2000 - Cost: 0.1876574675242106
HITS: 90.0% - AVG COST: 0.05579695788263861

Test 2:
Iteration: 2000 - Cost: 0.16073145866394042
HITS: 96.66666666666667% - AVG COST: 0.049029032673246965

Test 3:
Iteration: 2000 - Cost: 0.1765014330546061
HITS: 96.66666666666667% - AVG COST: 0.049482178071048113

## NN4 (tensors versions)

[
  %Neurons{
    weights: Nx.tensor(
        [w,w,w,w,w,w,w...,w],    \
        [w,w,w,w,w,w,w...,w],     |
        [w,w,w,w,w,w,w...,w],     |
        [w,w,w,w,w,w,w...,w],     |
        [w,w,w,w,w,w,w...,w],     |- N neurons weigths
        [w,w,w,w,w,w,w...,w],     |
        [w,w,w,w,w,w,w...,w],     |
        ...                       |
        [w,w,w,w,w,w,w...,w]     /
    ),
    biases: Nx.tensor(
        [b,b,b,b,b,b,..,b]   ---> N biases
    ),
    connections: [
        [0,1,2,3...,X],    \
        [0,1,2,3...,X],     |
        [0,1,2,3...,X],     |
        [0,1,2,3...,X],     |
        [0,1,2,3...,X],     |- N connections
        [0,1,2,3...,X],     |
        [0,1,2,3...,X],     |
        ...                 |
        [0,1,2,3...,X]     /
    ]
  }
]

l = [[1,1,1,1,1], [2,2,2,2,2], [3,3,3,3,3], [4,4,4,4,4], [5,5,5,5,5], [6,6,6,6,6]]
t = Nx.tensor(l)

inputs = [[1,0], [0,1], [0,0], [1,1]]
outputs = [[0], [0], [0], [1]]

m1 = NN4.build_model({2,2,1}, rate: 0.5, act_func: :tanh, output_act_func: :sigmoid)
m2 = NN4.train_model(m1, inputs, outputs, 100, rate: 0.2)


model = NN4.build_model({2,2,1}, rate: 0.5, act_func: :tanh, output_act_func: :sigmoid)
mm = NN3.build_model({2,2,1}, rate: 0.5, act_func: :tanh)

mm2 = NN3.train_model(mm, 100, inputs, outputs, rate: 0.2)
model2 = NN4.train_model(model, inputs, outputs, 100, rate: 0.2)

NN3.train_model(mm, 1, inputs, outputs, rate: 0.2)
NN4.train_model(model, inputs, outputs, 1, rate: 0.2)


model = NN4.build_model({2,2,1}, rate: 0.5, act_func: :tanh, output_act_func: :sigmoid)
model2 = NN4.train_model(model, inputs, outputs, 100, rate: 0.2)


## MINIST

{train_images, train_labels} = Scidata.MNIST.download()

{images_binary, images_type, images_shape} = train_images
{labels_binary, labels_type, labels_shape} = train_labels

x_train = images_binary |> Nx.from_binary(images_type) |> Nx.reshape({60000,784}) |> Nx.to_batched(10000) |> Enum.at(0) |> Nx.to_list() |> Enum.take(1) |> Enum.map(&Nx.tensor/1)

y_train = labels_binary |> :binary.bin_to_list() |> Enum.take(10000) |> Enum.map(fn n -> Integer.to_string(n,2) |> String.pad_leading(10, "0") |> String.split("", trim: true) |> Enum.map(&String.to_integer/1) end)  |> Enum.take(1)  |> Enum.map(&Nx.tensor/1)


x_test = images_binary |> Nx.from_binary(images_type) |> Nx.reshape({60000,784}) |> Nx.to_batched(10000) |> Enum.at(1) |> Nx.to_list() |> Enum.take(1000) |> Enum.map(&Nx.tensor/1)

y_test = labels_binary |> :binary.bin_to_list() |> Enum.drop(10000) |> Enum.take(1000) |> Enum.map(fn n -> Integer.to_string(n,2) |> String.pad_leading(10, "0") |> String.split("", trim: true) |> Enum.map(&String.to_integer/1) end)  |> Enum.map(&Nx.tensor/1)


mapf = fn o -> o |> Enum.map(&Integer.to_string/1) |> Enum.join("") |> String.to_integer(2) end
model = NN5.build_model({784, [128, 64], 10}, act_func: :tanh, output_act_func: :sigmoid, map_output_func: mapf)

model2 = NN4.train_model(model, x_train, y_train, 1)

## MINIST BATCH

model = NN5.build_model({784, [128, 64], 10}, act_func: :tanh, output_act_func: :sigmoid, map_output_func: mapf)

{train_images, train_labels} = Scidata.MNIST.download()

### Normalize and batch images
{images_binary, images_type, images_shape} = train_images

batched_images =  images_binary |> Nx.from_binary(images_type) |> Nx.reshape({60000,784}) |> Nx.divide(255) |> Nx.to_batched(10000) |> Enum.at(0) |> Nx.to_batched(32)

### One-hot-encode and batch labels
{labels_binary, labels_type, _shape} = train_labels

batched_labels = labels_binary |> Nx.from_binary(labels_type) |> Nx.new_axis(-1) |> Nx.equal(Nx.tensor(Enum.to_list(0..9))) |> Nx.to_batched(10000) |> Enum.at(0) |> Nx.to_batched(32)

### For test

test_images =  images_binary |> Nx.from_binary(images_type) |> Nx.reshape({60000,784}) |> Nx.divide(255) |> Nx.to_batched(1000) |> Enum.at(10) |> Nx.to_batched(32)

test_labels = labels_binary |> Nx.from_binary(labels_type) |> Nx.new_axis(-1) |> Nx.equal(Nx.tensor(Enum.to_list(0..9))) |> Nx.to_batched(1000) |> Enum.at(10) |> Nx.to_batched(32)


Ok, now the important question:
Suppose you have a model with shape 4-3-1 (just an example). So you will have the train datas separated in N batchs of size X:
Suppose X = 6 and this is the first batch:
[
  [0,1,0,1], # Input1
  [0,0,1,0], # Input2
  [1,0,0,0], # Input3
  [0,1,1,0], # Input4
  [0,0,1,1], # Input5
  [0,1,0,0]  # Input6
]

And then you have in the hidden layer 3 neurons, everyone with 4 weights:
[
  [w1, w2, w3, w4], # Neuron1
  [w1, w2, w3, w4], # Neuron2
  [w1, w2, w3, w4]  # Neuron3
]

In my implementation, when I do the forward I take the first input of the train batch, multply by the weights and sum the bias. I repeat the same althroug the rest of the layers until the output. Then take the second input and I do the forward for it, and so on.

There is using Nx a way to make all the inputs of a batch a operate agains weights at the same time?


Ok. Now I can do the forward, but I have a question about backward.
In the case of the MINIST, when I calculate the deltas_outputs to start backward, now I get a batch of 32 deltas (shape {32,10}), one for final outputs of the batch of 32 inputs.

You said that the formule to calculate the next deltas for the next hidden layer, I must do:

W2^T · δ_output * derivative(z_hidden)

First question: W2 are the weights of the output layer or the next hidden layer?

Second question: the δ_output now are the deltas of the batch of 32 inputs, right?

Third question: if W2 are the weights of the output layer and I use it to calculate the δ_hidden, W2 have shape {10,64}?

Now, after the forward backward of the first batch of 32 inputs, I got 3 pairs of gradients (one for weights and one for biases). Each of the gradients for weights has respectively these shapes:
{128, 784}, {64,128}, {10,64}. Is that right?

What should I do with these gradients? How I apply them to the layers to learn?

-------------------

I need to understand well something.
0. Model 784-128-64-10
1. I have a batch of 32 inputs, a tensor shape {32, 784}
2. I make the forward of each 32 inputs and get a batch of 32 forward path (I get it in reverse
   for better process in the backward):
   {
     {z_tensor{32,10}, a_tensor{32, 10}},
     {z_tensor{32,64}, a_tensor{32, 64}},
     {z_tensor{32,128}, a_tensor{32, 128}}
   }
Is this ok until here?

-- 

When I start to do the backward, first I calculate the delta_output. For that (gessing sigmoid) I do the following:
```
  # tt_output is the expected output of the bach of 32 inputs
  t1 = Nx.subtract(a_tensor{32, 10}, tt_output)
  t2 = derivative(a_tensor{32, 10})
  delta_output = Nx.multiply(t1, t2)
```

delta_output is a tensor{32, 10}

is this ok?

--

Now I start the backward. I am doing this: 
```
  t1 = Nx.transpose(delta_output) # t1 is {10, 32}
  gradients_output = Nx.dot(t1, a_tensor{32, 64})
```

Is this ok? 

-- 

Now I calculate: 
```
derivative_z = derivative(a_tensor{32,64}) # hidden layer 64 neurons

```

and then the deltas of the this hiddel layer:
```
t1 = Nx.dot(delta_output{32, 10}, output_layer.weights{10, 64}) # t1 is {32,64}
delta_hidden64 = Nx.multiply(t1{32,64}, derivative_z{32,64})
```

delta_hidden64 is {32,64}

Is this right?


