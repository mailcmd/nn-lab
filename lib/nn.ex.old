defmodule NN do
  defmodule Model do
	  defstruct [
      layers: nil,
      eps: nil,
      rate: nil,
      act_func: nil
    ]
  end

  defmodule Layer do
	  defstruct [
      weights: nil,
      biases: nil
    ]
  end

  defmodule Activation do
    import Nx.Defn
    
    defn softmax(t) do
      shifted = t - Nx.reduce_max(t, axes: [-1], keep_axes: true)
      exps = Nx.exp(shifted)
      exps / Nx.sum(exps, axes: [-1], keep_axes: true)
    end

    defn relu(t) do
      Nx.max(0, t)
    end

    defn sigmoid(t) do
      Nx.sigmoid(t)
    end

    defn tanh(t) do
      Nx.tanh(t)
    end
  end

  @default_rate 0.2
  @default_eps 0.001
  @default_act_func :sigmoid
  
  @inputs [
     [0,0],
     [1,0],
     [0,1],
     [1,1]
   ]

  @outputs [
     [0],
     [1],
     [1],
     [0]
   ]

  def test(count \\ 100, opts \\ nil) do
    model = 
      {2, [2], 1}
      |> build_model(opts || [eps: 0.01, rate: 0.3, act_func: :sigmoid])
      |> IO.inspect
      |> train_model(count, @inputs, @outputs)
      |> IO.inspect

    Enum.map(@inputs, fn input -> {input, model |> run_model(input) |> Nx.to_list()} end)    
  end

  def build_model(shape_or_list, opts \\ [])
  def build_model(list, opts) when is_list(list) do
    %Model{
      eps: Keyword.get(opts, :eps, @default_eps),
      rate: Keyword.get(opts, :rate, @default_rate),
      act_func: Keyword.get(opts, :act_func, @default_act_func),
      layers:
        list
        |> Enum.map(fn {ws, bs} ->
          %Layer{
            weights: Nx.tensor(ws, names: [:neuron, :input]),
            biases: Nx.tensor(bs, names: [:neuron])
          }
        end)
    }
  end
  def build_model({inputs, layers, outputs}, opts) do
    act_func = Keyword.get(opts, :act_func, @default_act_func)
    ranges = calc_range(act_func, inputs, outputs)
    
    %Model{
      eps: Keyword.get(opts, :eps, @default_eps),
      rate: Keyword.get(opts, :rate, @default_rate),
      act_func: act_func,
      layers: build_model_h(inputs, layers, outputs, ranges)
    }
  end
  defp build_model_h(_, _, 0, _), do: []
  defp build_model_h(inputs, [], outputs, ranges),
       do: build_model_h(inputs, [outputs], nil, ranges)
  defp build_model_h(inputs, [layer | layers], outputs,
                     {{w_ini, w_end}, {b_ini, b_end}} = ranges) do
    # key = Nx.Random.key(System.os_time())
    key = Nx.Random.key(1972)
    {weights, key} =
      Nx.Random.uniform(key, w_ini, w_end, shape: {layer, inputs}, names: [:neuron, :input])
    {biases, _} =
      Nx.Random.normal(key, b_ini, b_end, shape: {layer}, names: [:neuron])
    
    [
      %Layer{
        weights: weights,
        biases: biases
      }
      |
      build_model_h(layer, layers, outputs || 0, ranges)
    ]
  end

  def train_model(model, count \\ 500, inputs \\ @inputs, outputs \\ @outputs)
  def train_model(model, count, inputs, outputs)
      when is_list(hd(inputs)) or is_list(hd(outputs)) do
    lt_inputs = Enum.map(inputs, fn inp -> Nx.tensor(inp, type: :f32) end)
    lt_outputs = Enum.map(outputs, fn outp -> Nx.tensor(outp, type: :f32) end)
    train_model(model, count, lt_inputs, lt_outputs)
  end

  def train_model(model, 0, _, _), do: model
  def train_model(model, count, lt_inputs,lt_outputs) do
    cost = eval_model(model, lt_inputs, lt_outputs)
    IO.puts "Iteration: #{count} - Cost: #{cost}"

    model
    |> finite_diff(cost, lt_inputs, lt_outputs)
    |> learn(model)
    |> train_model(count - 1, lt_inputs, lt_outputs)
  end
  
  def run_model(model, input) when is_list(input) do
    t_input = Nx.tensor(input, type: :f32) 
    run_model(model, t_input)
  end
  def run_model(model, t_inputs) do
    feed_forward(model, [t_inputs]) |> hd()
  end
  
  ################################################################################################
  ## Private tools
  ################################################################################################

  def eval_model(model, lt_inputs, lt_outputs) do
    model
    |> feed_forward(lt_inputs)
    |> calc_cost(lt_outputs)
  end

  def feed_forward(%Model{} = model, lt_inputs),
      do: feed_forward(model.layers, lt_inputs, model.act_func)
  def feed_forward(_, [], _), do: []
  def feed_forward(layers, [t_input | lt_inputs], act_func) do
    [feed_forward_h(layers, t_input, act_func) | feed_forward(layers, lt_inputs, act_func)]
  end
  defp feed_forward_h([], t_input, _), do: t_input
  defp feed_forward_h([layer | layers], t_input, act_func) do
    t_input =
      t_input
      |> Nx.multiply(layer.weights)
      |> Nx.sum(axes: [:input])
      |> Nx.add(layer.biases)
      |> Nx.tensor()

    t_input = apply(Activation, act_func, [t_input])
    
    feed_forward_h(layers, t_input, act_func)
  end

  def calc_cost(lt_new_outputs, lt_outputs, sum \\ 0, count \\ 0)
  def calc_cost([], [], sum, count), do: sum / count
  def calc_cost([no | lt_new_outputs], [o | lt_outputs], sum, count) do
    sum =
      no
      |> Nx.subtract(o)
      |> Nx.pow(2)
      |> Nx.sum()
      |> Nx.add(sum)
      |> Nx.to_number()
    calc_cost(lt_new_outputs, lt_outputs, sum, count + 1)
  end

  def finite_diff(model, cost, lt_inputs, lt_outputs) do
    %Model{layers: layers} = model

    layers_weights =
      Enum.map(0..length(layers)-1, fn i ->
        {layer, _} = List.pop_at(layers, i)
        {rows, cols} = layer.weights.shape
        Enum.map(0..rows-1, fn r ->
          Enum.map(0..cols-1, fn c ->
            coords = Nx.tensor([r, c])
            # Set r,c weight + eps and get the cost of new model
            w =
              layer.weights
              |> Nx.gather(coords)
              |> Nx.add(model.eps)
              |> Nx.to_number()
            
            layer = %{layer | weights: Nx.indexed_put(layer.weights, coords, w)}
            n_cost =
              %Model{model | layers: List.replace_at(layers, i, layer)}
              |> eval_model(lt_inputs, lt_outputs)

            # Calcule new weight in function of eval result moving toward better cost
            ((n_cost - cost) / model.eps) 
          end)
        end)
      end)

    layers_biases =
      Enum.map(0..length(layers)-1, fn i ->
        {layer, _} = List.pop_at(layers, i)
        {rows} = layer.biases.shape
        Enum.map(0..rows-1, fn r ->
          coords = Nx.tensor([r])
          # Set r bias + eps and get the cost of new model
          b =
            layer.biases            
            |> Nx.gather(coords)
            |> Nx.add(model.eps)
            |> Nx.to_number()

          layer = %{layer | biases: Nx.indexed_put(layer.biases, coords, b)}
          n_cost =
            %Model{model | layers: List.replace_at(layers, i, layer)}
            |> eval_model(lt_inputs, lt_outputs)

          # Calcule new weight in function of eval result moving toward better cost
          ((n_cost - cost) / model.eps) 
        end)
      end)

    {opts, _} = model |> Map.to_list |> Keyword.split([:eps, :rate, :act_func])
    layers_weights
    |> :lists.zip(layers_biases)
    |> build_model(opts)
  end

  def learn(%Model{layers: new_layers}, %Model{layers: layers} = model),
      do: %Model{model | layers: learn(new_layers, layers, model.rate)}
  def learn([], [], _), do: [] 
  def learn([nl | new_layers], [l | layers], rate) do
    sub_weights = Nx.dot(nl.weights, rate)
    sub_biases = Nx.dot(nl.biases, rate)
    [
      %Layer{
        weights: l.weights |> Nx.subtract(sub_weights),
        biases: l.biases |> Nx.subtract(sub_biases)
      }
      |
      learn(new_layers, layers, rate)
    ]
  end

  # return {range_for_weights, range_for_biases}
  def calc_range(act_func, n_inputs, n_outputs) when act_func in [:tanh, :sigmoid] do
    range = :math.sqrt(6 / (n_inputs + n_outputs))
    {{-range, range}, {0, :math.sqrt(2 / (n_inputs + n_outputs))}}
  end
  def calc_range(:relu, n_inputs, _n_outputs) do
    range = :math.sqrt(2 / n_inputs)
    {{0, range}, {0, 0}}
  end

end
