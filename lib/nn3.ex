defmodule NN3 do
  defmodule Model do
	  defstruct [
      layers: nil,
      eps: nil,
      rate: nil,
      act_func: nil
    ]
  end

  defmodule Neuron do
	  defstruct [
      weights: nil,
      bias: nil,
      connections: nil
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

  @default_rate 0.1
  @default_eps 0.01
  @default_act_func :tanh

  @inputs [
     [0,1],
     [0,0],
     [1,0],
     [1,1]
   ]

  @outputs [
     [1],
     [0],
     [1],
     [0]
   ]

  # def test(count \\ 100, opts \\ nil) do
  #   model =
  #     {2, [2], 1}
  #     |> build_model(opts || [eps: 0.01, rate: 0.3, act_func: :sigmoid])
  #     |> IO.inspect
  #     |> train_model(count, @inputs, @outputs)
  #     |> IO.inspect
  #   Enum.map(@inputs, fn input -> {input, model |> run_model(input) |> Nx.to_list()} end)
  # end

  def test(model, inputs, outputs) do
    inputs
    |> Enum.zip(outputs)
    |> Enum.map(fn {inp, out} ->
      rv = model |> run_model(inp) |> Nx.to_list() |> hd()
      {inp, out, round(rv), rv}
    end)
  end

  def build_model(shape_or_list, opts \\ [])
  def build_model(list, opts) when is_list(list) do
    %Model{
      eps: Keyword.get(opts, :eps, @default_eps),
      rate: Keyword.get(opts, :rate, @default_rate),
      act_func: Keyword.get(opts, :act_func, @default_act_func),
      layers:
        list
        |> Enum.map(fn ln_layer ->
          ln_layer
          |> Enum.map(fn {ws, b, cs} ->
            %Neuron{
              weights: Nx.tensor(ws, names: [:weights]),
              bias: Nx.tensor(b, names: [:bias]),
              connections: Nx.tensor(cs)
            }
          end)
        end)
    }
  end
  def build_model({inputs, layers, outputs}, opts) when not is_list(layers),
      do: build_model({inputs, [layers], outputs}, opts)
  def build_model({inputs, layers, outputs}, opts) do
    act_func = Keyword.get(opts, :act_func, @default_act_func)
    ranges = calc_range(act_func, inputs, outputs)
    default_connections =
      [inputs, layers, outputs]
      |> List.flatten()
      |> build_all_connections()

    connections = Keyword.get(opts, :connections, default_connections)

    %Model{
      eps: Keyword.get(opts, :eps, @default_eps),
      rate: Keyword.get(opts, :rate, @default_rate),
      act_func: act_func,
      layers: build_model_layers(connections, ranges)
    }
  end
  defp build_model_layers([], _), do: []
  defp build_model_layers([layer_conns | connections],
                          {{w_ini, w_end}, {b_ini, b_end}} = ranges) do
    # key = Nx.Random.key(System.os_time())
    key = Nx.Random.key(1972)

    neurons_list =
      layer_conns
      |> Enum.reduce({key, []}, fn neuron_conns, {key, res} ->
        {weights, key} =
          Nx.Random.uniform(key, w_ini, w_end,
                            shape: {length(neuron_conns)}, names: [:weights])
        {bias, key} =
          Nx.Random.normal(key, b_ini, b_end, shape: {1}, names: [:bias])

        {key, res ++ [
          %Neuron{
            weights: weights,
            bias: bias,
            connections: Nx.tensor(neuron_conns)
          }
        ]}
      end)
      |> elem(1)

    [neurons_list | build_model_layers(connections, ranges)]
  end

  defp build_all_connections([_]), do: []
  defp build_all_connections([is,ns | rest]) do
    [
      Enum.map(0..ns-1, fn _ ->
        Range.to_list(0..is-1)
      end)
      |
      build_all_connections([ns | rest])
    ]
  end

  def train_model(model, count \\ 500, inputs \\ @inputs, outputs \\ @outputs, total \\ 0)
  def train_model(model, count, inputs, outputs, _)
      when is_list(hd(inputs)) or is_list(hd(outputs)) do
    lt_inputs = Enum.map(inputs, fn inp -> Nx.tensor(inp, type: :f32) end)
    lt_outputs = Enum.map(outputs, fn outp -> Nx.tensor(outp, type: :f32) end)
    train_model(model, count, lt_inputs, lt_outputs, count)
  end

  def train_model(model, 0, _, _, _), do: model
  def train_model(model, count, lt_inputs,lt_outputs, total) do
    model
    |> back_propagation(lt_inputs, lt_outputs)
    |> eval_and_report(lt_outputs, count, total)
    |> train_model(count - 1, lt_inputs, lt_outputs, total)
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

  def back_propagation(model, lt_inputs, lt_outputs, new_lt_outputs \\ [])
  def back_propagation(model, [], _, new_lt_outputs), do: {model, new_lt_outputs}
  def back_propagation(model, [t_inputs | lt_inputs], [t_outputs | lt_outputs], new_lt_outputs) do
    {model, new_t_outputs} = back_propagation_h(model, t_inputs, t_outputs)
    back_propagation(model, lt_inputs, lt_outputs, new_lt_outputs ++ [new_t_outputs])
  end
  defp back_propagation_h(model, t_inputs, t_outputs) do
    %Model{layers: layers, act_func: act_func} = model

    ## First do forward operation and get all z's and a's
    ## zas -> [{z_layer_n, a_layer_n}, ..., {z_layer_2, a_layer_2}, {z_layer_1, a_layer_1}]
    {zas, new_t_outputs} = Enum.reduce(
                 layers, {[{t_inputs, t_inputs}], t_inputs}, fn ln_layer, {r_zas, t_inputs} ->
      {_, a_output} = za = forward(ln_layer, t_inputs, act_func)
      {[za | r_zas], a_output}
    end)

    ## Second calculate the output delta
    [{t_z, t_a} | zas] = zas
    t1 = Nx.subtract(t_a, t_outputs)
    t2 = derivative(model.act_func, t_z)
    deltas_o = Nx.multiply(t1, t2)

    rlayers = Enum.reverse(layers)

    new_model =
      rlayers
      |> backward(zas, deltas_o, {model.rate, model.act_func})
      |> learn(model)

    {new_model, new_t_outputs}
  end

  defp backward(layers, zas, deltas, model_params, result \\ [])
  defp backward(_, [], _, _, result), do: result
  defp backward([ln_layer | layers], [{t_z, t_a} | zas], deltas, {rate, act_func}, result) do
    # deltas |> IO.inspect(label: "DELTAS")
    # ln_layer |> IO.inspect(label: "LAYER")
    grads =
      ln_layer
      |> Enum.with_index()
      |> Enum.map(fn {_, i} ->
        delta = Nx.take(deltas, i)
        {Nx.multiply(t_a, delta), delta}
      end)

    derivative_z = derivative(act_func, t_z)
    deltas =
      if length(layers) > 0 do
        ln_layer
        |> Enum.with_index()
        |> Enum.reduce(nil, fn {n_layer, i}, acc ->
          d = n_layer.weights |> Nx.multiply(Nx.take(deltas, i)) |> Nx.multiply(derivative_z)
          acc && Nx.concatenate([acc, d]) || d
        end)
        # |> IO.inspect(label: "DELTAS")
      else
        nil
      end
    # IO.inspect "----------------------------------------"
    backward(layers, zas, deltas, {rate, act_func}, [grads | result])
  end

  defp learn(grads, model) do
    %{layers: layers} = model
    %Model{model |
      layers: 
        layers
        |> Enum.zip(grads)
        |> Enum.map(fn {ln_layer, l_grads} ->
          ln_layer
          |> Enum.zip(l_grads)
          |> Enum.map(fn {n_layer, {grad_ws, grad_b}} ->
            ws =
              n_layer.weights
              |> Nx.subtract(Nx.dot(grad_ws, model.rate))

            b =
              n_layer.bias
              |> Nx.rename([nil])
              |> Nx.subtract(Nx.dot(grad_b, model.rate))
              |> Nx.rename([:bias])
            %Neuron{n_layer | weights: ws, bias: b}
          end)
        end)      
    }
  end
  
  defp forward(ln_layer, t_input, act_func) do
    t_output =
      Enum.map(ln_layer, fn neuron ->
        t_input
        |> Nx.take(neuron.connections)
        |> Nx.multiply(neuron.weights)
        |> Nx.sum(axes: [:weights])
        |> Nx.add(neuron.bias)
        |> Nx.tensor()
      end)
      |> Nx.concatenate()
      |> Nx.rename([nil])

    # {z, a} -> {layer_output, activation(layer_output)}
    {t_output, apply(Activation, act_func, [t_output])}
  end

  def feed_forward(%Model{} = model, lt_inputs),
      do: feed_forward(model.layers, lt_inputs, model.act_func)
  def feed_forward(_, [], _), do: []
  def feed_forward(layers, [t_input | lt_inputs], act_func) do
    [feed_forward_h(layers, t_input, act_func) | feed_forward(layers, lt_inputs, act_func)]
  end
  defp feed_forward_h([], t_input, _), do: t_input
  defp feed_forward_h([ln_layer | layers], t_input, act_func) do
    {_, new_t_input} = forward(ln_layer, t_input, act_func)
    feed_forward_h(layers, new_t_input, act_func)
  end

  def eval_and_report({model, lt_new_outputs}, lt_outputs, count, total) do
    cost = calc_cost(lt_new_outputs, lt_outputs)
    IO.puts "Iteration: #{total - count + 1} - Cost: #{cost}"
    model
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

  # return {range_for_weights, range_for_biases}
  def calc_range(act_func, n_inputs, n_outputs) when act_func in [:tanh, :sigmoid] do
    range = :math.sqrt(6 / (n_inputs + n_outputs))
    {{-range, range}, {0, :math.sqrt(2 / (n_inputs + n_outputs))}}
  end
  def calc_range(:relu, n_inputs, _n_outputs) do
    range = :math.sqrt(2 / n_inputs)
    {{0, range}, {0, 0}}
  end

  def derivative(:tanh, t) do
    Nx.subtract(Nx.tensor(1), Nx.pow(Nx.tanh(t), 2))
  end

  def derivative(:sigmoid, t) do
    Nx.dot(t, Nx.subtract(Nx.tensor(1), t))
  end

end
