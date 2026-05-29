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

  @default_rate 0.3
  @default_eps 0.01
  @default_act_func :tanh

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
        |> Enum.map(fn {ln_ws, ln_bs, ln_cs} ->
          :lists.zip3(ln_ws, ln_bs, ln_cs)
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
    # key = Nx.Random.key(1972)

    neurons_list =
      layer_conns
      |> Enum.map(fn neuron_conns ->
        key = Nx.Random.key(System.os_time())
        {weights, key} =
          Nx.Random.uniform(key, w_ini, w_end,
                            shape: {length(neuron_conns)}, names: [:weights])
        {bias, _} =
          Nx.Random.normal(key, b_ini, b_end, shape: {1}, names: [:bias])

        %Neuron{
          weights: weights,
          bias: bias,
          connections: Nx.tensor(neuron_conns)
        }
      end)

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
    cost = eval_model(model, lt_inputs, lt_outputs)
    IO.puts "Iteration: #{total - count + 1} - Cost: #{cost}"

    model
    |> finite_diff(cost, lt_inputs, lt_outputs)
    |> learn(model)
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
  defp feed_forward_h([ln_layer | layers], t_input, act_func) do
    t_input =
      Enum.map(ln_layer, fn neuron ->
        t_input
        |> Nx.take(neuron.connections)
        |> Nx.multiply(neuron.weights)
        |> Nx.sum(axes: [:weights])
        |> Nx.add(neuron.bias)
        |> Nx.tensor()
      end)
      |> Nx.concatenate()

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
      layers
      |> Enum.with_index()
      |> Enum.map(fn {ln_layer, i} ->
        ln_layer
        |> Enum.with_index()
        |> Enum.map(fn {neuron, r} ->
          {n} = Nx.shape(neuron.weights)
          Enum.map(0..n-1, fn c ->
            # Set  weight + eps and get the cost of new model
            w =
              neuron.weights
              |> Nx.take(c)
              |> Nx.add(model.eps)
              |> Nx.to_number()

            neuron = %{neuron | weights: Nx.indexed_put(neuron.weights, Nx.tensor([c]), w)}
            ln_layer = List.replace_at(ln_layer, r, neuron)

            n_cost_p =
              %Model{model | layers: List.replace_at(layers, i, ln_layer)}
              |> eval_model(lt_inputs, lt_outputs)

            w =
              neuron.weights
              |> Nx.take(c)
              |> Nx.subtract(model.eps)
              |> Nx.to_number()

            neuron = %{neuron | weights: Nx.indexed_put(neuron.weights, Nx.tensor([c]), w)}
            ln_layer = List.replace_at(ln_layer, r, neuron)

            n_cost_m =
              %Model{model | layers: List.replace_at(layers, i, ln_layer)}
              |> eval_model(lt_inputs, lt_outputs)

            # Calcule new weight in function of eval result moving toward better cost
            ((n_cost_p - n_cost_m) / (2*model.eps))
          end)
        end)
      end)

    layers_biases =
      layers
      |> Enum.with_index()
      |> Enum.map(fn {ln_layer, i} ->
        ln_layer
        |> Enum.with_index()
        |> Enum.map(fn {neuron, r} ->
          {n} = Nx.shape(neuron.bias)
          Enum.map(0..n-1, fn _ ->
            # Set  weight + eps and get the cost of new model
            b =
              neuron.bias
              |> Nx.add(model.eps)

            neuron = %{neuron | bias: b}
            ln_layer = List.replace_at(ln_layer, r, neuron)

            n_cost =
              %Model{model | layers: List.replace_at(layers, i, ln_layer)}
              |> eval_model(lt_inputs, lt_outputs)

            # Calcule new weight in function of eval result moving toward better cost
            ((n_cost - cost) / model.eps)
          end)
        end)
      end)

    layers_connections =      
      Enum.map(layers, fn ln_layer ->
        Enum.map(ln_layer, fn neuron ->
          Nx.to_list(neuron.connections)
        end)
      end)

    {opts, _} = model |> Map.to_list |> Keyword.split([:eps, :rate, :act_func])
    layers_weights
    |> :lists.zip3(layers_biases, layers_connections)
    |> build_model(opts)
  end

  def learn(%Model{layers: new_layers}, %Model{layers: layers} = model),
      do: %Model{model | layers: learn(new_layers, layers, model.rate)}
  def learn([], [], _), do: []
  def learn([ln_new_layer | new_layers], [ln_layer | layers], rate) do
    [
      ln_new_layer
      |> Enum.zip(ln_layer)
      |> Enum.map(fn {new_neuron, neuron} ->
        sub_weights = Nx.dot(new_neuron.weights, rate)
        sub_bias = Nx.dot(new_neuron.bias, rate)
        %Neuron{
          weights: Nx.subtract(neuron.weights, sub_weights),
          bias: Nx.subtract(neuron.bias, sub_bias),
          connections: neuron.connections
        }
      end)
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
