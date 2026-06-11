defmodule NN4 do
  defmodule Model do
	  defstruct [
      layers: nil,
      rate: nil,
      act_funcs: nil,
      shape: nil,
      map_output_func: nil,
      shuffle: true,
      batch_mode: false,
      _temp: %{}
    ]
  end

  defmodule Neurons do
	  defstruct [
      weights: nil,
      biases: nil,
      connections: []
    ]
  end

  defmodule Activation do
    import Nx.Defn

    defn softmax(z) do
      max_z = Nx.reduce_max(z, axes: [0], keep_axes: true)
      exp_z = Nx.exp(Nx.subtract(z, max_z))
      sum_exp = Nx.sum(exp_z, axes: [0], keep_axes: true)
      Nx.divide(exp_z, sum_exp)
    end

    defn relu(z) do
      Nx.max(0, z)
    end

    defn sigmoid(z) do
      Nx.sigmoid(z)
    end

    defn tanh(z) do
      Nx.tanh(z)
    end

    defn none(z) do
      z
    end
  end

  @default_rate 0.1
  @default_act_func :tanh

  def test(model, inputs, outputs \\ nil) do
    fun = fn {list, hits, costs} ->
      IO.puts "HITS: #{100 * hits / length(inputs)}% - AVG COST: #{costs / length(inputs)}"
      list
    end
    map = model.map_output_func

    output_act_func = model.act_funcs |> Enum.reverse() |> hd()

    [
      {"input", "expected", "rounded out", "raw out"}
      |
        inputs
        |> Enum.zip(outputs)
        |> Enum.reduce({[], 0, 0}, fn {inp, out}, {list, hits, costs} ->
          rv = model |> run_model(inp) 
          cost = calc_cost(output_act_func, [rv], [Nx.tensor(out)])
          rv = rv |> Nx.to_list()
          exp_out = Enum.map(out, fn v -> map.(v) end)
          got_out = Enum.map(rv, fn v -> map.(v) end)
          {
            [{inp, exp_out, got_out, rv} | list],
            hits + (exp_out == got_out && 1 || 0),
            costs + cost
          }
        end)
        |> fun.()
    ]
  end

  def suggest_hidden_size(inputs, outputs, samples) do
    # Rule of thumb
    size1 = (inputs + outputs) * 2 / 3
    size2 = samples / (inputs + outputs)

    # Take average, round up
    max(round((size1 + size2) / 2), 2)
  end

  @doc """
  :act_func        -> :sigmoid | :tanh | :relu | :softmax  (def. :tanh)
  :output_act_func -> :sigmoid | :tanh | :relu | :softmax  (def. act_func)
  :map_output_func -> function to decode output (def. fn v -> v end)
  :rate            -> default learning rate if not set in train_model call (def @default_rate)
  :batch_mode      -> if true process gradients in batch (def. true)
  :shuffle         -> if true shuffle input in every epoch (def. true)
  """
  def build_model({inputs, layers, outputs}, opts) when not is_list(layers),
      do: build_model({inputs, [layers], outputs}, opts)
  def build_model({inputs, layers, outputs}, opts) do
    total_layers =
      [inputs, layers, outputs]
      |> List.flatten()
      |> Enum.filter(&(&1 != 0))

    hidden_layers = Enum.filter(layers, &(&1 != 0))

    act_func = Keyword.get(opts, :act_func, @default_act_func)
    output_act_func = Keyword.get(opts, :output_act_func, act_func)
    act_funcs = Enum.map(1..length(hidden_layers)//1, fn _ -> act_func end) ++ [output_act_func]
    map_output_func = Keyword.get(opts, :map_output_func, fn v -> v end)

    connections = []
      # case Keyword.get(opts, :connections, :all) do
      #   :none -> []
      #   list when is_list(list) -> list
      #   _ -> build_all_connections(inputs, layers, outputs)
      # end

    %Model{
      rate: Keyword.get(opts, :rate, @default_rate),
      batch_mode: Keyword.get(opts, :batch_mode, true),
      act_funcs: act_funcs,
      shape: List.flatten([inputs, layers, outputs]),
      layers: build_model_layers(total_layers, connections, act_funcs, outputs),
      map_output_func: map_output_func
    }
  end

  defp build_model_layers(layers_list, connections, act_funcs, outputs) do
    # key = Nx.Random.key(System.os_time())
    key = Nx.Random.key(1972)
    Enum.reduce(1..length(layers_list)-1, {[], key}, fn i, {layers, key} ->
      connections = Enum.at(connections, i-1)
      neurons_count = Enum.at(layers_list, i)
      weights_count = Enum.at(layers_list, i-1)

      {{w_ini, w_end}, {b_ini, b_end}} =
          act_funcs
          |> Enum.at(i-1)
          |> calc_range(neurons_count, outputs)

      {weights, key} =
        Nx.Random.uniform(
          key,
          w_ini, w_end,
          shape: {neurons_count, weights_count},
          names: [:neuron, :weights]
        )

      {biases, key} =
        Nx.Random.uniform(
          key,
          b_ini, b_end,
          shape: {neurons_count, 1},
          names: [:neuron, :biases]
        )

      {
        layers ++ [
          %Neurons{
            weights: weights,
            biases: biases,
            connections: connections
          }
        ],
        key
      }
    end)
    |> elem(0)
  end

  def model_from(model) do
    %Model{
      rate: model.rate,
      act_funcs: model.act_funcs,
      shape: model.shape,
      map_output_func: model.map_output_func,
      shuffle: model.shuffle,
      batch_mode: model.batch_mode,
      layers:        
        Enum.map(model.layers, fn layer ->
          # new_layer = 
            Enum.reduce(layer, nil, fn neuron, neurons ->
              if neurons do
                w = Nx.stack([neuron.weights]) |> Nx.rename([:neuron, :weights])
                b = Nx.stack([neuron.bias]) |> Nx.rename([:neuron, :biases])
                %Neurons{
                  weights: Nx.concatenate([neurons.weights, w]),
                  biases: Nx.concatenate([neurons.biases, b]), 
                  connections: neurons.connections ++ [neuron.connections]
                }
              else
                %Neurons{
                  weights: Nx.stack([neuron.weights]) |> Nx.rename([:neuron, :weights]),
                  biases: Nx.stack([neuron.bias]) |> Nx.rename([:neuron, :biases]),
                  connections: [neuron.connections]
                }
              end
            end)
        end)
    }
  end

  def train_model(model, inputs, outputs, count \\ 500, opts \\ [])
  def train_model(model, inputs, outputs, count, opts) do
    lt_inputs = 
      if is_list(hd(inputs)) do
        Enum.map(inputs, fn val -> Nx.tensor(val, type: :f32) end)
      else
        inputs
      end

    lt_outputs = 
      if is_list(hd(outputs)) do
        Enum.map(outputs, fn val -> Nx.tensor(val, type: :f32) end)
      else
        outputs
      end
    
    every = Keyword.get(opts, :every, 1)

    %{model |
      rate: Keyword.get(opts, :rate, model.rate),
      batch_mode: Keyword.get(opts, :batch_mode, model.batch_mode),
      shuffle: Keyword.get(opts, :shuffle, model.shuffle),
      act_funcs: Keyword.get(opts, :act_funcs, model.act_funcs),
      map_output_func: Keyword.get(opts, :map_output_func, model.map_output_func),
      _temp:
        model._temp
        |> put_in([:count], count)
        |> put_in([:total], count)
        |> put_in([:every], every)
        |> put_in([:cost], Float.max_finite())      
    }
    |> train_model_h(lt_inputs, lt_outputs, count)
  end

  def train_model_h(model, _, _, 0), do: model
  def train_model_h(model, lt_inputs, lt_outputs, count) do
    model = %{model | _temp: put_in(model._temp, [:count], count)}
    
    {lt_inputs, lt_outputs} =
      if model.shuffle do
        indices = Enum.shuffle(0..length(lt_inputs)-1)
        {
          indices |> Enum.map(fn i -> Enum.at(lt_inputs, i) end),
          indices |> Enum.map(fn i -> Enum.at(lt_outputs, i) end)
        }
      else
        {lt_inputs, lt_outputs}
      end

    if not model.batch_mode do
      model
      |> back_propagation(lt_inputs, lt_outputs)
      |> eval_and_report(lt_outputs)
    else
      model
      |> back_propagation(lt_inputs, lt_outputs)
      |> learn_batch()
      |> eval_and_report(lt_outputs)
    end
    |> train_model_h(lt_inputs, lt_outputs, count - 1)
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

  ## return: {model, new_lt_outputs, accum_grads}
  defp back_propagation(model, lt_inputs, lt_outputs, new_lt_outputs \\ [], accum_grads \\ [])
  defp back_propagation(model, [], _, new_lt_outputs, accum_grads),
       do: {model, new_lt_outputs, accum_grads}
  defp back_propagation(
         model,
         [t_inputs | lt_inputs],
         [t_outputs | lt_outputs],
         new_lt_outputs,
         accum_grads
       ) do
    {model, new_t_outputs, grads_list} = back_propagation_h(model, t_inputs, t_outputs)

    back_propagation(
      model,
      lt_inputs,
      lt_outputs,
      new_lt_outputs ++ [new_t_outputs],
      accum_grads ++ [grads_list]
    )
  end
  defp back_propagation_h(model, t_inputs, t_outputs) do
    %Model{layers: layers, act_funcs: act_funcs} = model
    ## First do forward operation and get all z's and a's
    ## zas -> [{z_layer_n, a_layer_n}, ..., {z_layer_2, a_layer_2}, {z_layer_1, a_layer_1}]
    {zas, new_t_outputs} =
      layers
      |> Enum.with_index()
      |> Enum.reduce({[{t_inputs, t_inputs}], t_inputs}, fn {layer, i}, {r_zas, t_inputs} ->
        {_, a_output} = za = forward(layer, t_inputs, Enum.at(act_funcs, i))
        {[za | r_zas], a_output}
      end)

    r_act_funcs = Enum.reverse(model.act_funcs)

    ## Second calculate the output delta
    [{t_z, t_a} | zas] = zas
    deltas_o =
      case hd(r_act_funcs) do
        :softmax -> Nx.subtract(t_a, t_outputs)
        other ->
          t1 = Nx.subtract(t_a, t_outputs)
          t2 = derivative(other, t_z, t_a, t_outputs)
          Nx.multiply(t1, t2) 
      end 

    grads_list =
      layers
      |> Enum.reverse()
      |> backward(zas, deltas_o, {model.rate, r_act_funcs})

    if not model.batch_mode do
      {learn(grads_list, model), new_t_outputs, grads_list}
    else
      {model, new_t_outputs, grads_list}
    end
  end

  ## return: grads_list
  defp backward(layers, zas, deltas, model_params, grads_list \\ [])
  defp backward(_, [], _, _, grads_list), do: grads_list
  defp backward(
      [layer | layers],
      [{t_z, t_a} | zas],
      deltas,
      {rate, [act_func | act_funcs]},
      grads_list) do

    # deltas |> IO.inspect(label: "DELTAS")
    # t_a |> IO.inspect(label: "T_A")
    # zas |> IO.inspect(label: "ZAS")
    # layers |> IO.inspect(label: "LAYERS")
    # layer |> IO.inspect(label: "LAYER")

    grads =
      Enum.reduce(0..Nx.size(deltas)-1, nil, fn
        i,  nil ->
          delta = Nx.take(deltas, i)
          t = Nx.multiply(t_a, delta)
          t = t  |> Nx.reshape({1,Nx.size(t)})
          {t,  Nx.reshape(delta, {1,1})}
        i, {gw, gb} ->
          delta = Nx.take(deltas, i)
          t = Nx.multiply(t_a, delta)
          t = t  |> Nx.reshape({1,Nx.size(t)})
          {
            Nx.concatenate([gw, t]),
            Nx.concatenate([gb, delta |> Nx.reshape({1,1})])
          }
      end)
      |> IO.inspect(label: "GRADS")

    derivative_z = derivative(act_func, t_z, t_a) 

    deltas =
      if length(layers) > 0 do
        ws = layer.weights |> Nx.transpose()
        ws |> Nx.dot(deltas) |> Nx.multiply(derivative_z) 
      else
        nil
      end
      # |> IO.inspect(label: "DELTAS")
    
    # IO.inspect "----------------------------------------"
    backward(layers, zas, deltas, {rate, act_funcs}, [grads | grads_list])
  end

  ## return: {model, new_lt_outputs, accum_grads}
  defp learn_batch({model, new_lt_outputs, accum_grads}) do
    grads_count = length(accum_grads) 
    [first_layers_list | accum_grads] = accum_grads

    grads =
      accum_grads
      |> Enum.reduce(first_layers_list, fn layers_list, acc_layers_list ->
        acc_layers_list
        |> Enum.zip(layers_list)
        |> Enum.map(fn {{aws, abs}, {gws, gbs}} ->
            {Nx.add(aws, gws), Nx.add(abs, gbs)}
        end)
      end)
      |> Enum.map(fn {gws, gbs} ->
        {Nx.divide(gws, grads_count), Nx.divide(gbs, grads_count)}
      end)

    {learn(grads, model), new_lt_outputs, grads}
  end

  ## return model
  defp learn(grads, model) do
    %{layers: layers} = model
    %{model |
      layers:
        layers
        |> Enum.zip(grads)
        |> Enum.map(fn {layer, {gws, gbs}} ->
          ws =
            layer.weights
            |> Nx.subtract(Nx.dot(gws, model.rate))

          bs =
            layer.biases
            |> Nx.subtract(Nx.dot(gbs, model.rate))
          %{layer | weights: ws, biases: bs}
        end)        
    }
  end

  ## return: {z, a} -> {layer_output, activation(layer_output)}
  defp forward(layer, t_input, act_func) do
    z =
      t_input
      |> Nx.multiply(layer.weights)
      |> Nx.sum(axes: [:weights])
      |> Nx.stack()
      |> Nx.transpose()
      |> Nx.add(layer.biases)
      |> Nx.transpose()
      |> Nx.reshape({Nx.size(layer.biases)})

    {z, apply(Activation, act_func, [z])}
  end

  defp feed_forward(%Model{} = model, lt_inputs),
      do: feed_forward(model.layers, lt_inputs, model.act_funcs)
  defp feed_forward(_, [], _), do: []
  defp feed_forward(layers, [t_input | lt_inputs], act_funcs) do
    [feed_forward_h(layers, t_input, act_funcs) | feed_forward(layers, lt_inputs, act_funcs)]
  end
  defp feed_forward_h([], t_input, _), do: t_input
  defp feed_forward_h([layer | layers], t_input, [act_func | act_funcs]) do
    {_, new_t_input} = forward(layer, t_input, act_func)
    feed_forward_h(layers, new_t_input, act_funcs)
  end

  defp eval_and_report(
         {%{_temp: %{count: count, total: total, every: every}} = model, lt_new_outputs, _},
         lt_outputs
       ) when rem(count, every) == 0 or count == total do
    output_act_func = model.act_funcs |> Enum.reverse() |> hd()
    cost = calc_cost(output_act_func, lt_new_outputs, lt_outputs)
    IO.puts "Iteration: #{total - count + 1} - Cost: #{cost}"
    %{model | _temp: put_in(model._temp, [:cost], cost)}
  end

  ## return [ [ [],[],... ], [ [],[],... ] ]
  def build_all_connections(inputs, layers, outputs),
       do:
         [inputs, layers, outputs]
         |> List.flatten()
         |> Enum.filter(&(&1 != 0))
         |> build_all_connections
  def build_all_connections([_]), do: []
  def build_all_connections([is,ns | rest]) do
    [
      Enum.map(0..ns-1, fn _ ->
        Range.to_list(0..is-1)
      end)
      |
      build_all_connections([ns | rest])
    ]
  end


  defp calc_cost(act_func, lt_new_outputs, lt_outputs, sum \\ 0, count \\ 0)
  ## for softmax
  defp calc_cost(:softmax, lt_new_outputs, lt_outputs, _sum, _count) do
    a = Nx.stack(lt_new_outputs)
    target = Nx.stack(lt_outputs)
    log_a = Nx.log(Nx.add(a, 1.0e-8))
    product = Nx.multiply(target, log_a)
    sum = Nx.sum(product, axes: [1])
    mean = Nx.mean(sum)
    mean |> Nx.negate() |> Nx.to_number()
  end
  ## for the rest
  defp calc_cost(_, [], [], sum, count), do: sum / count
  defp calc_cost(act_func, [no | lt_new_outputs], [o | lt_outputs], sum, count) do
    sum =
      no
      |> Nx.subtract(o)
      |> Nx.pow(2)
      |> Nx.sum()
      |> Nx.add(sum)
      |> Nx.to_number()
    calc_cost(act_func, lt_new_outputs, lt_outputs, sum, count + 1)
  end

  # return {range_for_weights, range_for_biases}
  defp calc_range(:none, n_inputs, n_outputs) do
    range = :math.sqrt(6 / (n_inputs + n_outputs))
    {{-range, range}, {0, :math.sqrt(2 / (n_inputs + n_outputs))}}
  end
  defp calc_range(:tanh, n_inputs, n_outputs) do
    range = :math.sqrt(6 / (n_inputs + n_outputs))
    {{-range, range}, {0, :math.sqrt(2 / (n_inputs + n_outputs))}}
  end
  defp calc_range(:softmax, n_inputs, n_outputs) do
    range = :math.sqrt(6 / (n_inputs + n_outputs))
    {{-range, range}, {0, :math.sqrt(2 / (n_inputs + n_outputs))}}
  end
  defp calc_range(:sigmoid, n_inputs, n_outputs) do
    {{rw1, rw2}, {rb1, rb2}} = calc_range(:tanh, n_inputs, n_outputs)
    {{rw1/2, rw2/2}, {rb1, rb2}}
  end
  defp calc_range(:relu, n_inputs, _n_outputs) do
    range = :math.sqrt(2 / n_inputs)
    {{0, range}, {0, 0}}
  end

  defp derivative(act_func, t_z, t_a, t_output \\ 0)
  defp derivative(:tanh, _t_z, t_a, _t_output) do
    Nx.subtract(Nx.tensor(1), Nx.pow(t_a, 2))
  end
  defp derivative(:sigmoid, _t_z, t_a, _t_output) do
    Nx.multiply(t_a, Nx.subtract(Nx.tensor(1), t_a))
  end
  defp derivative(:relu, t_z, _t_a, _t_output) do
    Nx.greater(t_z, 0)
  end
  defp derivative(:softmax, _t_z, t_a, t_output) do
    Nx.subtract(t_a, t_output)
  end
  defp derivative(:none, t_z, _t_a, _t_output) do
    t_z
  end

  def list_get(value, []), do: value
  def list_get(list, [i | coords]) do
    Enum.at(list, i) |> list_get(coords)
  end

  def list_put(_, [], value), do: value
  def list_put(list, [i | coords], value) do
    List.update_at(list, i, fn sublist -> list_put(sublist, coords, value) end)
  end

  def list_update(value, [], fun), do: fun.(value)
  def list_update(list, [i | coords], fun) do
    List.update_at(list, i, fn sublist -> list_update(sublist, coords, fun) end)
  end
end
