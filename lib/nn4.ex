defmodule NN4 do
  defmodule Model do
	  defstruct [
      layers: nil,
      rate: nil,
      act_funcs: nil,
      shape: nil,
      map_output_func: nil,
      shuffle: true,
      batch_mode: false
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
          rv = model |> run_model(inp) #|> Nx.to_list()
          cost = calc_cost(output_act_func, rv, out)
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
  act_funcs: list(:sigmoid | :tanh | :relu | :softmax)  (def. :tanh)
  connections: :all | :none | list()  (def. :all)
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

    connections =
      case Keyword.get(opts, :connections, :all) do
        :none -> []
        list when is_list(list) -> list
        _ -> build_all_connections(inputs, layers, outputs)
      end

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
    key = Nx.Random.key(System.os_time())
    # key = Nx.Random.key(1972)
    Enum.reduce(1..length(layers_list)-1, {[], key}, fn i, {layers, key} ->
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
            connections: Enum.at(connections, i-1)
          }
        ],
        key
      }
    end)
    |> elem(0)
  end

  def train_model(
        model,
        inputs,
        outputs,
        count \\ 500,
        opts \\ [],
        total \\ 0,
        cost \\ 99999)
  def train_model(model, inputs, outputs, count, opts, _, cost)
      when is_list(hd(inputs)) or is_list(hd(outputs)) do
    lt_inputs = Enum.map(inputs, fn inp -> Nx.tensor(inp, type: :f32) end)
    lt_outputs = Enum.map(outputs, fn outp -> Nx.tensor(outp, type: :f32) end)
    train_model(model, lt_inputs, lt_outputs, count, opts, count, cost)
  end

  def train_model(model, 0, _, _, _, _, _), do: model
  def train_model(model, count, lt_inputs, lt_outputs, opts, total, previous_cost) do
    every = Keyword.get(opts, :every, 1)
    model = %{
      model |
      rate: Keyword.get(opts, :rate, model.rate),
      batch_mode: Keyword.get(opts, :batch_mode, model.batch_mode),
      shuffle: Keyword.get(opts, :shuffle, model.shuffle),
      act_funcs: Keyword.get(opts, :act_funcs, model.act_funcs),
      map_output_func: Keyword.get(opts, :map_output_func, model.map_output_func)
    }
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
      |> eval_and_report(lt_outputs, count, total, every)
    else
      model
      |> back_propagation(lt_inputs, lt_outputs)
      |> learn_batch(total - count)
      |> eval_and_report(lt_outputs, count, total, every)
    end
    |> train_model(count - 1, lt_inputs, lt_outputs, opts, total, previous_cost)
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
      |> Enum.reduce({[{t_inputs, t_inputs}], t_inputs}, fn {ln_layer, i}, {r_zas, t_inputs} ->
        {_, a_output} = za = forward(ln_layer, t_inputs, Enum.at(act_funcs, i))
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
      [ln_layer | layers],
      [{t_z, t_a} | zas],
      deltas,
      {rate, [act_func | act_funcs]},
      grads_list) do

    # deltas |> IO.inspect(label: "DELTAS")
    # layers |> IO.inspect(label: "LAYERS")
    # ln_layer |> IO.inspect(label: "LAYER")

    grads =
      ln_layer
      |> Enum.with_index()
      |> Enum.map(fn {n_layer, i} ->
        delta = Nx.take(deltas, i)
        {t_a |> Nx.take(Nx.tensor(n_layer.connections)) |> Nx.multiply(delta), delta}
      end)
      # |> IO.inspect(label: "GRADS")

    derivative_z = derivative(act_func, t_z, t_a)

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
    backward(layers, zas, deltas, {rate, act_funcs}, [grads | grads_list])
  end

  ## return: {model, new_lt_outputs, accum_grads}
  defp learn_batch({model, new_lt_outputs, accum_grads}, iteration) do
    grads_count = length(accum_grads)
    [first_layers_list | accum_grads] = accum_grads

    grads =
      accum_grads
      |> Enum.reduce(first_layers_list, fn layers_list, acc_layers_list ->
        acc_layers_list
        |> Enum.zip(layers_list)
        |> Enum.map(fn {grads_accum, grads_layer} ->
          grads_accum
          |> Enum.zip(grads_layer)
          |> Enum.map(fn {{aws, ab}, {gws, gb}} ->
            {Nx.add(aws, gws), Nx.add(ab, gb)}
          end)
        end)
      end)

    {learn(grads, model, grads_count, iteration), new_lt_outputs, grads}
  end

  ## return model
  defp learn(grads, model, grads_count \\ 1, _iteration \\ 1) do
    %{layers: layers} = model
    %{model |
      layers:
        layers
        |> Enum.zip(grads)
        |> Enum.map(fn {ln_layer, l_grads} ->
          ln_layer
          |> Enum.zip(l_grads)
          |> Enum.map(fn {n_layer, {grad_ws, grad_b}} ->
            ws =
              n_layer.weights
              |> Nx.subtract(
                grad_ws
                |> Nx.divide(grads_count)
                |> Nx.dot(model.rate) # * (0.99**iteration))
              )

            b =
              n_layer.bias
              |> Nx.rename([nil])
              |> Nx.subtract(
                grad_b
                |> Nx.divide(grads_count)
                |> Nx.dot(model.rate) # * (0.99**iteration))
              )
              |> Nx.rename([:bias])
            %{n_layer | weights: ws, bias: b}
          end)
        end)
    }
  end

  defp forward(ln_layer, t_input, act_func) do
    t_output =
      Enum.map(ln_layer, fn neuron ->
        t_input
        |> Nx.take(Nx.tensor(neuron.connections))
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

  defp feed_forward(%Model{} = model, lt_inputs),
      do: feed_forward(model.layers, lt_inputs, model.act_funcs)
  defp feed_forward(_, [], _), do: []
  defp feed_forward(layers, [t_input | lt_inputs], act_funcs) do
    [feed_forward_h(layers, t_input, act_funcs) | feed_forward(layers, lt_inputs, act_funcs)]
  end
  defp feed_forward_h([], t_input, _), do: t_input
  defp feed_forward_h([ln_layer | layers], t_input, [act_func | act_funcs]) do
    {_, new_t_input} = forward(ln_layer, t_input, act_func)
    feed_forward_h(layers, new_t_input, act_funcs)
  end

  defp eval_and_report(
         {model, lt_new_outputs, _}, lt_outputs, count, total, every)
       when rem(count, every) == 0 or count == total do
    output_act_func = model.act_funcs |> Enum.reverse() |> hd()
    cost = calc_cost(output_act_func, lt_new_outputs, lt_outputs)
    IO.puts "Iteration: #{total - count + 1} - Cost: #{cost}"
    model
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
