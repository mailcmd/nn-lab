defmodule NN5 do
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

  defmodule Layer do
	  defstruct [
      weights: nil,
      biases: nil,
      act_func: nil
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
    {size, _} = Nx.shape(inputs)

    [
      {"input", "expected", "rounded out", "raw out"}
      |
        0..size-1
        |> Enum.reduce({[], 0, 0}, fn i, {list, hits, costs} ->
          inp = Nx.take(inputs, i) 
          outp = Nx.take(outputs, i) 
          rv = model |> run_model(inp)
          # cost = calc_cost(output_act_func, Nx.tensor(rv), outp)
          cost = 0
          exp_out = Nx.to_list(outp)
          got_out = rv
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

    %Model{
      rate: Keyword.get(opts, :rate, @default_rate),
      batch_mode: Keyword.get(opts, :batch_mode, true),
      act_funcs: act_funcs,
      shape: List.flatten([inputs, layers, outputs]),
      layers: build_model_layers(total_layers, act_funcs, outputs),
      map_output_func: map_output_func,
      _temp: %{
        count: 0,
        total: 0,
        cost: Float.max_finite(),
        every: 1
      }
    }
  end

  defp build_model_layers(layers_list, act_funcs, outputs) do
    # key = Nx.Random.key(System.os_time())
    key = Nx.Random.key(1972)
    Enum.reduce(1..length(layers_list)-1, {[], key}, fn i, {layers, key} ->
      neurons_count = Enum.at(layers_list, i)
      weights_count = Enum.at(layers_list, i-1)
      act_func = Enum.at(act_funcs, i-1)

      {{w_ini, w_end}, {b_ini, b_end}} = calc_range(act_func, neurons_count, outputs)

      {weights, key} =
        Nx.Random.uniform(
          key,
          w_ini, w_end,
          shape: {neurons_count, weights_count}
          # names: [:neuron, :weights]
        )

      {biases, key} =
        Nx.Random.uniform(
          key,
          b_ini, b_end,
          shape: {neurons_count, 1}          
          # names: [:neuron, :biases]
        )

      {
        layers ++ [
          %Layer{
            weights: weights,
            biases: biases,
            act_func: act_func
          }
        ],
        key
      }
    end)
    |> elem(0)
  end

  def run_model(model, input) when is_list(input) do
    t_input = Nx.tensor(input, type: :f32)
    run_model(model, t_input)
  end
  def run_model(model, t_inputs) do
    model
    |> go_forward(t_inputs |> Nx.new_axis(0))
    |> hd()
    |> elem(1)
    |> Nx.flatten()
    |> Nx.to_list()
  end
  
  @doc """
  `inputs`:
    - can be a stream of a batched tensor, a tensor with shape [rows][input_size] or a list of
      tensors.
  
  `outputs`:
    - can be a stream of a batched tensor, a tensor with shape [rows][input_size] or a list of
      tensors. In any case must have the same struct that `inputs`

  Note about var prefixes:
    - s_<varname>  : Stream
    - lt_<varname> : List of tensors
    - tt_<varname> : Tensor all in one (bidimension tensor)
  """
  def train_model(model, inputs, outputs, count \\ 500, opts \\ [])
  def train_model(model, inputs, outputs, count, opts) do
    model
    |> tune_model(opts)
    |> train_model_h(inputs, outputs, count)
  end

  defp train_model_h(model, _, _, 0), do: model
  ## inputs IS A STREAM (A BATCH)
  defp train_model_h(model, %{__struct__: Stream} = s_inputs, s_outputs, count) do
    s_inputs
    |> Stream.zip(s_outputs)
    |> Enum.reduce(model, fn {tt_input, tt_output}, model ->
      # count = -1 avoid take tt_input like the only input
      train_model_h(model, tt_input, tt_output, -1)
    end)
    |> train_model_h(s_inputs, s_outputs, count - 1)
  end
  
  ## inputs IS A TENSOR ALL IN ONE
  defp train_model_h(model, %{__struct__: Nx.Tensor} = tt_inputs, tt_outputs, count) do
    {batch_size, _} = Nx.shape(tt_inputs)
    model = 
      model
      |> go_forward(tt_inputs)
      |> eval_and_report(model, tt_outputs)
      |> go_backwards(model, tt_outputs)
      # |> average_gradients(batch_size)
      |> learn(model, batch_size)

    if count > -1 do 
      train_model_h(model, tt_inputs, tt_outputs, count - 1)
    else
      model
    end    
  end

  ################################################################################################
  ## Private tools
  ################################################################################################

  defp go_forward(model, tt_inputs) do
    Enum.reduce(model.layers, {tt_inputs, [{tt_inputs,tt_inputs}]}, fn layer, {batch, res} ->
      z =
        batch
        |> Nx.dot(Nx.transpose(layer.weights))
        |> Nx.add(Nx.transpose(layer.biases))      
      a = apply(Activation, layer.act_func, [z])
      {a, [{z, a} | res]}
    end)
    |> elem(1)
  end

  defp go_backwards([{t_zs, t_as} | l_forwards], model, tt_outputs) do
    {batch_size, _} = Nx.shape(tt_outputs)
    act_func = model.act_funcs |> Enum.reverse() |> hd()
    deltas_outputs = 
      case act_func do
        :softmax -> Nx.subtract(t_as, tt_outputs)
        other ->
          t1 = Nx.subtract(t_as, tt_outputs)
          t2 = derivative(other, t_zs, t_as, tt_outputs)
          Nx.multiply(t1, t2) 
      end 
    
    model.layers
    |> Enum.reverse()
    |> backward(l_forwards, deltas_outputs, model.rate, batch_size)
  end

  ## return: grads_list
  defp backward(layers, l_forwards, deltas, rate, batch_size, grads_list \\ [])
  defp backward(_, [], _, _, _, grads_list), do: grads_list
  defp backward(
      [layer | layers],
      [{t_zs, t_as} | l_forwards],
      deltas,
      rate,
      batch_size,
      grads_list) do

    # deltas |> IO.inspect(label: "DELTAS")
    # t_as |> IO.inspect(label: "T_AS")
    # l_forwards |> IO.inspect(label: "FORWARD")
    # layers |> IO.inspect(label: "LAYERS")
    # layer |> IO.inspect(label: "LAYER")

    grads = {
      deltas
      |> Nx.transpose()
      |> Nx.rename([nil,nil])
      |> Nx.dot(t_as)
      |> Nx.divide(batch_size),
      deltas
    }
    
    derivative_z = derivative(layer.act_func, t_zs, t_as)
    
    deltas = 
      if length(layers) > 0 do
        deltas
        |> Nx.dot(layer.weights)        
        |> Nx.multiply(derivative_z)
      else
        nil
      end
    
    # # IO.inspect "----------------------------------------"
    backward(layers, l_forwards, deltas, rate, batch_size, [grads | grads_list])
  end

  ## return: {model, new_lt_outputs, accum_grads}
  # defp average_gradients(l_grads, batch_size) do
  #   Enum.map(l_grads, fn {gws, gbs} ->
  #     {
  #       gws |> Nx.sum(axes: [0]) |> Nx.divide(batch_size),
  #       gbs |> Nx.sum(axes: [0]) |> Nx.divide(batch_size)
  #     }
  #   end)
  # end

  ## return model
  defp learn(l_grads, model, batch_size) do
    %{layers: layers} = model
    %{model |
      layers:
        layers
        |> Enum.zip(l_grads)
        |> Enum.map(fn {layer, {gws, gbs}} ->
          ws =
            layer.weights
            |> Nx.subtract(Nx.dot(gws, model.rate))

          gbs = gbs |> Nx.sum(axes: [0]) |> Nx.divide(batch_size) |> Nx.new_axis(1)
          bs =
            layer.biases
            |> Nx.subtract(Nx.dot(gbs, model.rate))
          %{layer | weights: ws, biases: bs}
        end)        
    }
  end
  
  defp tune_model(model, opts) do
    every = Keyword.get(opts, :every, 1)
    %{model |
      rate: Keyword.get(opts, :rate, model.rate),
      batch_mode: Keyword.get(opts, :batch_mode, model.batch_mode),
      shuffle: Keyword.get(opts, :shuffle, model.shuffle),
      act_funcs: Keyword.get(opts, :act_funcs, model.act_funcs),
      map_output_func: Keyword.get(opts, :map_output_func, model.map_output_func),
      _temp: put_in(model._temp, [:every], every)
    }    
  end
  
  defp eval_and_report(
         forwards,
         %{_temp: %{count: count, total: total, every: every}} = model, 
         tt_outputs
       ) when rem(count, every) == 0 or count == total do
    output_act_func = model.act_funcs |> Enum.reverse() |> hd()
    tt_new_outputs = forwards |> hd() |> elem(1)
    cost = calc_cost(output_act_func, tt_new_outputs, tt_outputs)
    IO.puts "Iteration: #{total - count + 1} - Cost: #{cost}"
    forwards
  end

  # defp calc_cost(act_func, lt_new_outputs, lt_outputs)
  # ## for softmax
  # defp calc_cost(:softmax, lt_new_outputs, lt_outputs, _sum, _count) do
  #   a = Nx.stack(lt_new_outputs)
  #   target = Nx.stack(lt_outputs)
  #   log_a = Nx.log(Nx.add(a, 1.0e-8))
  #   product = Nx.multiply(target, log_a)
  #   sum = Nx.sum(product, axes: [1])
  #   mean = Nx.mean(sum)
  #   mean |> Nx.negate() |> Nx.to_number()
  # end
  ## for the rest
  defp calc_cost(_act_func, tt_new_outputs, tt_outputs) do
    tt_new_outputs
    |> Nx.subtract(tt_outputs)
    |> Nx.sum(axes: [1])
    |> Nx.pow(2)
    |> Nx.mean()
    |> Nx.to_number()
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
