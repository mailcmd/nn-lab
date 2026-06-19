
defmodule NN5 do
  defmodule Model do
	  defstruct [
      layers: nil,
      rate: nil,
      initial_rate: nil,
      act_funcs: nil,
      shape: nil,
      map_output_func: nil,
      shuffle: true,
      stop_thresold: 0,
      learning_decay: 0,
      apply_momentum: false,
      apply_gradient_clipping: false,
      stop: false,
      _temp: %{}
    ]
  end

  defmodule Layer do
	  defstruct [
      weights: nil,
      biases: nil,
      act_func: nil,
      velocities: nil
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
      Nx.tanh(z)  end

    defn none(z) do
      z
    end
  end

  @default_rate 0.1
  @default_act_func :tanh

  def test(model, inputs, outputs, report_type \\ :summary) do
    {size, _} = Nx.shape(inputs)
    fun = fn {list, hits, costs} ->
      IO.puts "HITS: #{100 * hits / size}% - AVG COST: #{costs / size}"
      if report_type == :details, do: list
    end

    mapf = model.map_output_func
    output_act_func = model.act_funcs |> Enum.reverse() |> hd()

    [
      {"expected", "rounded out", "out"}
      |
        0..size-1
        |> Enum.reduce({[], 0, 0}, fn i, {list, hits, costs} ->
          inp = Nx.take(inputs, i)
          outp = Nx.take(outputs, i)
          rv = model |> run_model(inp)
          {new_o, o} = {
            rv |> Nx.tensor() |> Nx.new_axis(0),
            outp |> Nx.as_type(:f32) |> Nx.new_axis(0)
          }
          cost = calc_cost(output_act_func, new_o, o)
          exp_out = outp |> Nx.to_list() |> mapf.()
          got_out = mapf.(rv)
          {
            [{exp_out, got_out, rv} | list],
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
  :shuffle         -> if true shuffle input in every epoch (def. true)
  :stop_thresold   -> 0
  :learning_decay  -> 0
  :apply_momentum  -> false
  :apply_gradient_clipping   -> false
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
    map_output_func = Keyword.get(opts, :map_output_func, fn l -> Enum.map(l, &round/1) end)

    %Model{
      rate: Keyword.get(opts, :rate, @default_rate),
      initial_rate: Keyword.get(opts, :initial_rate, @default_rate),
      act_funcs: act_funcs,
      shape: List.flatten([inputs, layers, outputs]),
      layers: build_model_layers(total_layers, act_funcs, outputs),
      map_output_func: map_output_func,
      stop_thresold: Keyword.get(opts, :stop_thresold, 0),
      apply_momentum: Keyword.get(opts, :apply_momentum, false),
      apply_gradient_clipping: Keyword.get(opts, :apply_gradient_clipping, false),
      learning_decay: Keyword.get(opts, :learning_decay, 0),
      _temp: %{
        count: 0,
        total: 0,
        cost: Float.max_finite(),
        every: 1
      }
    }
  end

  defp build_model_layers(layers_list, act_funcs, outputs) do
    key = Nx.Random.key(System.os_time())
    # key = Nx.Random.key(1984)
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
            act_func: act_func,
            velocities: {Nx.broadcast(0, weights), Nx.broadcast(0, biases)}
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
    |> Nx.take(0)
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
    - tt_<varname> : Tensor all in one (bidimensional tensor)
  """
  def train_model(model, inputs, outputs, count \\ 500, opts \\ [])
  def train_model(model, inputs, outputs, count, opts) do
    model
    |> tune_model(opts)
    |> update_model_params(count: count, total: count)
    |> train_model_h(inputs, outputs)
  end

  defp train_model_h(%{stop: true} = model, _, _), do: model
  defp train_model_h(%{_temp: %{count: 0}} = model, _, _), do: model
  ## inputs IS A STREAM (A BATCH)
  defp train_model_h(model, %{__struct__: Stream} = s_inputs, s_outputs) do
    output_act_func = model.act_funcs |> Enum.reverse() |> hd()
    inputs_length = s_inputs.enum.last + 1
    model = update_model_params(model, cost: 0)

    epoch_number = model._temp.total - model._temp.count

    s_inputs
    |> Stream.zip(s_outputs)
    |> Stream.with_index()
    |> Enum.reduce(model, fn {{tt_inputs, tt_outputs}, i}, model ->
      {batch_size, _} = Nx.shape(tt_inputs)
      {gradients, forwards} = train_batch(model, tt_inputs, tt_outputs)
      model = learn(model, gradients, batch_size, epoch_number)
      tt_new_outputs = forwards |> hd() |> elem(1)
      cost = calc_cost(output_act_func, tt_new_outputs, tt_outputs)
      IO.write "\rEPOCH: #{round(100 * i/inputs_length)}%"
      update_model_params(model, cost: model._temp.cost + cost)
    end)
    |> report_and_update(inputs_length)
    |> check_stop_thresold()
    |> train_model_h(s_inputs, s_outputs)
  end

  defp train_model_h(model, %{__struct__: Nx.Tensor} = tt_inputs, tt_outputs) do
    batch_size = 1
    train_model_h(
      model,
      Nx.to_batched(tt_inputs, batch_size),
      Nx.to_batched(tt_outputs, batch_size)
    )
  end

  ## inputs IS A TENSOR ALL IN ONE
  defp train_batch(model, %{__struct__: Nx.Tensor} = tt_inputs, tt_outputs) do
    forwards = go_forward(model, tt_inputs)
    gradients = go_backwards(model, forwards, tt_outputs)

    {gradients, forwards}
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

  ## receiving cost as first param skip backward
  ## return: grads_list
  defp go_backwards(cost, _, _) when is_number(cost), do: cost
  defp go_backwards(model, [{t_zs, t_as} | l_forwards], tt_outputs) do
    {batch_size, _} = Nx.shape(tt_outputs)
    act_func = model.act_funcs |> Enum.reverse() |> hd()
    deltas_outputs =
      case act_func do
        :softmax ->
          Nx.subtract(t_as, tt_outputs)
        act_func ->
          t1 = Nx.subtract(t_as, tt_outputs)
          t2 = derivative(act_func, t_zs, t_as, tt_outputs)
          Nx.multiply(t1, t2)
      end

    model.layers
    |> Enum.reverse()
    |> backward(l_forwards, deltas_outputs, model.rate, batch_size)
  end

  ## return: grads_list
  defp backward(layers, l_forwards, deltas, rate, batch_size, grads_list \\ [])
  defp backward(_, [], _, _, _, grads_list) do
    grads_list
  end
  defp backward(
      [layer | layers],
      [{t_zs, t_as} | l_forwards],
      deltas,
      rate,
      batch_size,
      grads_list) do

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

    backward(layers, l_forwards, deltas, rate, batch_size, [grads | grads_list])
  end

  ## return model
  defp learn(model, l_grads, batch_size, iteration_count) do
    %{layers: layers} = model
    %{model |
      rate: model.initial_rate * :math.pow((1 - model.learning_decay), iteration_count),
      layers:
        layers
        |> Enum.zip(l_grads)
        |> Enum.map(fn {layer, {gws, gbs}} ->
          gws =
            if model.apply_gradient_clipping do
              Nx.min(Nx.max(gws, -1), 1)
            else
              gws
            end

          gbs = gbs |> Nx.sum(axes: [0]) |> Nx.divide(batch_size) |> Nx.new_axis(1)
          gbs =
            if model.apply_gradient_clipping do
              Nx.min(Nx.max(gbs, -1), 1)
            else
              gbs
            end

          {vel_w, vel_b} = layer.velocities
          beta = 0.9
          eps = 1.0e-8

          # RMSProp for weights
          {rate, vel_w} =
            if model.apply_momentum do
              t1 = Nx.multiply(beta, vel_w)
              t2 = Nx.multiply(1-beta, Nx.pow(gws, 2))
              t3 = Nx.add(t1, t2)
              t4 = Nx.add(Nx.sqrt(t3), eps)
              {Nx.divide(model.rate, t4), t3}
            else
               {model.rate, vel_w}
            end

          ## update weights
          ws =
            layer.weights
            |> Nx.subtract(Nx.multiply(gws, rate))

          # RMSProp for biases
          {rate, vel_b} =
            if model.apply_momentum do
              t1 = Nx.multiply(beta, vel_b)
              t2 = Nx.multiply(1-beta, Nx.pow(gbs, 2))
              t3 = Nx.add(t1, t2)
              t4 = Nx.add(Nx.sqrt(t3), eps)
              {Nx.divide(model.rate, t4), t3}
            else
               {model.rate, vel_b}
            end

          ## update biases
          bs =
            layer.biases
            |> Nx.subtract(Nx.multiply(gbs, rate))

          %{layer |
            weights: ws,
            biases: bs,
            velocities: (model.apply_momentum && {vel_w, vel_b} || layer.velocities)
          }
        end)
    }
  end

  defp tune_model(model, opts) do
    every = Keyword.get(opts, :every, 1)
    %{model |
      rate: Keyword.get(opts, :rate, model.rate),
      initial_rate: Keyword.get(opts, :rate, model.rate),
      shuffle: Keyword.get(opts, :shuffle, model.shuffle),
      act_funcs: Keyword.get(opts, :act_funcs, model.act_funcs),
      map_output_func: Keyword.get(opts, :map_output_func, model.map_output_func),
      stop_thresold: Keyword.get(opts, :stop_thresold, model.stop_thresold),
      apply_momentum: Keyword.get(opts, :apply_momentum, model.apply_momentum),
      apply_gradient_clipping:
        Keyword.get(opts, :apply_gradient_clipping, model.apply_gradient_clipping),
      learning_decay: Keyword.get(opts, :learning_decay, model.learning_decay),
      _temp: put_in(model._temp, [:every], every)
    }
  end

  defp report_and_update(%{_temp: %{count: count, total: total, every: every}} = model, length)
       when rem(count, every) == 0 or count == total do
    data = model._temp
    cost = data.cost / length
    IO.puts "\rIteration: #{data.total - data.count + 1} - Cost: #{cost} - Rate: #{model.rate}"
    update_model_params(
      model,
      count: model._temp.count - 1,
      cost: model._temp.cost / length
    )
  end
  defp report_and_update(model, _), do: model

  defp check_stop_thresold(model) do
    model_cost = model._temp.cost
    %{model | stop: model_cost >= 0 and model_cost <= model.stop_thresold}
  end

  defp update_model_params(model, opts) do
    Enum.reduce(opts, model, fn {k, v}, m -> %{m | _temp: %{m._temp | k => v}} end)
  end

  ## for softmax
  defp calc_cost(:softmax, tt_new_outputs, tt_outputs) do
    tt_new_outputs
    |> Nx.add(1.0e7)
    |> Nx.log()
    |> Nx.multiply(tt_outputs)
    |> Nx.sum(axes: [1])
    |> Nx.mean()
    |> Nx.negate()
  end
  ## for the rest
  defp calc_cost(_act_func, tt_new_outputs, tt_outputs) do
    tt_new_outputs
    |> Nx.subtract(tt_outputs)
    |> Nx.pow(2)
    |> Nx.sum(axes: [1])
    |> Nx.mean()
    |> Nx.to_number()
  end

  # return {range_for_weights, range_for_biases}
  defp calc_range(:none, n_inputs, n_outputs) do
    range = :math.sqrt(6 / (n_inputs + n_outputs))
    {{-range, range}, {0, 0}}
  end
  defp calc_range(:tanh, n_inputs, n_outputs) do
    range = :math.sqrt(6 / (n_inputs + n_outputs))
    {{-range, range}, {0, 0}}
  end
  defp calc_range(:sigmoid, n_inputs, n_outputs) do
    {{rw1, rw2}, {rb1, rb2}} = calc_range(:tanh, n_inputs, n_outputs)
    {{rw1, rw2}, {rb1, rb2}}
  end
  defp calc_range(:softmax, n_inputs, n_outputs) do
    range = :math.sqrt(6 / (n_inputs + n_outputs))
    {{-range, range}, {0, 0}}
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
end
