defmodule NN3 do
  defmodule Model do
	  defstruct [
      layers: nil,
      eps: nil,
      rate: nil,
      act_funcs: nil,
      shape: nil,
      ranges: nil,
      map_output_func: nil
    ]
  end

  defmodule Neuron do
	  defstruct [
      weights: nil,
      bias: nil,
      connections: []
    ]
  end

  defmodule Activation do
    import Nx.Defn

    # defn softmax(z) do
    #   # shifted = z - Nx.reduce_max(z, axes: [-1], keep_axes: true)
    #   # exps = Nx.exp(shifted)
    #   # exps / Nx.sum(exps, axes: [-1], keep_axes: true)
      
    #   # z is a vector of raw scores (one per class)
    #   # subtract max for numerical stability
    #   exp_z = Nx.exp(Nx.subtract(z, Nx.reduce_max(z)))
    #   exp_z / Nx.sum(exp_z)
    # end

    defn softmax(z) do
      # z is a vector of raw scores (one per class)
      max_z = Nx.reduce_max(z)  # Not Nx.max/1
      exp_z = Nx.exp(z - max_z)  # subtract max for numerical stability
      exp_z / Nx.sum(exp_z)
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

  def test(model, inputs, outputs) do
    fun = fn {list, num} ->
      IO.puts "AVG ERROR: #{num / length(inputs)}"
      list
    end
    map = model.map_output_func
    
    [
      {"input", "expected", "rounded out", "raw out"}
      |
        inputs
        |> Enum.zip(outputs)
        |> Enum.reduce({[], 0}, fn {inp, out}, {list, costs} ->
          rv = model |> run_model(inp) |> Nx.to_list()
          cost = calc_cost(rv, out)
          exp_out = Enum.map(out, fn v -> map.(v) end)
          got_out = Enum.map(rv, fn v -> map.(v) end)
          {[{inp, exp_out, got_out, rv} | list], costs + cost}
        end)
        |> fun.()
    ]
  end

  @doc """
  act_funcs: list(:sigmoid | :tanh | :relu)  (def. :tanh)
  connections: :all | :none | list()  (def. :all)
  """
  def build_model({inputs, layers, outputs}, opts) when not is_list(layers),
      do: build_model({inputs, [layers], outputs}, opts)
  def build_model({inputs, layers, outputs}, opts) do
    act_func = Keyword.get(opts, :act_func, @default_act_func)
    output_act_func = Keyword.get(opts, :output_act_func, act_func)
    act_funcs = Enum.map(1..length(layers), fn _ -> act_func end) ++ [output_act_func]
    map_output_func = Keyword.get(opts, :map_output_func, fn v -> v end)
    
    connections =
      case Keyword.get(opts, :connections, :all) do
        :none                   -> []
        list when is_list(list) -> list
        _everything_else        -> build_all_connections(inputs, layers, outputs)
      end

    model =
      %Model{
        eps: Keyword.get(opts, :eps, @default_eps),
        rate: Keyword.get(opts, :rate, @default_rate),
        act_funcs: act_funcs,
        shape: List.flatten([inputs, layers, outputs]),
        ranges: calc_range(act_func, inputs, outputs),
        layers: [layers, outputs] |> List.flatten() |> build_model_layers(),
        map_output_func: map_output_func
      }

    update_model_connections(connections, model)
  end

  defp build_model_layers([]), do: []
  defp build_model_layers([len | shape]) do
    [
      Enum.map(0..len-1, fn _ ->
          %Neuron{
            weights: [],
            bias: nil,
            connections: []
          }
      end)
      |
      build_model_layers(shape)
    ]

  end

  def update_model_connections([], model), do: model
  def update_model_connections(connections, model) do
    # key = Nx.Random.key(System.os_time())
    key = Nx.Random.key(1972)
    %Model{layers: layers} = model
    {{w_ini, w_end}, {b_ini, b_end}} = model.ranges

    layers =
      layers
      |> Enum.with_index()
      |> Enum.reduce({key,[]}, fn {layer, i}, {key, new_layers} ->
        {
          key,
          new_layers ++ [
            layer
            |> Enum.with_index()
            |> Enum.reduce({key, []}, fn {neuron, j}, {key, new_layer} ->
              {key, new_neuron} =
                case list_get(connections, [i,j]) do
                  [] ->
                    {key, neuron}
                  conns ->
                    {weights, key} =
                      Nx.Random.uniform(key, w_ini, w_end,
                                        shape: {length(conns)}, names: [:weights])
                    {bias, key} =
                      Nx.Random.normal(key, b_ini, b_end, shape: {1}, names: [:bias])

                    {
                      key,
                      %Neuron{
                        neuron |
                          connections: list_get(connections, [i,j]),
                          weights: weights,
                          bias: bias
                      }
                    }
                end

              {
                key,
                new_layer ++ [ new_neuron ]
              }
            end)
          ]
        }
      end)
      |> elem(1)
      |> Enum.map(fn {_, l} -> l end)

    %Model{model | layers: layers}
  end

  def train_model(model, count \\ 500, inputs \\ @inputs, outputs \\ @outputs, total \\ 0)
  def train_model(model, count, inputs, outputs, _)
      when is_list(hd(inputs)) or is_list(hd(outputs)) do
    lt_inputs = Enum.map(inputs, fn inp -> Nx.tensor(inp, type: :f32) end)
    lt_outputs = Enum.map(outputs, fn outp -> Nx.tensor(outp, type: :f32) end)
    train_model(model, count, lt_inputs, lt_outputs, count)
  end

  def train_model(model, 0, _, _, _), do: model
  def train_model(model, count, lt_inputs, lt_outputs, total) do
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

  defp build_all_connections(inputs, layers, outputs),
       do: [inputs, layers, outputs] |> List.flatten() |> build_all_connections
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

  def connect(%Model{} = model, from, to) do
    %Model{layers: layers} = model
    Enum.map(layers, fn layer ->
      Enum.map(layer, fn neuron -> neuron.connections end)
    end)
    |> connect_h(model, from, to)
    |> update_model_connections(model)
  end
  def connect_h(connections, model, from, to) do
    output = "#{List.last(model.shape)},"
    target =
      to
      |> String.replace("output", output)
      |> String.replace("layer", "")
      |> String.replace("][", ",")
      |> String.replace("[", "")
      |> String.replace("]", "")
      |> String.split(",")
      |> Enum.map(&String.to_integer/1)

    from =
      from
      |> String.replace("input", "")
      |> String.replace("layer", "")
      |> String.replace("][", ",")
      |> String.replace("[", "")
      |> String.replace("]", "")
      |> String.split(",")
      |> Enum.map(&String.to_integer/1)
      |> Enum.reverse()
      |> hd()

    list_update(connections, target, fn cn -> [from | cn] |> Enum.sort() |> Enum.uniq() end)
  end

  defp back_propagation(model, lt_inputs, lt_outputs, new_lt_outputs \\ [])
  defp back_propagation(model, [], _, new_lt_outputs), do: {model, new_lt_outputs}
  defp back_propagation(model, [t_inputs | lt_inputs], [t_outputs | lt_outputs], new_lt_outputs) do
    {model, new_t_outputs} = back_propagation_h(model, t_inputs, t_outputs)
    back_propagation(model, lt_inputs, lt_outputs, new_lt_outputs ++ [new_t_outputs])
  end
  defp back_propagation_h(model, t_inputs, t_outputs) do
    %Model{layers: layers, act_funcs: act_funcs} = model

    ## First do forward operation and get all z's and a's
    ## zas -> [{z_layer_n, a_layer_n}, ..., {z_layer_2, a_layer_2}, {z_layer_1, a_layer_1}]
    {zas, new_t_outputs} =
      layers
      |> Enum.with_index()
      |> Enum.reduce({[{t_inputs, t_inputs}], t_inputs}, fn {ln_layer, i}, {r_zas, t_inputs} ->
        {_, a_output} = za = forward(ln_layer, t_inputs, :lists.nth(i+1, act_funcs))
        # IO.inspect {:lists.nth(i+1, act_funcs), za}
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

    new_model =
      layers
      |> Enum.reverse()
      |> backward(zas, deltas_o, {model.rate, r_act_funcs})
      |> learn(model)

    {new_model, new_t_outputs}
  end

  defp backward(layers, zas, deltas, model_params, result \\ [])
  defp backward(_, [], _, _, result), do: result
  defp backward(
      [ln_layer | layers],
      [{t_z, t_a} | zas],
      deltas,
      {rate, [act_func | act_funcs]},
      result) do

    # deltas |> IO.inspect(label: "DELTAS")
    # ln_layer |> IO.inspect(label: "LAYER")
    grads =
      ln_layer
      |> Enum.with_index()
      |> Enum.map(fn {n_layer, i} ->
        delta = Nx.take(deltas, i)
        {t_a |> Nx.take(Nx.tensor(n_layer.connections)) |> Nx.multiply(delta), delta}
        # {Nx.multiply(t_a, delta), delta}
      end)

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
    backward(layers, zas, deltas, {rate, act_funcs}, [grads | result])
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
            # g_ws = Nx.take(grad_ws, Nx.tensor(n_layer.connections))
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

  defp eval_and_report({model, lt_new_outputs}, lt_outputs, count, total) do
    cost = calc_cost(lt_new_outputs, lt_outputs)
    IO.puts "Iteration: #{total - count + 1} - Cost: #{cost}"
    model
  end

  defp calc_cost(lt_new_outputs, lt_outputs, sum \\ 0, count \\ 0)
  defp calc_cost([], [], sum, count), do: sum / count
  defp calc_cost([no | lt_new_outputs], [o | lt_outputs], sum, count) do
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
  defp calc_range(:tanh, n_inputs, n_outputs) do
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

  def list_get(value, []), do: value
  def list_get(list, [i | coords]) do
    :lists.nth(i+1, list) |> list_get(coords)
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
