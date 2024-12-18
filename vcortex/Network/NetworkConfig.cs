﻿using System.Text.Json;
using System.Text.Json.Serialization;
using vcortex.Input;
using vcortex.Layers;

namespace vcortex.Network;

public class NetworkConfig
{
    [JsonPropertyName("input")]
    public IInputConfig Input { get; set; }

    [JsonPropertyName("layers")]
    public Layer[] Layers { get; set; }

    [JsonIgnore]
    public readonly NetworkData NetworkData;

    [JsonIgnore]
    public readonly int OutputCount;
    
    public NetworkConfig(Layer[] layers, IInputConfig input)
    {
        Layers = layers;
        Input = input;

        layers[0].Connect(input);
        for (var i = 1; i < Layers.Length; i++) layers[i].Connect(layers[i - 1]);

        OutputCount = layers[^1].NumOutputs;
        var activationCount = layers.Sum(l => l.NumOutputs) + Layers[0].NumInputs;
        var parameterCount = layers.Sum(l => l.ParameterCount);
        NetworkData = new NetworkData(activationCount, parameterCount);
    }

    #region Io

    
    public static NetworkConfig DeserializeFromDisk(string filepath)
    {
        var jsonContent = File.ReadAllText(filepath);
        return JsonSerializer.Deserialize<NetworkConfig>(jsonContent, Utils.SerializerOptions) ?? throw new Exception("Failed to deserialize network config");
    }
    
    public void SerializeToDisk(string filepath)
    {
        var jsonContent = JsonSerializer.Serialize(this, Utils.SerializerOptions);
        File.WriteAllText(filepath, jsonContent);
    }
    
    #endregion
   
    
}