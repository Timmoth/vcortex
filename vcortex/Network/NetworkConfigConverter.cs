using System.Text.Json;
using System.Text.Json.Serialization;
using vcortex.Input;
using vcortex.Layers;

namespace vcortex.Network;

public class NetworkConfigConverter : JsonConverter<NetworkConfig>
{
    public override NetworkConfig Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        // Deserialize into a temporary object to access `input` and `layers`
        var temp = JsonSerializer.Deserialize<NetworkConfigTemp>(ref reader, options);

        if (temp == null)
            throw new JsonException("Failed to deserialize NetworkConfig.");

        // Now use the constructor with the deserialized properties
        return new NetworkConfig(temp.Layers, temp.Input);
    }

    public override void Write(Utf8JsonWriter writer, NetworkConfig value, JsonSerializerOptions options)
    {
        // Serialize normally by passing it to the default serializer
        JsonSerializer.Serialize(writer, new NetworkConfigTemp { Layers = value.Layers, Input = value.Input }, options);
    }

    // Temporary class used to deserialize JSON properties
    private class NetworkConfigTemp
    {
        [JsonPropertyName("input")]
        public IInputConfig Input { get; set; }

        [JsonPropertyName("layers")]
        public Layer[] Layers { get; set; }
    }
}