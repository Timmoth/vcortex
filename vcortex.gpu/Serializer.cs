using ILGPU.Runtime;

namespace vcortex.gpu;

public static class Serializer
{
    public static void SaveToDisk(this INetworkAgent agent, string filePath)
    {
        using (var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write))
        using (var writer = new BinaryWriter(stream))
        {
            // Write the number of arrays to allow easy deserialization
            writer.Write(agent.Network.ParameterCount);
            var parameters = new float[agent.Network.ParameterCount];
            agent.Buffers.Parameters.View.CopyToCPU(parameters);
            foreach (var value in parameters)
            {
                writer.Write(value);
            }
        }
    }
    
    public static void ReadFromDisk(this INetworkAgent agent, string filePath)
    {
        using (var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
        using (var reader = new BinaryReader(stream))
        {
            // Read the number of arrays
            int length = reader.ReadInt32();
            var parameters = new float[length];

            for (int j = 0; j < length; j++)
            {
                parameters[j] = reader.ReadSingle();
            }
            agent.Buffers.Parameters.View.CopyFromCPU(parameters);
        }
    }
}