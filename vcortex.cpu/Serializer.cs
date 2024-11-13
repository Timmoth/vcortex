
namespace vcortex.cpu;

public static class Serializer
{
    //public static void SaveToDisk(this INetworkAgent agent, string filePath)
    //{
    //    using (var stream = new FileStream(filePath, FileMode.Create, FileAccess.Write))
    //    using (var writer = new BinaryWriter(stream))
    //    {
    //        // Write the number of arrays to allow easy deserialization
    //        writer.Write(agent.Network.NetworkData.ParameterCount);
    //        var parameters = new float[agent.Network.NetworkData.ParameterCount];
    //        agent.Buffers.Parameters.View.CopyToCPU(parameters);
    //        foreach (var value in parameters) writer.Write(value);
    //    }
    //}

    //public static void ReadFromDisk(this INetworkAgent agent, string filePath)
    //{
    //    using (var stream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
    //    using (var reader = new BinaryReader(stream))
    //    {
    //        // Read the number of arrays
    //        var length = reader.ReadInt32();
    //        var parameters = new float[length];

    //        for (var j = 0; j < length; j++) parameters[j] = reader.ReadSingle();
    //        agent.Buffers.Parameters.View.CopyFromCPU(parameters);
    //    }
    //}
}