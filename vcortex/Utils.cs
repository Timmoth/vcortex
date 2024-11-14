namespace vcortex;

public static class Utils
{
    public static string ToSignificantFigures(this float value, int significantFigures)
    {
        if (value == 0)
            return "0";

        var decimalPlaces = significantFigures - (int)Math.Floor(Math.Log10(Math.Abs(value))) - 1;
        var roundedValue = (float)Math.Round(value, decimalPlaces, MidpointRounding.AwayFromZero);

        return roundedValue.ToString("G" + significantFigures);
    }
}