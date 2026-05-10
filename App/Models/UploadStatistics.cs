namespace App.Models;

public class UploadStatistics
{
    public long Id { get; set; }

    public long UploadId { get; set; }

    public Upload Upload { get; set; } = null!;

    public decimal Score { get; set; }

    public decimal Threshold { get; set; }

    public decimal RawThreshold { get; set; }

    public decimal Margin { get; set; }

    public string PredictedLabel { get; set; } = string.Empty;

    public string AssessedExerciseName { get; set; } = string.Empty;

    public string WorstJointsJson { get; set; } = "[]";

    public DateTimeOffset AnalyzedAtUtc { get; set; }

    public ICollection<JointInsight> JointInsights { get; set; } = new List<JointInsight>();
}