namespace App.Models;

public class AnalysisHistoryEntry
{
    public long Id { get; set; }

    public string? UserId { get; set; }

    public ApplicationUser? User { get; set; }

    public long UploadId { get; set; }

    public Upload Upload { get; set; } = null!;

    public long? StatisticsId { get; set; }

    public UploadStatistics? Statistics { get; set; }

    public string OriginalFileName { get; set; } = string.Empty;

    public string ExerciseName { get; set; } = string.Empty;

    public string PredictedLabel { get; set; } = string.Empty;

    public decimal Score { get; set; }

    public decimal Threshold { get; set; }

    public decimal RawThreshold { get; set; }

    public decimal Margin { get; set; }

    public decimal? ScoreDeltaFromPrevious { get; set; }

    public string Summary { get; set; } = string.Empty;

    public DateTimeOffset RecordedAtUtc { get; set; }
}