namespace App.Models;

public class Upload
{
    public long Id { get; set; }

    public string? UserId { get; set; }

    public ApplicationUser? User { get; set; }

    public string OriginalFileName { get; set; } = string.Empty;

    public string SelectedExerciseId { get; set; } = string.Empty;

    public long FileSizeBytes { get; set; }

    public DateTimeOffset UploadedAtUtc { get; set; }

    public UploadStatistics? Statistics { get; set; }

    public ICollection<AnalysisHistoryEntry> HistoryEntries { get; set; } = new List<AnalysisHistoryEntry>();
}