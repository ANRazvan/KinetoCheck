namespace App.Models;

public class JointInsight
{
    public long Id { get; set; }

    public long UploadStatisticsId { get; set; }

    public UploadStatistics UploadStatistics { get; set; } = null!;

    public int RankIndex { get; set; }

    public string JointName { get; set; } = string.Empty;

    public int JointIndex { get; set; }

    public decimal Deviation { get; set; }

    public decimal Importance { get; set; }

    public decimal ProblemScore { get; set; }
}