namespace App.ViewModels;

public class HomeIndexViewModel
{
    public string SelectedExerciseId { get; set; } = "auto";

    public IReadOnlyList<ExerciseOptionViewModel> AvailableExercises { get; set; } = Array.Empty<ExerciseOptionViewModel>();

    public int TotalAnalyses { get; set; }

    public decimal? AverageScore { get; set; }

    public decimal? BestScore { get; set; }

    public decimal? LastThreshold { get; set; }

    public decimal? LastRawThreshold { get; set; }

    public decimal? LastMargin { get; set; }

    public string? FavoriteExercise { get; set; }

    public decimal? RecentImprovement { get; set; }

    public IReadOnlyList<RecentUploadViewModel> RecentUploads { get; set; } = Array.Empty<RecentUploadViewModel>();

    public AnalysisResultViewModel? LatestResult { get; set; }

    public bool HasAnalysisResult { get; set; }

    public string? AnnotatedVideoUrl { get; set; }

    public string? LatestSummary { get; set; }

    public IReadOnlyList<JointInsightViewModel> LatestJointInsights { get; set; } = Array.Empty<JointInsightViewModel>();

    // Sample videos for the selected exercise (relative URLs under /samples/{exerciseId}/...)
    public IReadOnlyList<string> SampleCorrectVideos { get; set; } = Array.Empty<string>();

    public IReadOnlyList<string> SampleIncorrectVideos { get; set; } = Array.Empty<string>();
}

public sealed record ExerciseOptionViewModel(string Id, string Name);

public class RecentUploadViewModel
{
    public string FileName { get; set; } = string.Empty;

    public string ExerciseName { get; set; } = string.Empty;

    public decimal Score { get; set; }

    public string Prediction { get; set; } = string.Empty;

    public decimal? ScoreDeltaFromPrevious { get; set; }

    public DateTimeOffset RecordedAtUtc { get; set; }

    public string Summary { get; set; } = string.Empty;
}

public class AnalysisResultViewModel
{
    public string OriginalFileName { get; set; } = string.Empty;

    public string AssessedExercise { get; set; } = string.Empty;

    public decimal Score { get; set; }

    public string Prediction { get; set; } = string.Empty;

    public string WorstJoints { get; set; } = string.Empty;

    public DateTimeOffset RecordedAtUtc { get; set; }

    public decimal? ScoreDeltaFromPrevious { get; set; }

    public decimal Threshold { get; set; }

    public decimal RawThreshold { get; set; }

    public decimal Margin { get; set; }
}

public class JointInsightViewModel
{
    public int RankIndex { get; set; }

    public string JointName { get; set; } = string.Empty;

    public int JointIndex { get; set; }

    public decimal Deviation { get; set; }

    public decimal Importance { get; set; }

    public decimal ProblemScore { get; set; }
}

public class UploadResultPageViewModel
{
    public long HistoryEntryId { get; set; }

    public string OriginalFileName { get; set; } = string.Empty;

    public string AssessedExercise { get; set; } = string.Empty;

    public decimal Score { get; set; }

    public string Prediction { get; set; } = string.Empty;

    public decimal? ScoreDeltaFromPrevious { get; set; }

    public decimal Threshold { get; set; }

    public decimal RawThreshold { get; set; }

    public decimal Margin { get; set; }

    public string Summary { get; set; } = string.Empty;

    public DateTimeOffset RecordedAtUtc { get; set; }

    public string? AnnotatedVideoUrl { get; set; }

    public IReadOnlyList<JointInsightViewModel> JointInsights { get; set; } = Array.Empty<JointInsightViewModel>();
}

public class StatisticsPageViewModel
{
    public int TotalAnalyses { get; set; }

    public decimal? AverageScore { get; set; }

    public decimal? BestScore { get; set; }

    public IReadOnlyList<ExerciseAggregateViewModel> ExerciseAggregates { get; set; } = Array.Empty<ExerciseAggregateViewModel>();

    public IReadOnlyList<ScorePointViewModel> RecentScores { get; set; } = Array.Empty<ScorePointViewModel>();

    public int TotalCorrect { get; set; }

    public int TotalIncorrect { get; set; }

    public IReadOnlyList<JointAggregateViewModel> WorstJoints { get; set; } = Array.Empty<JointAggregateViewModel>();
}

public class ExerciseAggregateViewModel
{
    public string ExerciseName { get; set; } = string.Empty;

    public int AnalysisCount { get; set; }

    public decimal AverageScore { get; set; }

    public decimal BestScore { get; set; }

    public int CorrectCount { get; set; }

    public int IncorrectCount { get; set; }
}

public class JointAggregateViewModel
{
    public string JointName { get; set; } = string.Empty;

    public int Count { get; set; }
}

public class ScorePointViewModel
{
    public string Label { get; set; } = string.Empty;

    public decimal Score { get; set; }

    public string ExerciseName { get; set; } = string.Empty;
}