using System.Diagnostics;
using System.Text.Json;
using App.Data;
using App.Models;
using App.ViewModels;
using System.Text.RegularExpressions;
using Microsoft.AspNetCore.Identity;
using Microsoft.AspNetCore.Mvc;
using Microsoft.EntityFrameworkCore;

namespace App.Controllers;

public class HomeController : Controller
{
    private static readonly IReadOnlyList<ExerciseOptionViewModel> ExerciseOptions = new[]
    {
        new ExerciseOptionViewModel("auto", "System auto-detect"),
        new ExerciseOptionViewModel("1", "Exercise 01"),
        new ExerciseOptionViewModel("2", "Exercise 02"),
        new ExerciseOptionViewModel("3", "Exercise 03"),
        new ExerciseOptionViewModel("4", "Exercise 04"),
        new ExerciseOptionViewModel("5", "Exercise 05"),
        new ExerciseOptionViewModel("6", "Exercise 06"),
        new ExerciseOptionViewModel("7", "Exercise 07"),
        new ExerciseOptionViewModel("8", "Exercise 08"),
        new ExerciseOptionViewModel("9", "Exercise 09"),
        new ExerciseOptionViewModel("10", "Exercise 10")
    };

    private readonly ILogger<HomeController> _logger;
    private readonly IWebHostEnvironment _environment;
    private readonly AppDbContext _dbContext;
    private readonly IHttpClientFactory _httpClientFactory;
    private readonly UserManager<ApplicationUser> _userManager;

    public HomeController(
        ILogger<HomeController> logger,
        IWebHostEnvironment environment,
        AppDbContext dbContext,
        IHttpClientFactory httpClientFactory,
        UserManager<ApplicationUser> userManager)
    {
        _logger = logger;
        _environment = environment;
        _dbContext = dbContext;
        _httpClientFactory = httpClientFactory;
        _userManager = userManager;
    }

    public async Task<IActionResult> Index(string? exercise_id)
    {
        var selected = string.IsNullOrWhiteSpace(exercise_id) ? "auto" : exercise_id;
        var model = await BuildDashboardViewModelAsync(selected);
        return View(model);
    }

    [HttpPost]
    [ValidateAntiForgeryToken]
    public async Task<IActionResult> UploadVideo(IFormFile videoFile, string exercise_id)
    {
        var selectedExerciseId = string.IsNullOrWhiteSpace(exercise_id) ? "auto" : exercise_id;
        var model = await BuildDashboardViewModelAsync(selectedExerciseId);

        _logger.LogInformation("Selected exercise ID: {ExerciseId}", selectedExerciseId);
        string tmpFolder = Path.Combine(Path.GetTempPath(), "kinetocheck_uploads");
        Directory.CreateDirectory(tmpFolder);

        if (videoFile == null || videoFile.Length == 0)
        {
            ModelState.AddModelError(string.Empty, "Please select a video file to upload.");
            return View("Index", model);
        }

        var tmpFile = Path.Combine(tmpFolder, $"{Guid.NewGuid():N}_{Path.GetFileName(videoFile.FileName)}");
        await using (var stream = System.IO.File.Create(tmpFile))
        {
            await videoFile.CopyToAsync(stream);
        }

        try
        {
            using var client = _httpClientFactory.CreateClient("AnalysisService");
            using var content = new MultipartFormDataContent();
            await using var fs = System.IO.File.OpenRead(tmpFile);
            content.Add(new StreamContent(fs), "video", Path.GetFileName(tmpFile));
            content.Add(new StringContent(selectedExerciseId), "exercise_id");

            HttpResponseMessage resp;
            try
            {
                resp = await client.PostAsync("analyze-video/", content);
            }
            catch (Exception ex)
            {
                ModelState.AddModelError(string.Empty, "Failed to contact analysis service: " + ex.Message);
                return View("Index", model);
            }

            if (!resp.IsSuccessStatusCode)
            {
                var err = await resp.Content.ReadAsStringAsync();
                ModelState.AddModelError(string.Empty, "Analysis failed: " + err);
                return View("Index", model);
            }

            var json = await resp.Content.ReadAsStringAsync();
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            var currentUserId = _userManager.GetUserId(User);
            var analyzedAtUtc = DateTimeOffset.UtcNow;
            var originalFileName = Path.GetFileName(tmpFile);
            var assessedExercise = GetStringOrDefault(root, "best", "exercise_name") ?? selectedExerciseId;
            var predictedLabel = GetStringOrDefault(root, "best", "predicted_label") ?? "Unknown";
            var score = GetDecimalOrDefault(root, "best", "score");
            var threshold = GetDecimalOrDefault(root, "best", "threshold");
            var rawThreshold = GetDecimalOrDefault(root, "best", "raw_threshold");
            var margin = GetDecimalOrDefault(root, "best", "margin");
            var worstJointsJson = root.TryGetProperty("worst_joints", out var worstJoints)
                ? worstJoints.GetRawText()
                : "[]";
            var jointInsights = ParseJointInsights(root);
            var worstJointsSummary = SummarizeWorstJoints(jointInsights);

            var previousHistory = await _dbContext.AnalysisHistoryEntries
                .AsNoTracking()
                .Where(entry => entry.UserId == currentUserId && entry.ExerciseName == assessedExercise)
                .OrderByDescending(entry => entry.RecordedAtUtc)
                .FirstOrDefaultAsync();

            var upload = new Upload
            {
                UserId = currentUserId,
                OriginalFileName = originalFileName,
                SelectedExerciseId = selectedExerciseId,
                FileSizeBytes = videoFile.Length,
                UploadedAtUtc = analyzedAtUtc
            };

            var statistics = new UploadStatistics
            {
                Score = score,
                Threshold = threshold,
                RawThreshold = rawThreshold,
                Margin = margin,
                PredictedLabel = predictedLabel,
                AssessedExerciseName = assessedExercise,
                WorstJointsJson = worstJointsJson,
                AnalyzedAtUtc = analyzedAtUtc,
                Upload = upload
            };

            var scoreDelta = previousHistory == null ? (decimal?)null : score - previousHistory.Score;
            var historyEntry = new AnalysisHistoryEntry
            {
                UserId = currentUserId,
                Upload = upload,
                Statistics = statistics,
                OriginalFileName = originalFileName,
                ExerciseName = assessedExercise,
                PredictedLabel = predictedLabel,
                Score = score,
                ScoreDeltaFromPrevious = scoreDelta,
                Threshold = threshold,
                RawThreshold = rawThreshold,
                Margin = margin,
                Summary = BuildHistorySummary(assessedExercise, score, scoreDelta, predictedLabel),
                RecordedAtUtc = analyzedAtUtc
            };

            foreach (var jointInsight in jointInsights)
            {
                statistics.JointInsights.Add(new JointInsight
                {
                    RankIndex = jointInsight.RankIndex,
                    JointName = jointInsight.JointName,
                    JointIndex = jointInsight.JointIndex,
                    Deviation = jointInsight.Deviation,
                    Importance = jointInsight.Importance,
                    ProblemScore = jointInsight.ProblemScore
                });
            }

            _dbContext.AnalysisHistoryEntries.Add(historyEntry);
            await _dbContext.SaveChangesAsync();

            var dashboard = await BuildDashboardViewModelAsync(selectedExerciseId, new AnalysisResultViewModel
            {
                OriginalFileName = originalFileName,
                AssessedExercise = assessedExercise,
                Score = score,
                Prediction = predictedLabel,
                WorstJoints = worstJointsSummary,
                RecordedAtUtc = analyzedAtUtc,
                ScoreDeltaFromPrevious = scoreDelta,
                Threshold = threshold,
                RawThreshold = rawThreshold,
                Margin = margin
            });

            dashboard.HasAnalysisResult = true;
            dashboard.AnnotatedVideoUrl = CopyAnnotatedVideo(root);
            dashboard.LatestSummary = historyEntry.Summary;
            dashboard.LastThreshold = threshold;
            dashboard.LastRawThreshold = rawThreshold;
            dashboard.LastMargin = margin;
            dashboard.LatestJointInsights = jointInsights.Select(joint => new JointInsightViewModel
            {
                RankIndex = joint.RankIndex,
                JointName = joint.JointName,
                JointIndex = joint.JointIndex,
                Deviation = joint.Deviation,
                Importance = joint.Importance,
                ProblemScore = joint.ProblemScore
            }).ToList();

            // Preserve annotated video url across redirect and go to result page
            TempData["AnnotatedVideoUrl"] = dashboard.AnnotatedVideoUrl ?? string.Empty;
            return RedirectToAction("Result", new { id = historyEntry.Id });
        }
        catch (Exception ex)
        {
            ModelState.AddModelError(string.Empty, "Failed to parse results: " + ex.Message);
            return View("Index", model);
        }
        finally
        {
            try
            {
                if (System.IO.File.Exists(tmpFile))
                {
                    System.IO.File.Delete(tmpFile);
                }
            }
            catch (Exception cleanupEx)
            {
                _logger.LogWarning(cleanupEx, "Failed to delete temporary upload file {TempFile}", tmpFile);
            }
        }
    }

    public IActionResult Privacy()
    {
        return View();
    }

    [HttpGet]
    public async Task<IActionResult> Result(long id)
    {
        var entry = await _dbContext.AnalysisHistoryEntries
            .AsNoTracking()
            .Include(e => e.Statistics)
                .ThenInclude(s => s.JointInsights)
            .Include(e => e.Upload)
            .FirstOrDefaultAsync(e => e.Id == id);

        if (entry == null)
        {
            return NotFound();
        }

        var vm = new App.ViewModels.UploadResultPageViewModel
        {
            HistoryEntryId = entry.Id,
            OriginalFileName = entry.OriginalFileName,
            AssessedExercise = entry.ExerciseName,
            Score = entry.Score,
            Prediction = entry.PredictedLabel,
            ScoreDeltaFromPrevious = entry.ScoreDeltaFromPrevious,
            Threshold = entry.Threshold,
            RawThreshold = entry.RawThreshold,
            Margin = entry.Margin,
            Summary = entry.Summary,
            RecordedAtUtc = entry.RecordedAtUtc,
            AnnotatedVideoUrl = (TempData["AnnotatedVideoUrl"] as string) ?? null,
            JointInsights = entry.Statistics?.JointInsights.Select(j => new App.ViewModels.JointInsightViewModel
            {
                RankIndex = j.RankIndex,
                JointName = j.JointName,
                JointIndex = j.JointIndex,
                Deviation = j.Deviation,
                Importance = j.Importance,
                ProblemScore = j.ProblemScore
            }).ToList() ?? new List<App.ViewModels.JointInsightViewModel>()
        };

        return View(vm);
    }

    [HttpGet]
    public async Task<IActionResult> Statistics()
    {
        var total = await _dbContext.AnalysisHistoryEntries.CountAsync();
        var avg = await _dbContext.AnalysisHistoryEntries.AnyAsync()
            ? await _dbContext.AnalysisHistoryEntries.AverageAsync(e => e.Score)
            : (decimal?)null;

        var aggregates = await _dbContext.AnalysisHistoryEntries
            .GroupBy(e => e.ExerciseName)
            .Select(g => new App.ViewModels.ExerciseAggregateViewModel
            {
                ExerciseName = g.Key,
                AnalysisCount = g.Count(),
                AverageScore = g.Average(x => x.Score),
                BestScore = g.Max(x => x.Score),
                CorrectCount = g.Count(x => x.Score >= x.Threshold),
                IncorrectCount = g.Count(x => x.Score < x.Threshold)
            })
            .OrderByDescending(a => a.AnalysisCount)
            .ToListAsync();

        var recent = await _dbContext.AnalysisHistoryEntries
            .AsNoTracking()
            .OrderByDescending(e => e.RecordedAtUtc)
            .Take(20)
            .Select(e => new App.ViewModels.ScorePointViewModel
            {
                Label = e.RecordedAtUtc.ToLocalTime().ToString("MM-dd"),
                Score = e.Score,
                ExerciseName = e.ExerciseName
            })
            .ToListAsync();

        var totalCorrect = await _dbContext.AnalysisHistoryEntries.CountAsync(e => e.Score >= e.Threshold);
        var totalIncorrect = total - totalCorrect;

        var jointCounts = await (from ji in _dbContext.JointInsights
                                 join h in _dbContext.AnalysisHistoryEntries on ji.UploadStatisticsId equals h.StatisticsId
                                 group ji by ji.JointName into g
                                 select new App.ViewModels.JointAggregateViewModel
                                 {
                                     JointName = g.Key,
                                     Count = g.Count()
                                 })
                                .OrderByDescending(j => j.Count)
                                .ToListAsync();

        var vm = new App.ViewModels.StatisticsPageViewModel
        {
            TotalAnalyses = total,
            AverageScore = avg,
            BestScore = aggregates.Any() ? aggregates.Max(a => a.BestScore) : (decimal?)null,
            ExerciseAggregates = aggregates,
            RecentScores = recent,
            TotalCorrect = totalCorrect,
            TotalIncorrect = totalIncorrect,
            WorstJoints = jointCounts
        };

        return View(vm);
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }

    private async Task<HomeIndexViewModel> BuildDashboardViewModelAsync(string selectedExerciseId, AnalysisResultViewModel? latestResult = null)
    {
        var currentUserId = _userManager.GetUserId(User);
        var query = _dbContext.AnalysisHistoryEntries
            .AsNoTracking()
            .OrderByDescending(entry => entry.RecordedAtUtc)
            .Include(entry => entry.Upload)
            .Include(entry => entry.Statistics)
            .AsQueryable();

        query = string.IsNullOrWhiteSpace(currentUserId)
            ? query.Where(entry => entry.UserId == null)
            : query.Where(entry => entry.UserId == currentUserId);

        var history = await query.Take(5).ToListAsync();

        var scoreQuery = _dbContext.AnalysisHistoryEntries.AsNoTracking();
        scoreQuery = string.IsNullOrWhiteSpace(currentUserId)
            ? scoreQuery.Where(entry => entry.UserId == null)
            : scoreQuery.Where(entry => entry.UserId == currentUserId);

        var scores = await scoreQuery.Select(entry => entry.Score).ToListAsync();
        var totalAnalyses = scores.Count;
        var averageScore = totalAnalyses == 0 ? (decimal?)null : scores.Average();
        var bestScore = totalAnalyses == 0 ? (decimal?)null : scores.Max();
        var recentImprovement = history.FirstOrDefault()?.ScoreDeltaFromPrevious;
        var favoriteExercise = history
            .GroupBy(entry => entry.ExerciseName)
            .OrderByDescending(group => group.Count())
            .Select(group => group.Key)
            .FirstOrDefault();

        var vm = new HomeIndexViewModel
        {
            SelectedExerciseId = selectedExerciseId,
            AvailableExercises = ExerciseOptions,
            TotalAnalyses = totalAnalyses,
            AverageScore = averageScore,
            BestScore = bestScore,
            FavoriteExercise = favoriteExercise,
            RecentImprovement = recentImprovement,
            RecentUploads = history.Select(entry => new RecentUploadViewModel
            {
                FileName = entry.OriginalFileName,
                ExerciseName = entry.ExerciseName,
                Score = entry.Score,
                Prediction = entry.PredictedLabel,
                ScoreDeltaFromPrevious = entry.ScoreDeltaFromPrevious,
                RecordedAtUtc = entry.RecordedAtUtc,
                Summary = entry.Summary
            }).ToList(),
            LatestResult = latestResult
        };

        // Populate sample video URLs by scanning wwwroot/samples for filenames like exercise_01_correct.mp4
        try
        {
            var samplesRoot = Path.Combine(_environment.WebRootPath, "samples");
            List<string> correct = new();
            List<string> incorrect = new();

            // Only enumerate samples when a concrete exercise is selected
            if (!string.Equals(selectedExerciseId, "auto", StringComparison.OrdinalIgnoreCase) && Directory.Exists(samplesRoot))
            {
                var files = Directory.EnumerateFiles(samplesRoot, "*.*", SearchOption.AllDirectories)
                    .Where(p => p.EndsWith(".mp4", StringComparison.OrdinalIgnoreCase) || p.EndsWith(".webm", StringComparison.OrdinalIgnoreCase) || p.EndsWith(".mov", StringComparison.OrdinalIgnoreCase))
                    .OrderBy(p => p);

                var rx = new Regex("exercise[_-]?0*([0-9]+)", RegexOptions.IgnoreCase | RegexOptions.Compiled);

                var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                var webDir = Path.Combine(samplesRoot, "web");

                foreach (var f in files)
                {
                    // skip files in the web cache directory to avoid duplicates
                    if (!string.IsNullOrEmpty(webDir) && f.StartsWith(webDir + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
                    {
                        continue;
                    }

                    var fn = Path.GetFileName(f);
                    if (seen.Contains(fn)) continue;
                    seen.Add(fn);

                    var m = rx.Match(fn);
                    if (!m.Success)
                    {
                        continue;
                    }

                    var num = m.Groups[1].Value; // e.g. "1" or "01"
                    // match selectedExerciseId (allow "auto" to show nothing)
                    if (!string.Equals(selectedExerciseId, "auto", StringComparison.OrdinalIgnoreCase) && num != selectedExerciseId)
                    {
                        // also allow numeric compare (e.g. selected 1 vs filename 01)
                        if (!int.TryParse(num, out var parsed) || parsed.ToString() != selectedExerciseId)
                        {
                            continue;
                        }
                    }

                    var url = "/samples/preview/" + Uri.EscapeDataString(fn);

                    var fnameLower = fn.ToLowerInvariant();
                    // Prefer exact 'incorrect' match first because 'incorrect' contains 'correct'
                    if (fnameLower.Contains("_incorrect") || fnameLower.Contains("-incorrect") || fnameLower.Contains("incorrect"))
                    {
                        incorrect.Add(url);
                    }
                    else if (fnameLower.Contains("_correct") || fnameLower.Contains("-correct") || fnameLower.Contains("correct"))
                    {
                        correct.Add(url);
                    }
                }
            }

            vm.SampleCorrectVideos = correct;
            vm.SampleIncorrectVideos = incorrect;
        }
        catch (Exception ex)
        {
            _logger.LogDebug(ex, "Failed to enumerate sample videos");
        }

        return vm;
    }
    private string? CopyAnnotatedVideo(JsonElement root)
    {
        if (!root.TryGetProperty("annotated_video", out var annotatedVideo))
        {
            return null;
        }

        var sourcePath = annotatedVideo.GetString();
        if (string.IsNullOrWhiteSpace(sourcePath) || !System.IO.File.Exists(sourcePath))
        {
            return null;
        }

        var sessionId = root.TryGetProperty("session_id", out var sid) ? sid.GetString() : null;
        var normalizedSessionId = string.IsNullOrWhiteSpace(sessionId) ? Guid.NewGuid().ToString("N") : sessionId;
        var uploadDir = Path.Combine(_environment.WebRootPath, "uploads", normalizedSessionId);
        Directory.CreateDirectory(uploadDir);

        var fileName = Path.GetFileName(sourcePath);
        var destinationPath = Path.Combine(uploadDir, fileName);
        System.IO.File.Copy(sourcePath, destinationPath, overwrite: true);

        return $"/uploads/{normalizedSessionId}/{fileName}";
    }

    private static string? GetStringOrDefault(JsonElement root, string parentProperty, string childProperty)
    {
        if (root.TryGetProperty(parentProperty, out var parent) && parent.TryGetProperty(childProperty, out var value))
        {
            return value.GetString();
        }

        return null;
    }

    private static decimal GetDecimalOrDefault(JsonElement root, string parentProperty, string childProperty)
    {
        if (root.TryGetProperty(parentProperty, out var parent) && parent.TryGetProperty(childProperty, out var value))
        {
            return (decimal)value.GetDouble();
        }

        return 0m;
    }

    private static string SummarizeWorstJoints(JsonElement root)
    {
        if (!root.TryGetProperty("worst_joints", out var worstJoints) || worstJoints.ValueKind != JsonValueKind.Array)
        {
            return "No joint breakdown available.";
        }

        var joints = new List<string>();
        foreach (var joint in worstJoints.EnumerateArray().Take(3))
        {
            if (joint.TryGetProperty("joint", out var jointName))
            {
                joints.Add(jointName.GetString() ?? "unknown");
            }
        }

        return joints.Count == 0 ? "No joint breakdown available." : string.Join(", ", joints);
    }

    private static string SummarizeWorstJoints(IEnumerable<JointInsightViewModel> jointInsights)
    {
        var names = jointInsights
            .OrderBy(joint => joint.RankIndex)
            .Take(3)
            .Select(joint => joint.JointName)
            .ToList();

        return names.Count == 0 ? "No joint breakdown available." : string.Join(", ", names);
    }

    private static List<JointInsightViewModel> ParseJointInsights(JsonElement root)
    {
        if (!root.TryGetProperty("worst_joints", out var worstJoints) || worstJoints.ValueKind != JsonValueKind.Array)
        {
            return new List<JointInsightViewModel>();
        }

        var insights = new List<JointInsightViewModel>();
        var rankIndex = 1;

        foreach (var item in worstJoints.EnumerateArray())
        {
            var jointName = item.TryGetProperty("joint", out var jointValue) ? jointValue.GetString() ?? "unknown" : "unknown";
            var jointIndex = item.TryGetProperty("joint_index", out var indexValue) ? indexValue.GetInt32() : -1;
            var deviation = item.TryGetProperty("deviation", out var deviationValue) ? (decimal)deviationValue.GetDouble() : 0m;
            var importance = item.TryGetProperty("importance", out var importanceValue) ? (decimal)importanceValue.GetDouble() : 0m;
            var problemScore = item.TryGetProperty("problem_score", out var problemScoreValue) ? (decimal)problemScoreValue.GetDouble() : 0m;

            insights.Add(new JointInsightViewModel
            {
                RankIndex = rankIndex++,
                JointName = jointName,
                JointIndex = jointIndex,
                Deviation = deviation,
                Importance = importance,
                ProblemScore = problemScore
            });
        }

        return insights;
    }

    private static string BuildHistorySummary(string exerciseName, decimal score, decimal? scoreDelta, string prediction)
    {
        var deltaText = scoreDelta.HasValue
            ? scoreDelta.Value >= 0
                ? $"Up {scoreDelta.Value:F3} from the previous result."
                : $"Down {Math.Abs(scoreDelta.Value):F3} from the previous result."
            : "This is your first recorded result for this exercise.";

        return $"{exerciseName} scored {score:F3} with {prediction}. {deltaText}";
    }
}
