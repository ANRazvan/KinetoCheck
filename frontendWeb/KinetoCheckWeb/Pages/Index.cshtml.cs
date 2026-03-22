using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using KinetoCheckWeb.Services;
using System.Collections.Concurrent;

namespace KinetoCheckWeb.Pages;

public class IndexModel : PageModel
{
    private static readonly ConcurrentDictionary<string, byte[]> _videoCache = new();
    private readonly KinetoCheckApiClient _api;
    private readonly IConfiguration _config;
    private readonly ILogger<IndexModel> _logger;

    public IndexModel(KinetoCheckApiClient api, IConfiguration config, ILogger<IndexModel> logger)
    {
        _api = api;
        _config = config;
        _logger = logger;
    }

    // ── View properties ──
    public bool BackendOnline { get; set; }
    public string BackendVersion { get; set; } = "";
    public string ActiveModel { get; set; } = "";
    public string BackendUrl => _config["BackendUrl"] ?? "http://localhost:8000";

    public List<KinetoCheckApiClient.ExerciseInfo> Exercises { get; set; } = new();
    public KinetoCheckApiClient.PredictionResponse? Prediction { get; set; }
    public KinetoCheckApiClient.AnnotatedVideoResult? AnnotatedVideo { get; set; }
    public string? ErrorMessage { get; set; }
    public string? VideoId { get; set; }
    public bool HasAnnotatedVideo => !string.IsNullOrEmpty(VideoId) && _videoCache.ContainsKey(VideoId);

    [BindProperty]
    public string SelectedExerciseKey { get; set; } = "";

    private bool TryParseSelectedExercise(out string dataset, out int exerciseId)
    {
        dataset = "";
        exerciseId = -1;

        if (string.IsNullOrWhiteSpace(SelectedExerciseKey))
        {
            return false;
        }

        var parts = SelectedExerciseKey.Split(':', 2, StringSplitOptions.TrimEntries);
        if (parts.Length != 2)
        {
            return false;
        }

        if (!int.TryParse(parts[1], out exerciseId))
        {
            return false;
        }

        dataset = parts[0].ToLowerInvariant();
        return dataset is "intellirehab" or "uiprmd";
    }

    public async Task OnGetAsync()
    {
        await CheckBackendAsync();
    }

    public async Task<IActionResult> OnPostAsync(IFormFile? videoFile)
    {
        await CheckBackendAsync();

        if (videoFile is null || videoFile.Length == 0)
        {
            ErrorMessage = "Please select a video file.";
            return Page();
        }

        if (!TryParseSelectedExercise(out var dataset, out var exerciseId))
        {
            ErrorMessage = "Please select an exercise from the list.";
            return Page();
        }

        try
        {
            using var stream = videoFile.OpenReadStream();
            Prediction = await _api.PredictVideoAsync(stream, videoFile.FileName, exerciseId, dataset);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Prediction failed");
            ErrorMessage = $"Prediction failed: {ex.Message}";
        }

        return Page();
    }

    public async Task<IActionResult> OnPostAnalyzeWithAnnotatedVideoAsync(IFormFile? videoFile)
    {
        await CheckBackendAsync();

        if (videoFile is null || videoFile.Length == 0)
        {
            ErrorMessage = "Please select a video file.";
            return Page();
        }

        if (!TryParseSelectedExercise(out var dataset, out var exerciseId))
        {
            ErrorMessage = "Please select an exercise from the list.";
            return Page();
        }

        try
        {
            // Read video file into byte array so we can use it for both API calls
            using var sourceStream = videoFile.OpenReadStream();
            using var tempMemoryStream = new MemoryStream();
            await sourceStream.CopyToAsync(tempMemoryStream);
            var videoBytes = tempMemoryStream.ToArray();

            // Get annotated video (with separate stream)
            using (var annotatedStream = new MemoryStream(videoBytes))
            {
                AnnotatedVideo = await _api.PredictVideoAnnotatedAsync(annotatedStream, videoFile.FileName, exerciseId, dataset);
                _logger.LogInformation($"AnnotatedVideo received: {AnnotatedVideo?.VideoData?.Length ?? 0} bytes");
                
                // Store video in cache with unique ID
                if (AnnotatedVideo?.VideoData != null)
                {
                    VideoId = Guid.NewGuid().ToString();
                    _videoCache.TryAdd(VideoId, AnnotatedVideo.VideoData);
                    
                    // Clean up old videos (keep only last 10)
                    if (_videoCache.Count > 10)
                    {
                        var oldestKey = _videoCache.Keys.FirstOrDefault();
                        if (oldestKey != null) _videoCache.TryRemove(oldestKey, out _);
                    }
                }
            }
            
            // Also get full prediction details with JSON endpoint for deviations (with separate stream)
            using (var predictionStream = new MemoryStream(videoBytes))
            {
                Prediction = await _api.PredictVideoAsync(predictionStream, videoFile.FileName, exerciseId, dataset);
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Annotated video generation failed");
            ErrorMessage = $"Analysis failed: {ex.Message}";
            AnnotatedVideo = null;
            VideoId = null;
        }

        return Page();
    }

    public IActionResult OnGetDownloadAnnotatedVideo(string videoId)
    {
        if (string.IsNullOrEmpty(videoId) || !_videoCache.TryGetValue(videoId, out var videoBytes))
        {
            return NotFound();
        }

        return File(videoBytes, "video/mp4", "annotated_movement.mp4");
    }

    public IActionResult OnGetStreamAnnotatedVideo(string videoId)
    {
        if (string.IsNullOrEmpty(videoId) || !_videoCache.TryGetValue(videoId, out var videoBytes))
        {
            return NotFound();
        }

        return File(videoBytes, "video/mp4");
    }

    private async Task CheckBackendAsync()
    {
        try
        {
            var health = await _api.HealthCheckAsync();
            BackendOnline = health?.Status == "ok";
            BackendVersion = health?.Version ?? "";

            var models = await _api.GetModelsAsync();
            ActiveModel = models?.ActiveModel ?? "";

            var exercises = await _api.GetExercisesAsync();
            Exercises = exercises ?? new();
        }
        catch
        {
            BackendOnline = false;
        }
    }
}
