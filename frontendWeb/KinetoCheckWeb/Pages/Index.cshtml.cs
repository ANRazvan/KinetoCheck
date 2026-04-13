using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using KinetoCheckWeb.Services;
using System.Collections.Concurrent;

namespace KinetoCheckWeb.Pages;

public class IndexModel : PageModel
{
    private static readonly ConcurrentDictionary<string, byte[]> _videoCache = new();
    private static readonly ConcurrentDictionary<string, byte[]> _inputVideoCache = new();
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
    public KinetoCheckApiClient.VideoTimelineResponse? TimelineResult { get; set; }
    public List<TimelineAnnotatedClip> TimelineAnnotatedClips { get; set; } = new();
    public string? ErrorMessage { get; set; }
    public string? VideoId { get; set; }
    public string? InputVideoId { get; set; }
    public bool HasAnnotatedVideo => !string.IsNullOrEmpty(VideoId) && _videoCache.ContainsKey(VideoId);
    public bool HasInputVideo => !string.IsNullOrEmpty(InputVideoId) && _inputVideoCache.ContainsKey(InputVideoId);

    [BindProperty]
    public string SelectedExerciseKey { get; set; } = "";

    public record TimelineAnnotatedClip(
        string VideoId,
        string ExerciseName,
        int ExerciseId,
        int StartFrame,
        int EndFrame,
        string Label,
        float Confidence
    );

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
        return dataset is "intellirehab" or "intellirehab_2d" or "uiprmd" or "uiprmd_2d";
    }

    private bool TryParseSelectedDataset(out string dataset)
    {
        dataset = "";

        if (string.IsNullOrWhiteSpace(SelectedExerciseKey))
        {
            return false;
        }

        var parts = SelectedExerciseKey.Split(':', 2, StringSplitOptions.TrimEntries);
        if (parts.Length != 2)
        {
            return false;
        }

        dataset = parts[0].ToLowerInvariant();
        return dataset is "intellirehab" or "intellirehab_2d" or "uiprmd" or "uiprmd_2d";
    }

    public async Task OnGetAsync()
    {
        await CheckBackendAsync();
    }

    public async Task<IActionResult> OnPostAsync(IFormFile? videoFile)
    {
        await CheckBackendAsync();
        TimelineAnnotatedClips = new();

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

        InputVideoId = await CacheInputVideoAsync(videoFile);

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
        TimelineAnnotatedClips = new();

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

            InputVideoId = CacheInputVideo(videoBytes);

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

    public async Task<IActionResult> OnPostAnalyzeTimelineAsync(IFormFile? videoFile)
    {
        await CheckBackendAsync();
        TimelineAnnotatedClips = new();

        if (videoFile is null || videoFile.Length == 0)
        {
            ErrorMessage = "Please select a video file.";
            return Page();
        }

        if (!TryParseSelectedDataset(out var dataset))
        {
            ErrorMessage = "Please select an exercise so the dataset can be inferred.";
            return Page();
        }

        try
        {
            using var sourceStream = videoFile.OpenReadStream();
            using var tempMemoryStream = new MemoryStream();
            await sourceStream.CopyToAsync(tempMemoryStream);
            var videoBytes = tempMemoryStream.ToArray();

            InputVideoId = CacheInputVideo(videoBytes);

            using var timelineStream = new MemoryStream(videoBytes);
            TimelineResult = await _api.PredictVideoTimelineAsync(
                timelineStream,
                videoFile.FileName,
                dataset,
                modelName: string.IsNullOrWhiteSpace(ActiveModel) ? null : ActiveModel,
                windowSize: 120,
                stride: 30,
                smoothingWindow: 5,
                minSegmentFrames: 60
            );

            if (TimelineResult is null)
            {
                ErrorMessage = "Timeline analysis returned no result.";
                return Page();
            }

            // In timeline mode, generate one annotated clip per detected segment.
            VideoId = null;
            AnnotatedVideo = null;
            foreach (var seg in TimelineResult.Segments ?? new())
            {
                using var annotatedStream = new MemoryStream(videoBytes);
                var annotated = await _api.PredictVideoAnnotatedSegmentAsync(
                    annotatedStream,
                    videoFile.FileName,
                    seg.ExerciseId,
                    dataset,
                    seg.StartFrame,
                    seg.EndFrame
                );

                if (annotated?.VideoData == null)
                {
                    continue;
                }

                var clipVideoId = Guid.NewGuid().ToString();
                _videoCache.TryAdd(clipVideoId, annotated.VideoData);

                if (_videoCache.Count > 20)
                {
                    var oldestKey = _videoCache.Keys.FirstOrDefault();
                    if (oldestKey != null) _videoCache.TryRemove(oldestKey, out _);
                }

                TimelineAnnotatedClips.Add(
                    new TimelineAnnotatedClip(
                        clipVideoId,
                        seg.ExerciseName,
                        seg.ExerciseId,
                        seg.StartFrame,
                        seg.EndFrame,
                        annotated.Label,
                        annotated.Confidence
                    )
                );

                // Keep the first generated segment clip in the legacy single-video card.
                if (VideoId == null)
                {
                    VideoId = clipVideoId;
                    AnnotatedVideo = annotated;
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Timeline analysis failed");
            ErrorMessage = $"Timeline analysis failed: {ex.Message}";
            TimelineResult = null;
        }

        return Page();
    }

    private static string CacheInputVideo(byte[] videoBytes)
    {
        var id = Guid.NewGuid().ToString();
        _inputVideoCache.TryAdd(id, videoBytes);

        // Keep cache bounded.
        if (_inputVideoCache.Count > 10)
        {
            var oldestKey = _inputVideoCache.Keys.FirstOrDefault();
            if (oldestKey != null) _inputVideoCache.TryRemove(oldestKey, out _);
        }

        return id;
    }

    private static async Task<string> CacheInputVideoAsync(IFormFile videoFile)
    {
        using var sourceStream = videoFile.OpenReadStream();
        using var memory = new MemoryStream();
        await sourceStream.CopyToAsync(memory);
        return CacheInputVideo(memory.ToArray());
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

        return new FileContentResult(videoBytes, "video/mp4")
        {
            EnableRangeProcessing = true,
        };
    }

    public IActionResult OnGetStreamInputVideo(string videoId)
    {
        if (string.IsNullOrEmpty(videoId) || !_inputVideoCache.TryGetValue(videoId, out var videoBytes))
        {
            return NotFound();
        }

        return new FileContentResult(videoBytes, "video/mp4")
        {
            EnableRangeProcessing = true,
        };
    }

    public async Task<IActionResult> OnGetReferenceVideoAsync(string dataset, int exerciseId)
    {
        if (string.IsNullOrWhiteSpace(dataset))
        {
            return BadRequest("Missing dataset.");
        }

        try
        {
            var videoBytes = await _api.GetReferenceVisualizationAsync(dataset, exerciseId);
            return new FileContentResult(videoBytes, "video/mp4")
            {
                EnableRangeProcessing = true,
            };
        }
        catch (HttpRequestException ex)
        {
            _logger.LogWarning(
                ex,
                "Reference video fetch failed for dataset {Dataset}, exercise {ExerciseId}",
                dataset,
                exerciseId
            );
            return NotFound();
        }
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
