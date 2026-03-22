using System.Net.Http.Headers;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace KinetoCheckWeb.Services;

/// <summary>
/// Typed HTTP client that talks to the KinetoCheck FastAPI backend.
/// </summary>
public class KinetoCheckApiClient
{
    private readonly HttpClient _http;
    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public KinetoCheckApiClient(HttpClient http)
    {
        _http = http;
    }

    // ---------- DTOs ----------

    public record HealthResponse(string Status, string Version);

    public record ExerciseInfo(string Dataset, int Id, string Name, bool HasWeights);

    public record PredictionResponse(
        string Label,
        float Confidence,
        int? ExerciseId,
        string? ExerciseName,
        string? Dataset,
        Dictionary<string, object>? Details,
        Dictionary<string, object>? ModelInfo,
        List<string>? ProblematicJoints,
        Dictionary<string, float>? JointDeviations);

    public record ModelInfoResponse(
        List<string> AvailableModels,
        string ActiveModel,
        List<ExerciseInfo>? Exercises);

    // ---------- API calls ----------

    public async Task<HealthResponse?> HealthCheckAsync()
    {
        var resp = await _http.GetAsync("/api/v1/health");
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<HealthResponse>(JsonOpts);
    }

    public async Task<ModelInfoResponse?> GetModelsAsync()
    {
        var resp = await _http.GetAsync("/api/v1/models");
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<ModelInfoResponse>(JsonOpts);
    }

    public async Task<List<ExerciseInfo>?> GetExercisesAsync()
    {
        var resp = await _http.GetAsync("/api/v1/exercises");
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<List<ExerciseInfo>>(JsonOpts);
    }

    public async Task<PredictionResponse?> PredictVideoAsync(
        Stream fileStream, string fileName, int exerciseId, string dataset)
    {
        using var content = new MultipartFormDataContent();

        var streamContent = new StreamContent(fileStream);
        streamContent.Headers.ContentType = new MediaTypeHeaderValue("video/mp4");
        content.Add(streamContent, "file", fileName);

        content.Add(new StringContent(exerciseId.ToString()), "exercise_id");
        content.Add(new StringContent(dataset), "dataset");

        var resp = await _http.PostAsync("/api/v1/predict/video", content);
        resp.EnsureSuccessStatusCode();
        return await resp.Content.ReadFromJsonAsync<PredictionResponse>(JsonOpts);
    }

    public record AnnotatedVideoResult(
        byte[] VideoData,
        string Label,
        float Confidence,
        List<string>? ProblematicJoints);

    public async Task<AnnotatedVideoResult?> PredictVideoAnnotatedAsync(
        Stream fileStream, string fileName, int exerciseId, string dataset)
    {
        using var content = new MultipartFormDataContent();

        var streamContent = new StreamContent(fileStream);
        streamContent.Headers.ContentType = new MediaTypeHeaderValue("video/mp4");
        content.Add(streamContent, "file", fileName);

        content.Add(new StringContent(exerciseId.ToString()), "exercise_id");
        content.Add(new StringContent(dataset), "dataset");

        var resp = await _http.PostAsync("/api/v1/predict/video_annotated", content);
        resp.EnsureSuccessStatusCode();

        var videoData = await resp.Content.ReadAsByteArrayAsync();

        // Extract prediction metadata from headers
        var label = resp.Headers.TryGetValues("X-Prediction-Label", out var labelValues)
            ? labelValues.FirstOrDefault() ?? "unknown"
            : "unknown";

        var confidence = resp.Headers.TryGetValues("X-Prediction-Confidence", out var confValues)
            && float.TryParse(confValues.FirstOrDefault(), out var conf) ? conf : 0f;

        var problematicJoints = resp.Headers.TryGetValues("X-Problematic-Joints", out var jointValues)
            ? jointValues.FirstOrDefault()?.Split(',', StringSplitOptions.RemoveEmptyEntries).ToList()
            : null;

        return new AnnotatedVideoResult(videoData, label, confidence, problematicJoints);
    }
}
