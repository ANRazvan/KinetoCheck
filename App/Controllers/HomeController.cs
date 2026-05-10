using System.Diagnostics;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc;
using App.Models;

namespace App.Controllers;

public class HomeController : Controller
{
    private readonly ILogger<HomeController> _logger;
    private readonly IWebHostEnvironment _environment;

    public HomeController(ILogger<HomeController> logger, IWebHostEnvironment environment)
    {
        _logger = logger;
        _environment = environment;
    }

    public IActionResult Index()
    {
        ViewBag.SelectedexerciseId = "auto";
        ViewBag.AvailableExercises = Enumerable.Range(1, 10)
            .Select(id => new { Id = id, Name = $"Exercise {id:00}" })
            .ToList();
        return View();
    }

    [HttpPost]
    public async Task<IActionResult> UploadVideo(IFormFile videoFile)
    {
        ViewBag.AvailableExercises = Enumerable.Range(1, 10)
            .Select(id => new { Id = id, Name = $"Exercise {id:00}" })
            .ToList();

        var selectedExerciseId = Request.Form["exercise_id"].ToString();
        _logger.LogInformation("Selected exercise ID: {ExerciseId}", selectedExerciseId);
        ViewBag.SelectedExerciseId = string.IsNullOrWhiteSpace(selectedExerciseId) ? "auto" : selectedExerciseId;
        _logger.LogInformation("Selected exercise ID: {ExerciseId}", selectedExerciseId);
        if (videoFile == null || videoFile.Length == 0)
        {
            ModelState.AddModelError("", "Please select a video file to upload.");
            return View("Index");
        }

        // Save to a temp file
        var tmpFolder = Path.Combine(Path.GetTempPath(), "kinetocheck_uploads");
        Directory.CreateDirectory(tmpFolder);
        var tmpFile = Path.Combine(tmpFolder, Path.GetFileName(videoFile.FileName));
        using (var stream = System.IO.File.Create(tmpFile))
        {
            await videoFile.CopyToAsync(stream);
        }

        // Forward to Python FastAPI service
        using var client = new HttpClient();
        using var content = new MultipartFormDataContent();
        using var fs = System.IO.File.OpenRead(tmpFile);
        content.Add(new StreamContent(fs), "video", Path.GetFileName(tmpFile));
        content.Add(new StringContent(selectedExerciseId), "exercise_id");

        HttpResponseMessage resp;
        try
        {
            resp = await client.PostAsync("http://localhost:8000/analyze-video/", content);
        }
        catch (Exception ex)
        {
            ModelState.AddModelError("", "Failed to contact analysis service: " + ex.Message);
            return View("Index");
        }

        if (!resp.IsSuccessStatusCode)
        {
            var err = await resp.Content.ReadAsStringAsync();
            ModelState.AddModelError("", "Analysis failed: " + err);
            return View("Index");
        }

        var json = await resp.Content.ReadAsStringAsync();

        // Parse JSON and extract key fields
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            ViewBag.HasAnalysisResult = true;

            // Extract score and label
            if (root.TryGetProperty("best", out var best))
            {
                if (best.TryGetProperty("score", out var score))
                    ViewBag.Score = score.GetDouble().ToString("F3");
                if (best.TryGetProperty("predicted_label", out var label))
                    ViewBag.Label = label.GetString();
                if (best.TryGetProperty("exercise_name", out var exerciseName))
                    ViewBag.AssessedExercise = exerciseName.GetString();
            }

            // Extract worst joints
            if (root.TryGetProperty("worst_joints", out var worst))
            {
                var joints = new List<string>();
                foreach (var j in worst.EnumerateArray().Take(3))
                {
                    if (j.TryGetProperty("joint", out var jname))
                        joints.Add(jname.GetString() ?? "unknown");
                }
                ViewBag.WorstJoints = string.Join(", ", joints);
            }

            // Copy annotated video from Python output into the MVC app's wwwroot
            if (root.TryGetProperty("annotated_video", out var annotatedVideo))
            {
                var sourcePath = annotatedVideo.GetString();
                if (!string.IsNullOrWhiteSpace(sourcePath) && System.IO.File.Exists(sourcePath))
                {
                    var sessionId = root.TryGetProperty("session_id", out var sid) ? sid.GetString() : Guid.NewGuid().ToString("N");
                    var uploadDir = Path.Combine(_environment.WebRootPath, "uploads", sessionId ?? Guid.NewGuid().ToString("N"));
                    Directory.CreateDirectory(uploadDir);

                    var fileName = Path.GetFileName(sourcePath);
                    var destinationPath = Path.Combine(uploadDir, fileName);
                    System.IO.File.Copy(sourcePath, destinationPath, overwrite: true);

                    ViewBag.AnnotatedVideoUrl = $"/uploads/{sessionId}/{fileName}";
                }
            }
        }
        catch (Exception ex)
        {
            ModelState.AddModelError("", "Failed to parse results: " + ex.Message);
        }
        
        return View("Index");
    }

    public IActionResult Privacy()
    {
        return View();
    }

    [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
    public IActionResult Error()
    {
        return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
    }
}
