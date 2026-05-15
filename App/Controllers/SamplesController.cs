using Microsoft.AspNetCore.Mvc;

namespace App.Controllers;

public class SamplesController : Controller
{
    private readonly IWebHostEnvironment _env;
    private readonly ILogger<SamplesController> _logger;

    public SamplesController(IWebHostEnvironment env, ILogger<SamplesController> logger)
    {
        _env = env;
        _logger = logger;
    }

    // GET /samples/preview/{fileName}
    [HttpGet("/samples/preview/{fileName}")]
    public async Task<IActionResult> Preview(string fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName)) return BadRequest();

        // Sanitize file name to prevent path traversal
        fileName = Path.GetFileName(fileName);

        var samplesRoot = Path.Combine(_env.WebRootPath, "samples");
        var originalPath = Path.Combine(samplesRoot, fileName);
        if (!System.IO.File.Exists(originalPath))
        {
            return NotFound();
        }

        var webDir = Path.Combine(samplesRoot, "web");
        Directory.CreateDirectory(webDir);
        var webPath = Path.Combine(webDir, fileName);

        try
        {
            var needsTranscode = true;
            if (System.IO.File.Exists(webPath))
            {
                var origTime = System.IO.File.GetLastWriteTimeUtc(originalPath);
                var webTime = System.IO.File.GetLastWriteTimeUtc(webPath);
                if (webTime >= origTime)
                {
                    needsTranscode = false;
                }
            }

            if (needsTranscode)
            {
                // Run ffmpeg to transcode to H.264 + AAC for browser compatibility
                // This requires ffmpeg to be installed and available in PATH on the host.
                var ffmpeg = "ffmpeg";
                var args = $"-y -i \"{originalPath}\" -c:v libx264 -preset veryfast -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart \"{webPath}\"";

                _logger.LogInformation("Transcoding sample {Original} -> {WebPath}", originalPath, webPath);

                var psi = new System.Diagnostics.ProcessStartInfo
                {
                    FileName = ffmpeg,
                    Arguments = args,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true
                };

                using var proc = System.Diagnostics.Process.Start(psi);
                if (proc == null)
                {
                    _logger.LogWarning("Failed to start ffmpeg process");
                }
                else
                {
                    var stderr = await proc.StandardError.ReadToEndAsync();
                    await proc.WaitForExitAsync();
                    if (proc.ExitCode != 0)
                    {
                        _logger.LogWarning(stderr);
                        // fallback: serve original if transcoding failed
                        return PhysicalFile(originalPath, "video/mp4");
                    }
                }
            }

            // Serve the transcoded file with range processing enabled
            var result = PhysicalFile(webPath, "video/mp4");
            if (result is PhysicalFileResult pfr)
            {
                pfr.EnableRangeProcessing = true;
            }
            return result;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error preparing sample preview");
            // fallback to serving original file
            var fallback = PhysicalFile(originalPath, "video/mp4");
            if (fallback is PhysicalFileResult pfr2) pfr2.EnableRangeProcessing = true;
            return fallback;
        }
    }
}
