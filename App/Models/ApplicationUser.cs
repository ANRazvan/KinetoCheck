using Microsoft.AspNetCore.Identity;

namespace App.Models;

public class ApplicationUser : IdentityUser
{
    public ICollection<Upload> Uploads { get; set; } = new List<Upload>();

    public ICollection<AnalysisHistoryEntry> HistoryEntries { get; set; } = new List<AnalysisHistoryEntry>();
}