using App.Models;
using Microsoft.AspNetCore.Identity.EntityFrameworkCore;
using Microsoft.EntityFrameworkCore;

namespace App.Data;

public class AppDbContext : IdentityDbContext<ApplicationUser>
{
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options)
    {
    }

    public DbSet<Upload> Uploads => Set<Upload>();

    public DbSet<UploadStatistics> UploadStatistics => Set<UploadStatistics>();

    public DbSet<JointInsight> JointInsights => Set<JointInsight>();

    public DbSet<AnalysisHistoryEntry> AnalysisHistoryEntries => Set<AnalysisHistoryEntry>();

    protected override void OnModelCreating(ModelBuilder builder)
    {
        base.OnModelCreating(builder);

        builder.Entity<Upload>(entity =>
        {
            entity.Property(upload => upload.OriginalFileName)
                .HasMaxLength(260)
                .IsRequired();

            entity.Property(upload => upload.SelectedExerciseId)
                .HasMaxLength(32)
                .IsRequired();

            entity.HasOne(upload => upload.User)
                .WithMany(user => user.Uploads)
                .HasForeignKey(upload => upload.UserId)
                .OnDelete(DeleteBehavior.SetNull);
        });

        builder.Entity<UploadStatistics>(entity =>
        {
            entity.Property(statistics => statistics.PredictedLabel)
                .HasMaxLength(120)
                .IsRequired();

            entity.Property(statistics => statistics.AssessedExerciseName)
                .HasMaxLength(120)
                .IsRequired();

            entity.Property(statistics => statistics.Score)
                .HasPrecision(6, 3);

            entity.Property(statistics => statistics.Threshold)
                .HasPrecision(6, 3);

            entity.Property(statistics => statistics.RawThreshold)
                .HasPrecision(6, 3);

            entity.Property(statistics => statistics.Margin)
                .HasPrecision(6, 3);

            entity.HasOne(statistics => statistics.Upload)
                .WithOne(upload => upload.Statistics)
                .HasForeignKey<UploadStatistics>(statistics => statistics.UploadId)
                .OnDelete(DeleteBehavior.Cascade);
        });

        builder.Entity<ApplicationUser>(entity =>
        {
        });

        builder.Entity<JointInsight>(entity =>
        {
            entity.Property(insight => insight.JointName)
                .HasMaxLength(120)
                .IsRequired();

            entity.Property(insight => insight.Deviation)
                .HasPrecision(6, 3);

            entity.Property(insight => insight.Importance)
                .HasPrecision(6, 3);

            entity.Property(insight => insight.ProblemScore)
                .HasPrecision(6, 3);

            entity.HasOne(insight => insight.UploadStatistics)
                .WithMany(statistics => statistics.JointInsights)
                .HasForeignKey(insight => insight.UploadStatisticsId)
                .OnDelete(DeleteBehavior.Cascade);
        });

        builder.Entity<AnalysisHistoryEntry>(entity =>
        {
            entity.Property(history => history.OriginalFileName)
                .HasMaxLength(260)
                .IsRequired();

            entity.Property(history => history.ExerciseName)
                .HasMaxLength(120)
                .IsRequired();

            entity.Property(history => history.PredictedLabel)
                .HasMaxLength(120)
                .IsRequired();

            entity.Property(history => history.Score)
                .HasPrecision(6, 3);

            entity.Property(history => history.ScoreDeltaFromPrevious)
                .HasPrecision(6, 3);

            entity.Property(history => history.Summary)
                .HasMaxLength(600)
                .IsRequired();

            entity.HasOne(history => history.User)
                .WithMany(user => user.HistoryEntries)
                .HasForeignKey(history => history.UserId)
                .OnDelete(DeleteBehavior.SetNull);

            entity.HasOne(history => history.Upload)
                .WithMany(upload => upload.HistoryEntries)
                .HasForeignKey(history => history.UploadId)
                .OnDelete(DeleteBehavior.Cascade);

            entity.HasOne(history => history.Statistics)
                .WithMany()
                .HasForeignKey(history => history.StatisticsId)
                .OnDelete(DeleteBehavior.SetNull);
        });
    }
}