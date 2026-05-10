using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace App.Migrations
{
    public partial class AddAnalysisInsights : Migration
    {
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AddColumn<decimal>(
                name: "Threshold",
                table: "AnalysisHistoryEntries",
                type: "decimal(6,3)",
                precision: 6,
                scale: 3,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.AddColumn<decimal>(
                name: "RawThreshold",
                table: "AnalysisHistoryEntries",
                type: "decimal(6,3)",
                precision: 6,
                scale: 3,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.AddColumn<decimal>(
                name: "Margin",
                table: "AnalysisHistoryEntries",
                type: "decimal(6,3)",
                precision: 6,
                scale: 3,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.AddColumn<decimal>(
                name: "Threshold",
                table: "UploadStatistics",
                type: "decimal(6,3)",
                precision: 6,
                scale: 3,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.AddColumn<decimal>(
                name: "RawThreshold",
                table: "UploadStatistics",
                type: "decimal(6,3)",
                precision: 6,
                scale: 3,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.AddColumn<decimal>(
                name: "Margin",
                table: "UploadStatistics",
                type: "decimal(6,3)",
                precision: 6,
                scale: 3,
                nullable: false,
                defaultValue: 0m);

            migrationBuilder.CreateTable(
                name: "JointInsights",
                columns: table => new
                {
                    Id = table.Column<long>(type: "bigint", nullable: false)
                        .Annotation("MySql:ValueGenerationStrategy", Microsoft.EntityFrameworkCore.Metadata.MySqlValueGenerationStrategy.IdentityColumn),
                    UploadStatisticsId = table.Column<long>(type: "bigint", nullable: false),
                    RankIndex = table.Column<int>(type: "int", nullable: false),
                    JointName = table.Column<string>(type: "varchar(120)", maxLength: 120, nullable: false)
                        .Annotation("MySql:CharSet", "utf8mb4"),
                    JointIndex = table.Column<int>(type: "int", nullable: false),
                    Deviation = table.Column<decimal>(type: "decimal(6,3)", precision: 6, scale: 3, nullable: false),
                    Importance = table.Column<decimal>(type: "decimal(6,3)", precision: 6, scale: 3, nullable: false),
                    ProblemScore = table.Column<decimal>(type: "decimal(6,3)", precision: 6, scale: 3, nullable: false)
                },
                constraints: table =>
                {
                    table.PrimaryKey("PK_JointInsights", x => x.Id);
                    table.ForeignKey(
                        name: "FK_JointInsights_UploadStatistics_UploadStatisticsId",
                        column: x => x.UploadStatisticsId,
                        principalTable: "UploadStatistics",
                        principalColumn: "Id",
                        onDelete: ReferentialAction.Cascade);
                })
                .Annotation("MySql:CharSet", "utf8mb4");

            migrationBuilder.CreateIndex(
                name: "IX_JointInsights_UploadStatisticsId",
                table: "JointInsights",
                column: "UploadStatisticsId");
        }

        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.DropTable(
                name: "JointInsights");

            migrationBuilder.DropColumn(
                name: "DisplayName",
                table: "AspNetUsers");

            migrationBuilder.DropColumn(
                name: "Threshold",
                table: "AnalysisHistoryEntries");

            migrationBuilder.DropColumn(
                name: "RawThreshold",
                table: "AnalysisHistoryEntries");

            migrationBuilder.DropColumn(
                name: "Margin",
                table: "AnalysisHistoryEntries");

            migrationBuilder.DropColumn(
                name: "Threshold",
                table: "UploadStatistics");

            migrationBuilder.DropColumn(
                name: "RawThreshold",
                table: "UploadStatistics");

            migrationBuilder.DropColumn(
                name: "Margin",
                table: "UploadStatistics");
        }
    }
}