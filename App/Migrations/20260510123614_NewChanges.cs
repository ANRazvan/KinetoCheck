using Microsoft.EntityFrameworkCore.Migrations;

#nullable disable

namespace App.Migrations
{
    /// <inheritdoc />
    public partial class NewChanges : Migration
    {
        /// <inheritdoc />
        protected override void Up(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AlterColumn<decimal>(
                name: "Threshold",
                table: "AnalysisHistoryEntries",
                type: "decimal(65,30)",
                nullable: false,
                oldClrType: typeof(decimal),
                oldType: "decimal(6,3)",
                oldPrecision: 6,
                oldScale: 3);

            migrationBuilder.AlterColumn<decimal>(
                name: "RawThreshold",
                table: "AnalysisHistoryEntries",
                type: "decimal(65,30)",
                nullable: false,
                oldClrType: typeof(decimal),
                oldType: "decimal(6,3)",
                oldPrecision: 6,
                oldScale: 3);

            migrationBuilder.AlterColumn<decimal>(
                name: "Margin",
                table: "AnalysisHistoryEntries",
                type: "decimal(65,30)",
                nullable: false,
                oldClrType: typeof(decimal),
                oldType: "decimal(6,3)",
                oldPrecision: 6,
                oldScale: 3);
        }

        /// <inheritdoc />
        protected override void Down(MigrationBuilder migrationBuilder)
        {
            migrationBuilder.AlterColumn<decimal>(
                name: "Threshold",
                table: "AnalysisHistoryEntries",
                type: "decimal(6,3)",
                precision: 6,
                scale: 3,
                nullable: false,
                oldClrType: typeof(decimal),
                oldType: "decimal(65,30)");

            migrationBuilder.AlterColumn<decimal>(
                name: "RawThreshold",
                table: "AnalysisHistoryEntries",
                type: "decimal(6,3)",
                precision: 6,
                scale: 3,
                nullable: false,
                oldClrType: typeof(decimal),
                oldType: "decimal(65,30)");

            migrationBuilder.AlterColumn<decimal>(
                name: "Margin",
                table: "AnalysisHistoryEntries",
                type: "decimal(6,3)",
                precision: 6,
                scale: 3,
                nullable: false,
                oldClrType: typeof(decimal),
                oldType: "decimal(65,30)");
        }
    }
}
