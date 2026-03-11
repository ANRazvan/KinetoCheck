using KinetoCheckWeb.Services;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.
builder.Services.AddRazorPages();

// Register typed HttpClient for the KinetoCheck backend API
var backendUrl = builder.Configuration["BackendUrl"] ?? "http://localhost:8000";
builder.Services.AddHttpClient<KinetoCheckApiClient>(client =>
{
    client.BaseAddress = new Uri(backendUrl);
    client.Timeout = TimeSpan.FromSeconds(120); // video processing can be slow
});

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error");
}

app.UseRouting();

app.UseAuthorization();

app.MapStaticAssets();
app.MapRazorPages()
   .WithStaticAssets();

app.Run();
