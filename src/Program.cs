using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using SkiaSharp;
using System.Text.Json;

var builder = WebApplication.CreateBuilder(args);

// Listen on HTTPS 5332 if desired; local dev cert may be required.
// builder.WebHost.UseUrls("https://0.0.0.0:5332");

builder.Services.AddSingleton<SimWorld>();
builder.Services.AddHostedService<SimLoop>();

var app = builder.Build();

app.MapGet("/", () => Results.Text("Robo-sim minimal API"));

app.MapGet("/webcam/1/image", (SimWorld world) =>
{
    var jpeg = world.RenderJpeg();
    return Results.File(jpeg, "image/jpeg");
});

app.MapGet("/servo/{id:int}/currentStep", (int id, SimWorld w) =>
    Results.Json(new { value = w.Servo(id).CurrentStep, timestamp = DateTimeOffset.UtcNow }));

app.MapGet("/servo/{id:int}/targetStep", (int id, SimWorld w) =>
    Results.Json(new { value = w.Servo(id).TargetStep, timestamp = DateTimeOffset.UtcNow }));

app.MapGet("/servo/{id:int}/load", (int id, SimWorld w) =>
    Results.Json(new { value = w.Servo(id).Load, timestamp = DateTimeOffset.UtcNow }));

app.MapPost("/servo/{id:int}/targetStep", async (int id, HttpRequest req, SimWorld w) =>
{
    try
    {
        var body = await JsonSerializer.DeserializeAsync<TargetDto>(req.Body);
        if (body == null) return Results.BadRequest();
        w.Servo(id).TargetStep = body.value;
        return Results.Ok();
    }
    catch
    {
        return Results.BadRequest();
    }
});

// Optional helpers
app.MapPost("/reset", (SimWorld w) =>
{
    w.Reset();
    return Results.Ok();
});

app.MapPost("/spawn-blocks", (SimWorld w) =>
{
    w.SpawnDefaultBlocks();
    return Results.Ok();
});

app.MapGet("/state", (SimWorld w) =>
{
    // Return a simple state snapshot
    var state = new
    {
        servos = Enumerable.Range(1, w.Servos.Length).Select(i => new
        {
            id = i,
            currentStep = w.Servo(i).CurrentStep,
            targetStep = w.Servo(i).TargetStep,
            load = w.Servo(i).Load
        }),
        blocks = w.Blocks.Select(b => new { x = b.X, y = b.Y, yaw = b.YawDeg, size = b.Size, attached = b.Attached })
    };
    return Results.Json(state);
});

app.Run();

record TargetDto(int value);