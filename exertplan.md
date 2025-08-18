You can build a lightweight, headless 2D simulator that exposes the same HTTP(S) API you plan for the Pi, without any game engine. The idea is:

- A .NET 9 Web API (Kestrel) that:
  - Advances a simple 2D arm+gripper+blocks simulation in a background loop (50–100 Hz).
  - Renders a top‑down “camera” image to JPEG using a CPU renderer (SkiaSharp).
  - Exposes endpoints like /webcam/1/image and /servo/{id}/... exactly as your real robot will.

High-level design

- World: a 2D table in mm coordinates (e.g., 640 x 480 mm).
- Robot: 3 planar joints (shoulder, elbow, wrist pitch), 1 wrist rotate, 1 gripper close. We’ll map “steps” to angles with a fixed steps-per-degree for simulation.
- Forward kinematics: compute end-effector (x, y) and orientation from joint angles.
- Gripper/contact: when the gripper closes near a block, attach that block to the gripper; while attached, move it with the end-effector. Simulate gripper load when holding.
- Camera: top-down render of the table, blocks, and arm onto a bitmap, encode as JPEG.
- Server: .NET minimal API with HTTPS on port 5332, endpoints to get image/servo states and to set target steps.

What you’ll need

- .NET 8 SDK
- NuGet: SkiaSharp (for rendering), MathNet.Numerics (optional, for matrix ops), Microsoft.Extensions.Hosting (for background service)

World/units

- Table: X in [0, 640] mm, Y in [0, 480] mm
- Camera image: 640 x 480 px, 1 px == 1 mm (simple identity mapping for now)

Suggested servo mapping (simulation only)

- steps_per_degree = 10 (so 0.1 deg per step)
- theta_deg = (currentStep - zeroStep) / steps_per_degree
- Link lengths: L1=150 mm, L2=120 mm, L3=80 mm
- Joint limits: clamp steps to keep angles in safe ranges
- Max servo speed: e.g., 200 steps/sec (you’ll see smooth motion)

Forward kinematics (planar 3R)

- Let t1, t2, t3 be shoulder, elbow, wrist angles in radians
- p0 = (baseX, baseY)
- p1 = p0 + (L1 cos t1, L1 sin t1)
- p2 = p1 + (L2 cos (t1+t2), L2 sin (t1+t2))
- p3 (end-effector) = p2 + (L3 cos (t1+t2+t3), L3 sin (t1+t2+t3))
- Wrist rotate (servo 4) adds a yaw for gripper drawing
- Gripper width derived from servo 5 steps (e.g., 0–50 mm)

Contact/attachment logic (simple, no physics engine)

- If gripper closes and nearest block center within r_attach (e.g., 20 mm) and gripper width is wide enough to “contain” block width:
  - Attach: block.attached = true; block.attachedOffset = transform from gripper frame to block pose
  - Set gripper load high (e.g., 0.6–0.9)
- While attached:
  - Set block pose from gripper pose each tick
  - If gripper opens beyond block width + margin, detach; load drops to baseline
- Otherwise, blocks are static (no pushing when open). This is enough for pick/place testing.

Server and endpoints

- GET /webcam/1/image -> image/jpeg, latest rendered frame
- GET /servo/{id}/currentStep -> JSON { value, timestamp }
- GET /servo/{id}/targetStep -> JSON { value, timestamp }
- GET /servo/{id}/load -> JSON { value, timestamp }
- POST /servo/{id}/targetStep -> JSON body { value: int }

Optional:
- POST /reset, POST /spawn-blocks, GET /state, GET /calibration/homography (identity)

Code skeleton (condensed)

Program.cs
- sets up HTTPS, background sim loop, endpoints.

using Microsoft.AspNetCore.Builder;
using Microsoft.Extensions.Hosting;
using SkiaSharp;

var builder = WebApplication.CreateBuilder(args);

// Listen on HTTPS 5332 (use dev cert locally)
builder.WebHost.UseUrls("https://0.0.0.0:5332");

builder.Services.AddSingleton<SimWorld>();
builder.Services.AddHostedService<SimLoop>();

var app = builder.Build();

app.MapGet("/webcam/1/image", async (SimWorld world) =>
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
    var body = await System.Text.Json.JsonSerializer.DeserializeAsync<TargetDto>(req.Body);
    if (body == null) return Results.BadRequest();
    w.Servo(id).TargetStep = body.value;
    return Results.Ok();
});

app.Run();

record TargetDto(int value);

Simulation core

public class SimWorld
{
    public readonly Servo[] Servos = new Servo[5];
    public readonly List<Block> Blocks = new();
    // table and camera
    public readonly int WidthPx = 640, HeightPx = 480;
    public readonly float PxPerMm = 1f;
    public readonly float BaseX = 120, BaseY = 240; // mm

    // Link lengths (mm) and steps/deg
    public float L1=150, L2=120, L3=80;
    public float StepsPerDeg = 10f;
    public float MaxServoSpeed = 200f; // steps/sec
    public int GripperMinStep = 0, GripperMaxStep = 500; // map to width

    public SimWorld()
    {
        for (int i=0;i<5;i++) Servos[i] = new Servo();
        // zero poses
        Servos[0].CurrentStep = Servos[0].TargetStep = 0;    // shoulder
        Servos[1].CurrentStep = Servos[1].TargetStep = 0;    // elbow
        Servos[2].CurrentStep = Servos[2].TargetStep = 0;    // wrist pitch
        Servos[3].CurrentStep = Servos[3].TargetStep = 0;    // wrist rotate
        Servos[4].CurrentStep = Servos[4].TargetStep = 400;  // gripper open

        // spawn some blocks
        Blocks.Add(new Block { X=350, Y=200, YawDeg=0,  Size=30, Color = new SKColor(220,30,30) });
        Blocks.Add(new Block { X=450, Y=320, YawDeg=45, Size=30, Color = new SKColor(30,180,60) });
    }

    public Servo Servo(int id) => Servos[id-1];

    public void Tick(double dt)
    {
        // move servos toward targets
        foreach (var s in Servos) s.Tick(dt, MaxServoSpeed);

        // compute FK
        var (p0, p1, p2, p3, eeYawRad) = ForwardKinematics();

        // gripper width
        float gripW = Map(Servos[4].CurrentStep, GripperMinStep, GripperMaxStep, 0f, 50f);

        // contact/attach
        UpdateGraspLogic(p3, eeYawRad, gripW);

        // simulate gripper load
        Servos[4].Load = Blocks.Any(b => b.Attached) ? 0.7f : 0.05f;
        for (int i=0;i<4;i++) Servos[i].Load = 0.05f; // simple baseline
    }

    public (SKPoint p0, SKPoint p1, SKPoint p2, SKPoint p3, float eeYaw) ForwardKinematics()
    {
        float t1 = DegToRad(Servos[0].CurrentStep / StepsPerDeg);
        float t2 = DegToRad(Servos[1].CurrentStep / StepsPerDeg);
        float t3 = DegToRad(Servos[2].CurrentStep / StepsPerDeg);

        var p0 = new SKPoint(BaseX, BaseY);
        var p1 = new SKPoint(
            p0.X + L1 * MathF.Cos(t1),
            p0.Y + L1 * MathF.Sin(t1));
        var p2 = new SKPoint(
            p1.X + L2 * MathF.Cos(t1 + t2),
            p1.Y + L2 * MathF.Sin(t1 + t2));
        var p3 = new SKPoint(
            p2.X + L3 * MathF.Cos(t1 + t2 + t3),
            p2.Y + L3 * MathF.Sin(t1 + t2 + t3));

        float wristRot = DegToRad(Servos[3].CurrentStep / StepsPerDeg);
        float eeYaw = t1 + t2 + t3 + wristRot;
        return (p0, p1, p2, p3, eeYaw);
    }

    void UpdateGraspLogic(SKPoint ee, float eeYaw, float gripW)
    {
        // Attach if closing near a block
        foreach (var b in Blocks)
        {
            if (b.Attached)
            {
                // follow gripper
                b.X = ee.X;
                b.Y = ee.Y;
                b.YawDeg = RadToDeg(eeYaw);
            }
        }

        // Simple attach rule: if grip width < block size+10 and distance < 20 mm, attach
        var nearest = Blocks.OrderBy(b => Dist(b.X, b.Y, ee.X, ee.Y)).FirstOrDefault();
        if (nearest != null && !nearest.Attached)
        {
            float d = Dist(nearest.X, nearest.Y, ee.X, ee.Y);
            if (d < 20 && gripW < nearest.Size + 10)
            {
                nearest.Attached = true;
            }
        }

        // Detach rule: if opened wider than block size + margin
        foreach (var b in Blocks.Where(b => b.Attached))
        {
            if (gripW > b.Size + 15) b.Attached = false;
        }
    }

    public byte[] RenderJpeg()
    {
        using var bmp = new SKBitmap(WidthPx, HeightPx, SKColorType.Rgba8888, SKAlphaType.Premul);
        using var canvas = new SKCanvas(bmp);
        canvas.Clear(new SKColor(240,240,240));

        // Grid
        using (var p = new SKPaint { Color = new SKColor(220,220,220), StrokeWidth = 1 })
        for (int x=0; x<=WidthPx; x+=50) canvas.DrawLine(x,0,x,HeightPx,p);
        for (int y=0; y<=HeightPx; y+=50) canvas.DrawLine(0,y,WidthPx,y,p);

        // Draw blocks
        foreach (var b in Blocks)
        {
            using var paint = new SKPaint { Color = b.Color, IsStroke = false };
            canvas.Save();
            canvas.Translate(b.X, b.Y);
            canvas.RotateDegrees(b.YawDeg);
            float s = b.Size;
            var rect = new SKRect(-s/2, -s/2, s/2, s/2);
            canvas.DrawRect(rect, paint);
            canvas.Restore();
        }

        // Draw arm
        var (p0,p1,p2,p3,eeYaw) = ForwardKinematics();
        using var armPaint = new SKPaint { Color = new SKColor(50,50,50), StrokeWidth=8, IsStroke=true, StrokeCap=SKStrokeCap.Round };
        canvas.DrawLine(p0, p1, armPaint);
        canvas.DrawLine(p1, p2, armPaint);
        canvas.DrawLine(p2, p3, armPaint);

        // Draw gripper
        float gripW = Map(Servos[4].CurrentStep, GripperMinStep, GripperMaxStep, 0f, 50f);
        using var gripPaint = new SKPaint { Color = new SKColor(30,30,30), StrokeWidth = 6, IsStroke = true };
        // Two fingers separated by gripW, oriented perpendicular to eeYaw
        var dx = MathF.Cos(eeYaw + MathF.PI/2) * gripW/2;
        var dy = MathF.Sin(eeYaw + MathF.PI/2) * gripW/2;
        var g1a = new SKPoint(p3.X + dx, p3.Y + dy);
        var g1b = new SKPoint(g1a.X + 15*MathF.Cos(eeYaw), g1a.Y + 15*MathF.Sin(eeYaw));
        var g2a = new SKPoint(p3.X - dx, p3.Y - dy);
        var g2b = new SKPoint(g2a.X + 15*MathF.Cos(eeYaw), g2a.Y + 15*MathF.Sin(eeYaw));
        canvas.DrawLine(g1a, g1b, gripPaint);
        canvas.DrawLine(g2a, g2b, gripPaint);

        using var img = SKImage.FromBitmap(bmp);
        using var data = img.Encode(SKEncodedImageFormat.Jpeg, 80);
        return data.ToArray();
    }

    static float Map(float v, float a1, float a2, float b1, float b2) =>
        b1 + (v - a1) * (b2 - b1) / (a2 - a1);

    static float DegToRad(float d) => (float)(Math.PI/180.0) * d;
    static float RadToDeg(float r) => (float)(180.0/Math.PI) * r;
    static float Dist(float x1,float y1,float x2,float y2) =>
        MathF.Sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
}

public class Servo
{
    public int CurrentStep;
    public int TargetStep;
    public float Load;
    public void Tick(double dt, float maxSpeedStepsPerSec)
    {
        float maxStepDelta = (float)(maxSpeedStepsPerSec * dt);
        float delta = TargetStep - CurrentStep;
        if (MathF.Abs(delta) <= maxStepDelta) CurrentStep = TargetStep;
        else CurrentStep += (int)MathF.CopySign(maxStepDelta, delta);
    }
}

public class Block
{
    public float X, Y;
    public float YawDeg;
    public float Size;
    public SKColor Color;
    public bool Attached;
}

Background loop

using Microsoft.Extensions.Hosting;

public class SimLoop : BackgroundService
{
    private readonly SimWorld _world;
    public SimLoop(SimWorld world) => _world = world;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var sw = System.Diagnostics.Stopwatch.StartNew();
        long last = sw.ElapsedMilliseconds;
        while (!stoppingToken.IsCancellationRequested)
        {
            long now = sw.ElapsedMilliseconds;
            double dt = Math.Max(0.0, (now - last) / 1000.0);
            last = now;

            _world.Tick(dt);

            await Task.Delay(10, stoppingToken); // ~100 Hz
        }
    }
}

How to run with HTTPS on port 5332

- Install dev cert once:
  - dotnet dev-certs https --trust
- Run:
  - dotnet run --urls "https://0.0.0.0:5332"

Test endpoints

- curl -k https://localhost:5332/servo/1/currentStep
- curl -k https://localhost:5332/webcam/1/image --output frame.jpg
- curl -k -X POST https://localhost:5332/servo/1/targetStep -H "Content-Type: application/json" -d '{"value":300}'

Notes and extensions

- Latency/noise: add a small delay to /webcam/1/image and noise to servo readings to better mimic reality.
- Homography: currently 1 px = 1 mm. If you want to exercise calibration code, draw a checkerboard corner in the render and generate a proper H; or expose GET /calibration/homography returning a 3x3 matrix.
- Blocks and place area: add endpoints to spawn/move blocks and define a place rectangle. You can also draw the place zone onto the image.
- 2 cameras: you can spin up /webcam/2/image by rendering from a slightly skewed virtual pose (perspective) if needed later.
- Behavior cloning: this sim lets you collect (image, servo states, actions) logs through the same API you’ll use on the Pi.

If you want, I can package this into a runnable repository skeleton with the NuGet references and a couple of extra endpoints (reset/spawn) so you can start coding your vision and control against it immediately.