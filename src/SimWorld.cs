using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using SkiaSharp;

public class SimWorld
{
    public readonly Servo[] Servos = new Servo[5];
    public readonly List<Block> Blocks = new();

    // table and camera
    public readonly int WidthPx = 640, HeightPx = 480;
    public readonly float PxPerMm = 1f;
    public readonly float BaseX = 120f, BaseY = 240f; // mm (base position)

    // Link lengths (mm) and steps/deg
    public float L1 = 150f, L2 = 120f, L3 = 80f;
    public float StepsPerDeg = 10f;
    public float MaxServoSpeed = 200f; // steps/sec
    public int GripperMinStep = 0, GripperMaxStep = 500; // map to width 0-50mm

    // grasping parameters
    private const float RAttach = 20f;
    private const float GraspMargin = 5f;

    public SimWorld()
    {
        for (int i = 0; i < 5; i++) Servos[i] = new Servo();

        // set servo limits (in steps)
        Servos[0].MinStep = -90 * (int)StepsPerDeg; Servos[0].MaxStep = 90 * (int)StepsPerDeg;
        Servos[1].MinStep = -80 * (int)StepsPerDeg; Servos[1].MaxStep = 20 * (int)StepsPerDeg;
        Servos[2].MinStep = 0; Servos[2].MaxStep = 120 * (int)StepsPerDeg;
        Servos[3].MinStep = -90 * (int)StepsPerDeg; Servos[3].MaxStep = 90 * (int)StepsPerDeg;
        Servos[4].MinStep = 0; Servos[4].MaxStep = 45 * (int)StepsPerDeg;

        // set initial poses
        Servos[0].CurrentStep = Servos[0].TargetStep = 0;    // base rotate
        Servos[1].CurrentStep = Servos[1].TargetStep = 0;    // shoulder
        Servos[2].CurrentStep = Servos[2].TargetStep = 0;    // elbow
        Servos[3].CurrentStep = Servos[3].TargetStep = 0;    // wrist rotate
        Servos[4].CurrentStep = Servos[4].TargetStep = 400;  // gripper open

        SpawnDefaultBlocks();
    }

    public void Reset()
    {
        Blocks.Clear();
        for (int i = 0; i < 5; i++)
        {
            Servos[i].CurrentStep = Servos[i].TargetStep = 0;
            Servos[i].Load = 0.05f;
        }
        Servos[4].CurrentStep = Servos[4].TargetStep = 400;
        SpawnDefaultBlocks();
    }

    public void SpawnDefaultBlocks()
    {
        Blocks.Clear();
        Blocks.Add(new Block { X = 350, Y = 200, YawDeg = 0, Size = 30, Color = new SKColor(220, 30, 30) });
        Blocks.Add(new Block { X = 450, Y = 320, YawDeg = 45, Size = 30, Color = new SKColor(30, 180, 60) });
    }

    public Servo Servo(int id)
    {
        if (id < 1 || id > Servos.Length) throw new ArgumentOutOfRangeException(nameof(id));
        return Servos[id - 1];
    }

    public void Tick(double dt)
    {
        // move servos toward targets
        foreach (var s in Servos) s.Tick(dt, MaxServoSpeed);

        // compute FK
        var (p0, p1, p2, p3, eeYaw) = ForwardKinematics();

        // gripper width in mm
        float gripW = Map(Servos[4].CurrentStep, GripperMinStep, GripperMaxStep, 0f, 50f);

        // contact/attach
        UpdateGraspLogic(p3, eeYaw, gripW);

        // simulate gripper load
        Servos[4].Load = Blocks.Any(b => b.Attached) ? 0.7f : 0.05f;
        for (int i = 0; i < 4; i++) Servos[i].Load = 0.05f; // simple baseline
    }

    public (Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float eeYaw) ForwardKinematics()
    {
        // Note: angles are defined in a way that may seem unintuitive but match a specific physical robot
        float t1_base_rotate = DegToRad(Servos[0].CurrentStep / StepsPerDeg);
        float t2_shoulder = DegToRad(Servos[1].CurrentStep / StepsPerDeg);
        float t3_elbow = DegToRad(Servos[2].CurrentStep / StepsPerDeg);
        float t4_wrist_rotate = DegToRad(Servos[3].CurrentStep / StepsPerDeg);

        var p0 = new Vector3(BaseX, BaseY, 0);

        // Shoulder position
        var p1 = new Vector3(
            p0.X + L1 * MathF.Cos(t2_shoulder) * MathF.Cos(t1_base_rotate),
            p0.Y + L1 * MathF.Cos(t2_shoulder) * MathF.Sin(t1_base_rotate),
            p0.Z + L1 * MathF.Sin(t2_shoulder)
        );

        // Elbow position
        var p2 = new Vector3(
            p1.X + L2 * MathF.Cos(t2_shoulder + t3_elbow) * MathF.Cos(t1_base_rotate),
            p1.Y + L2 * MathF.Cos(t2_shoulder + t3_elbow) * MathF.Sin(t1_base_rotate),
            p1.Z + L2 * MathF.Sin(t2_shoulder + t3_elbow)
        );

        // End-effector position
        var p3 = new Vector3(
            p2.X + L3 * MathF.Cos(t2_shoulder + t3_elbow) * MathF.Cos(t1_base_rotate),
            p2.Y + L3 * MathF.Cos(t2_shoulder + t3_elbow) * MathF.Sin(t1_base_rotate),
            p2.Z + L3 * MathF.Sin(t2_shoulder + t3_elbow)
        );

        float eeYaw = t1_base_rotate + t4_wrist_rotate;
        return (p0, p1, p2, p3, eeYaw);
    }

    void UpdateGraspLogic(Vector3 ee, float eeYaw, float gripW)
    {
        // if block attached -> follow gripper
        foreach (var b in Blocks.Where(b => b.Attached))
        {
            b.X = ee.X;
            b.Y = ee.Y;
            b.YawDeg = RadToDeg(eeYaw);
        }

        // attach logic: if gripper closed near a block and gripper width small enough
        // We'll consider "closing" if target/current small enough (< some threshold)
        float gripClosed = gripW; // mm
        foreach (var b in Blocks.Where(b => !b.Attached))
        {
            float d = Dist(b.X, b.Y, ee.X, ee.Y);
            if (d <= RAttach && gripClosed <= b.Size + GraspMargin)
            {
                b.Attached = true;
                // optional: store offset (not needed for the simple follow)
                b.OffsetX = b.X - ee.X;
                b.OffsetY = b.Y - ee.Y;
                b.OffsetYawDeg = b.YawDeg - RadToDeg(eeYaw);
                break; // attach only one
            }
        }

        // detach logic: if any attached block and gripper opens beyond block size+margin
        foreach (var b in Blocks.Where(b => b.Attached).ToArray())
        {
            if (gripClosed > b.Size + GraspMargin)
            {
                b.Attached = false;
            }
        }
    }

    public byte[] RenderJpeg()
    {
        using var bmp = new SKBitmap(WidthPx, HeightPx);
        using var canvas = new SKCanvas(bmp);
        canvas.Clear(new SKColor(240, 240, 240));

        using var gridPaint = new SKPaint { Color = new SKColor(200, 200, 200), StrokeWidth = 1, IsStroke = true };
        for (int x = 0; x < WidthPx; x += 64) canvas.DrawLine(x, 0, x, HeightPx, gridPaint);
        for (int y = 0; y < HeightPx; y += 64) canvas.DrawLine(0, y, WidthPx, y, gridPaint);

        foreach (var b in Blocks)
        {
            using var blockPaint = new SKPaint { Color = b.Color, IsStroke = false };
            canvas.Save();
            canvas.Translate(b.X, b.Y);
            canvas.RotateDegrees(b.YawDeg);
            canvas.DrawRect(-b.Size / 2, -b.Size / 2, b.Size, b.Size, blockPaint);
            canvas.Restore();
        }

        var (p0, p1, p2, p3, eeYaw) = ForwardKinematics();
        var pts = new[] { p0, p1, p2, p3 }.Select(p => new SKPoint(p.X, p.Y)).ToArray();

        using var armPaint = new SKPaint { Color = SKColors.Gray, StrokeWidth = 10, IsStroke = true, StrokeCap = SKStrokeCap.Round };
        using var jointPaint = new SKPaint { Color = SKColors.DarkGray, StrokeWidth = 4, IsStroke = false };
        using var font = new SKFont { Size = 12 };
        using var textPaint = new SKPaint(font) { Color = SKColors.Black };
        canvas.DrawLine(pts[0], pts[1], armPaint);
        canvas.DrawLine(pts[1], pts[2], armPaint);
        canvas.DrawLine(pts[2], pts[3], armPaint);
        for (int i = 0; i < pts.Length; i++)
        {
            canvas.DrawCircle(pts[i], 8, jointPaint);
            canvas.DrawText($"ID:{i + 1}", pts[i].X + 10, pts[i].Y - 10, textPaint);
        }

        float gripW = Map(Servos[4].CurrentStep, GripperMinStep, GripperMaxStep, 0f, 50f);
        using var gripPaint = new SKPaint { Color = SKColors.Black, StrokeWidth = 4, IsStroke = true };
        var ee = pts[3];
        canvas.DrawText("ID:5", ee.X + 10, ee.Y + 20, textPaint);
        float dx = gripW / 2 * MathF.Sin(eeYaw);
        float dy = gripW / 2 * -MathF.Cos(eeYaw);
        var g1a = new SKPoint(ee.X + dx, ee.Y + dy);
        var g1b = new SKPoint(g1a.X + 15 * MathF.Cos(eeYaw), g1a.Y + 15 * MathF.Sin(eeYaw));
        var g2a = new SKPoint(ee.X - dx, ee.Y - dy);
        var g2b = new SKPoint(g2a.X + 15 * MathF.Cos(eeYaw), g2a.Y + 15 * MathF.Sin(eeYaw));
        canvas.DrawLine(g1a, g1b, gripPaint);
        canvas.DrawLine(g2a, g2b, gripPaint);

        using var img = SKImage.FromBitmap(bmp);
        using var data = img.Encode(SKEncodedImageFormat.Jpeg, 80);
        return data.ToArray();
    }

    static float Map(float v, float a1, float a2, float b1, float b2) =>
        b1 + (v - a1) * (b2 - b1) / (a2 - a1);

    static float DegToRad(float d) => (float)(Math.PI / 180.0) * d;
    static float RadToDeg(float r) => (float)(180.0 / Math.PI) * r;
    static float Dist(float x1, float y1, float x2, float y2) =>
        MathF.Sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}