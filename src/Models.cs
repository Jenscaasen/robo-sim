using SkiaSharp;

public class Servo
{
    // Steps (integers) to keep parity with hardware
    public int CurrentStep { get; set; }
    public int TargetStep
    {
        get => _targetStep;
        set => _targetStep = Math.Clamp(value, MinStep, MaxStep);
    }
    private int _targetStep;

    public int MinStep { get; set; } = int.MinValue;
    public int MaxStep { get; set; } = int.MaxValue;
    public float Load { get; set; }

    public void Tick(double dt, float maxSpeedStepsPerSec)
    {
        float maxStepDelta = (float)(maxSpeedStepsPerSec * dt);
        float delta = TargetStep - CurrentStep;
        if (MathF.Abs(delta) <= maxStepDelta)
        {
            CurrentStep = TargetStep;
        }
        else
        {
            CurrentStep += (int)MathF.CopySign(maxStepDelta, delta);
        }
        CurrentStep = Math.Clamp(CurrentStep, MinStep, MaxStep);
    }
}

public class Block
{
    public float X;
    public float Y;
    public float YawDeg;
    public float Size;
    public SKColor Color;
    public bool Attached;

    // optional attached offset not strictly necessary for simple sim
    public float OffsetX;
    public float OffsetY;
    public float OffsetYawDeg;
}