using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Hosting;

public class SimLoop : BackgroundService
{
    private readonly SimWorld _world;
    public SimLoop(SimWorld world) => _world = world;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var sw = Stopwatch.StartNew();
        long last = sw.ElapsedMilliseconds;
        while (!stoppingToken.IsCancellationRequested)
        {
            long now = sw.ElapsedMilliseconds;
            double dt = Math.Max(0.0, (now - last) / 1000.0);
            last = now;

            try
            {
                _world.Tick(dt);
            }
            catch (Exception)
            {
                // swallow to keep loop alive
            }

            await Task.Delay(10, stoppingToken); // ~100 Hz
        }
    }
}