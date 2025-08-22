using System.Text;
using System.Text.Json;

namespace RoboSimWebUI.Services;

public class RoboSimApiService
{
    private readonly HttpClient _httpClient;
    private readonly JsonSerializerOptions _jsonOptions;
    private readonly Dictionary<int, double> _lastTargetPositions = new();

    public RoboSimApiService(IHttpClientFactory httpClientFactory)
    {
        _httpClient = httpClientFactory.CreateClient("RoboSimAPI");
        _jsonOptions = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };
    }

    // Get camera image as byte array (supports multiple camera IDs)
    public async Task<byte[]?> GetWebcamImageAsync(int cameraId = 1)
    {
        try
        {
            var response = await _httpClient.GetAsync($"/api/camera/{cameraId}");
            if (response.IsSuccessStatusCode)
            {
                return await response.Content.ReadAsByteArrayAsync();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error getting camera image (camera {cameraId}): {ex.Message}");
        }
        return null;
    }

    // Get all joints information
    public async Task<Dictionary<string, JointInfo>?> GetJointsAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync("/api/joints");
            if (response.IsSuccessStatusCode)
            {
                var json = await response.Content.ReadAsStringAsync();
                return JsonSerializer.Deserialize<Dictionary<string, JointInfo>>(json, _jsonOptions);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error getting joints: {ex.Message}");
        }
        return null;
    }

    // Set joint position (normal movement)
    public async Task<bool> SetJointPositionAsync(int jointId, double position)
    {
        try
        {
            var response = await _httpClient.GetAsync($"/api/joint/{jointId}/{position:F6}");
            if (response.IsSuccessStatusCode)
            {
                _lastTargetPositions[jointId] = position;
                return true;
            }
            return false;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error setting joint {jointId} position: {ex.Message}");
            return false;
        }
    }

    // Set joint position instantly
    public async Task<bool> SetJointPositionInstantAsync(int jointId, double position)
    {
        try
        {
            var response = await _httpClient.GetAsync($"/api/joint/{jointId}/{position:F6}/instant");
            if (response.IsSuccessStatusCode)
            {
                _lastTargetPositions[jointId] = position;
                return true;
            }
            return false;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error setting joint {jointId} position instantly: {ex.Message}");
            return false;
        }
    }

    // Check API health
    public async Task<bool> CheckHealthAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync("/api/health");
            return response.IsSuccessStatusCode;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error checking health: {ex.Message}");
            return false;
        }
    }

    // Get all joints with detailed information for UI display
    public async Task<JointsResponse?> GetAllJointsAsync()
    {
        try
        {
            var jointsDict = await GetJointsAsync();
            if (jointsDict != null)
            {
                var joints = jointsDict.Select(kvp => {
                    var jointId = int.Parse(kvp.Key);
                    var lastTarget = _lastTargetPositions.ContainsKey(jointId) ? _lastTargetPositions[jointId] : 0.0;
                    
                    return new DetailedJointInfo(
                        Id: jointId,
                        Name: kvp.Value.Name,
                        CurrentPosition: lastTarget, // Use last target as current position approximation
                        TargetPosition: lastTarget,  // Use last known target position
                        Lower: kvp.Value.Lower,
                        Upper: kvp.Value.Upper,
                        MaxForce: kvp.Value.MaxForce,
                        MaxVelocity: kvp.Value.MaxVelocity,
                        Type: kvp.Value.Type
                    );
                }).ToArray();

                return new JointsResponse(joints, DateTimeOffset.Now);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error getting all joints: {ex.Message}");
        }
        return null;
    }

    // Reset all joints to neutral position (0.0)
    public async Task<bool> ResetJointsAsync()
    {
        try
        {
            var jointsDict = await GetJointsAsync();
            if (jointsDict != null)
            {
                var tasks = new List<Task<bool>>();
                foreach (var kvp in jointsDict)
                {
                    var jointId = int.Parse(kvp.Key);
                    tasks.Add(SetJointPositionInstantAsync(jointId, 0.0));
                }
                
                var results = await Task.WhenAll(tasks);
                if (results.All(r => r))
                {
                    // Clear all stored target positions since we reset to 0
                    _lastTargetPositions.Clear();
                    return true;
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error resetting joints: {ex.Message}");
        }
        return false;
    }
}

// Response models for the new PyBullet API
public record JointInfo(
    double Lower,
    double Upper,
    double MaxForce,
    double MaxVelocity,
    string Name,
    int Type
);

public record DetailedJointInfo(
    int Id,
    string Name,
    double CurrentPosition,
    double TargetPosition,
    double Lower,
    double Upper,
    double MaxForce,
    double MaxVelocity,
    int Type
);

public record JointsResponse(
    DetailedJointInfo[] Joints,
    DateTimeOffset Timestamp
);

// Legacy models for backward compatibility (will be removed later)
public record ServoValueResponse(int Value, DateTimeOffset Timestamp);

public record SystemStateResponse(
    ServoInfo[] Servos,
    BlockInfo[] Blocks
);

public record ServoInfo(
    int Id,
    int CurrentStep,
    int TargetStep,
    double Load
);

public record BlockInfo(
    double X,
    double Y,
    double Yaw,
    double Size,
    bool Attached
);

public record ServosResponse(
    DetailedServoInfo[] Servos,
    DateTimeOffset Timestamp
);

public record DetailedServoInfo(
    int Id,
    string Name,
    int CurrentStep,
    int TargetStep,
    double Load,
    int MinStep,
    int MaxStep,
    double MinAngleDeg,
    double MaxAngleDeg,
    double StepsPerDeg
);