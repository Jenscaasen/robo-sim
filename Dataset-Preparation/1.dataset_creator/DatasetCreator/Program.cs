using System.Diagnostics;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using System.Threading;

namespace DatasetCreator;

public class JointInfo
{
    [JsonPropertyName("lower")]
    public double Lower { get; set; }
    
    [JsonPropertyName("upper")]
    public double Upper { get; set; }
    
    [JsonPropertyName("maxForce")]
    public double MaxForce { get; set; }
    
    [JsonPropertyName("maxVelocity")]
    public double MaxVelocity { get; set; }
    
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;
    
    [JsonPropertyName("type")]
    public int Type { get; set; }
}

public class JointConfiguration
{
    [JsonPropertyName("name")]
    public string Name { get; set; } = string.Empty;
    
    [JsonPropertyName("position")]
    public double Position { get; set; }
}

public class JointMoveRequest
{
    [JsonPropertyName("id")]
    public int Id { get; set; }
    
    [JsonPropertyName("pos")]
    public double Pos { get; set; }
}

public class JointMoveResult
{
    [JsonPropertyName("actual_position")]
    public double ActualPosition { get; set; }
    
    [JsonPropertyName("applied")]
    public double Applied { get; set; }
    
    [JsonPropertyName("joint_id")]
    public int JointId { get; set; }
    
    [JsonPropertyName("requested")]
    public double Requested { get; set; }
}

public class FastMoveResponse
{
    [JsonPropertyName("fast_mode")]
    public bool FastMode { get; set; }
    
    [JsonPropertyName("joints_moved")]
    public int JointsMoved { get; set; }
    
    [JsonPropertyName("movement_completed")]
    public bool MovementCompleted { get; set; }
    
    [JsonPropertyName("results")]
    public List<JointMoveResult> Results { get; set; } = new List<JointMoveResult>();
}

public class DatasetCreator : IDisposable
{
    private readonly List<HttpClient> _httpClients = new List<HttpClient>();
    private readonly List<string> _baseUrls = new List<string>();
    private readonly Random _random;
    private readonly Stopwatch _totalStopwatch = new Stopwatch();
    private readonly Stopwatch _stepStopwatch = new Stopwatch();
    private int _currentInstanceIndex = 0;
    private readonly object _instanceLock = new object();

    // Single instance constructor (backward compatibility)
    public DatasetCreator(string host = "127.0.0.1", int port = 5000)
    {
        _httpClients.Add(new HttpClient());
        _baseUrls.Add($"http://{host}:{port}");
        _random = new Random();
    }

    // Multiple instances constructor for direct port access (5001-5010)
    public DatasetCreator(int instanceCount = 10)
    {
        _random = new Random();

        // Create clients for ports 5001 to 5001+instanceCount-1
        for (int i = 0; i < instanceCount; i++)
        {
            int port = 5001 + i;
            _httpClients.Add(new HttpClient());
            _baseUrls.Add($"http://127.0.0.1:{port}");
            Console.WriteLine($"Created client for port {port}");
        }
    }

    // Get client for a specific entry (consistent instance per entry)
    private (HttpClient client, string baseUrl) GetClientForEntry(int entryNumber)
    {
        // Use consistent instance for each entry
        int instanceIndex = entryNumber % _httpClients.Count;
        var client = _httpClients[instanceIndex];
        var url = _baseUrls[instanceIndex];
        Console.WriteLine($"[ENTRY {entryNumber}] Using instance on port {url.Split(':').Last()}");
        return (client, url);
    }

    public async Task<Dictionary<string, JointInfo>?> GetJointInfoAsync()
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            Console.WriteLine($"[TIMING] Starting GetJointInfoAsync...");
            // Always use the first instance for joint info (only need to get this once)
            var client = _httpClients[0];
            var baseUrl = _baseUrls[0];
            var response = await client.GetAsync($"{baseUrl}/api/joints");
            response.EnsureSuccessStatusCode();

            var jsonString = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<Dictionary<string, JointInfo>>(jsonString);

            stopwatch.Stop();
            Console.WriteLine($"[TIMING] GetJointInfoAsync completed in {stopwatch.ElapsedMilliseconds}ms");
            return result;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            Console.WriteLine($"[TIMING] GetJointInfoAsync failed after {stopwatch.ElapsedMilliseconds}ms");
            Console.WriteLine($"Error fetching joint info: {ex.Message}");
            return null;
        }
    }
    
    public async Task<bool> ResetRobotInstantAsync(int entryNumber)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            Console.WriteLine($"[ENTRY {entryNumber}] [TIMING] Starting ResetRobotInstantAsync...");
            var (client, baseUrl) = GetClientForEntry(entryNumber);
            var response = await client.GetAsync($"{baseUrl}/api/reset/instant");
            response.EnsureSuccessStatusCode();

            stopwatch.Stop();
            Console.WriteLine($"[ENTRY {entryNumber}] [TIMING] ResetRobotInstantAsync completed in {stopwatch.ElapsedMilliseconds}ms");
            return true;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            Console.WriteLine($"[ENTRY {entryNumber}] [TIMING] ResetRobotInstantAsync failed after {stopwatch.ElapsedMilliseconds}ms");
            Console.WriteLine($"[ENTRY {entryNumber}] Error resetting robot: {ex.Message}");
            return false;
        }
    }

    public async Task<FastMoveResponse?> SetMultipleJointsFastAsync(int entryNumber, List<JointMoveRequest> jointRequests)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            Console.WriteLine($"[ENTRY {entryNumber}] [TIMING] Starting SetMultipleJointsFastAsync with {jointRequests.Count} joints...");
            var (client, baseUrl) = GetClientForEntry(entryNumber);
            var jsonContent = JsonSerializer.Serialize(jointRequests);
            var content = new StringContent(jsonContent, Encoding.UTF8, "application/json");

            var response = await client.PostAsync($"{baseUrl}/api/joints/instant", content);
            response.EnsureSuccessStatusCode();

            var responseJson = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<FastMoveResponse>(responseJson);

            stopwatch.Stop();
            Console.WriteLine($"[ENTRY {entryNumber}] [TIMING] SetMultipleJointsFastAsync completed in {stopwatch.ElapsedMilliseconds}ms");
            return result;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            Console.WriteLine($"[ENTRY {entryNumber}] [TIMING] SetMultipleJointsFastAsync failed after {stopwatch.ElapsedMilliseconds}ms");
            Console.WriteLine($"[ENTRY {entryNumber}] Error setting multiple joints: {ex.Message}");
            return null;
        }
    }

    public async Task<byte[]?> GetCameraImageAsync(int entryNumber, int cameraId)
    {
        var stopwatch = Stopwatch.StartNew();
        try
        {
            Console.WriteLine($"[ENTRY {entryNumber}] [TIMING] Starting GetCameraImageAsync for camera {cameraId}...");
            var (client, baseUrl) = GetClientForEntry(entryNumber);
            var response = await client.GetAsync($"{baseUrl}/api/camera/{cameraId}");
            response.EnsureSuccessStatusCode();

            var result = await response.Content.ReadAsByteArrayAsync();

            stopwatch.Stop();
            Console.WriteLine($"[ENTRY {entryNumber}] [TIMING] GetCameraImageAsync for camera {cameraId} completed in {stopwatch.ElapsedMilliseconds}ms, image size: {result.Length} bytes");
            return result;
        }
        catch (Exception ex)
        {
            stopwatch.Stop();
            Console.WriteLine($"[ENTRY {entryNumber}] [TIMING] GetCameraImageAsync for camera {cameraId} failed after {stopwatch.ElapsedMilliseconds}ms");
            Console.WriteLine($"[ENTRY {entryNumber}] Error capturing camera {cameraId}: {ex.Message}");
            return null;
        }
    }
    
    public List<JointConfiguration> GenerateRandomJointPositions(Dictionary<string, JointInfo> joints)
    {
        var stopwatch = Stopwatch.StartNew();
        Console.WriteLine($"[TIMING] Starting GenerateRandomJointPositions for {joints.Count} joints...");

        var configurations = new List<JointConfiguration>();

        foreach (var joint in joints)
        {
            var lower = joint.Value.Lower;
            var upper = joint.Value.Upper;
            var randomPosition = lower + (_random.NextDouble() * (upper - lower));

            configurations.Add(new JointConfiguration
            {
                Name = joint.Value.Name,
                Position = randomPosition
            });
        }

        stopwatch.Stop();
        Console.WriteLine($"[TIMING] GenerateRandomJointPositions completed in {stopwatch.ElapsedMilliseconds}ms");
        return configurations;
    }
    
    public async Task<bool> CreateDatasetEntryAsync(int entryNumber, string datasetFolder, Dictionary<string, JointInfo> joints)
    {
        _totalStopwatch.Restart();
        Console.WriteLine($"[ENTRY {entryNumber}] Starting dataset entry creation...");

        // Get client for this specific entry (consistent instance)
        var (client, baseUrl) = GetClientForEntry(entryNumber);

        // Generate random joint positions
        var jointConfigurations = GenerateRandomJointPositions(joints);

        // Reset robot to initial state before setting new positions
        var resetSuccess = await ResetRobotInstantAsync(entryNumber);
        if (!resetSuccess)
        {
            Console.WriteLine($"[ENTRY {entryNumber}] Failed to reset robot before setting joint positions");
            return false;
        }

        // Prepare joint move requests for fast endpoint
        _stepStopwatch.Restart();
        Console.WriteLine($"[ENTRY {entryNumber}] Preparing joint move requests...");
        var jointRequests = new List<JointMoveRequest>();
        int jointIndex = 0;
        foreach (var config in jointConfigurations)
        {
            jointRequests.Add(new JointMoveRequest
            {
                Id = jointIndex,
                Pos = config.Position
            });
            jointIndex++;
        }
        _stepStopwatch.Stop();
        Console.WriteLine($"[ENTRY {entryNumber}] Prepared {jointRequests.Count} joint move requests in {_stepStopwatch.ElapsedMilliseconds}ms");

        // Set all joint positions using fast endpoint
        var moveResponse = await SetMultipleJointsFastAsync(entryNumber, jointRequests);
        if (moveResponse == null || !moveResponse.MovementCompleted)
        {
            Console.WriteLine($"[ENTRY {entryNumber}] Failed to set joint positions using fast endpoint");
            return false;
        }

        // Update joint configurations with actual positions from response
        _stepStopwatch.Restart();
        Console.WriteLine($"[ENTRY {entryNumber}] Updating joint configurations with actual positions...");
        for (int i = 0; i < jointConfigurations.Count && i < moveResponse.Results.Count; i++)
        {
            var result = moveResponse.Results.FirstOrDefault(r => r.JointId == i);
            if (result != null)
            {
                jointConfigurations[i].Position = result.ActualPosition;
            }
        }
        _stepStopwatch.Stop();
        Console.WriteLine($"[ENTRY {entryNumber}] Updated joint configurations in {_stepStopwatch.ElapsedMilliseconds}ms");

        Console.WriteLine($"[ENTRY {entryNumber}] Successfully moved {moveResponse.JointsMoved} joints in fast mode");

        // Small delay to ensure joints are set
        _stepStopwatch.Restart();
        Console.WriteLine($"[ENTRY {entryNumber}] Waiting 100ms to ensure joints are set...");
        await Task.Delay(100);
        _stepStopwatch.Stop();
        Console.WriteLine($"[ENTRY {entryNumber}] Delay completed in {_stepStopwatch.ElapsedMilliseconds}ms");

        // Capture images from all 3 cameras
        for (int cameraId = 1; cameraId <= 3; cameraId++)
        {
            _stepStopwatch.Restart();
            Console.WriteLine($"[ENTRY {entryNumber}] Capturing image from camera {cameraId}...");
            var imageData = await GetCameraImageAsync(entryNumber, cameraId);
            if (imageData == null)
            {
                Console.WriteLine($"[ENTRY {entryNumber}] Failed to capture image from camera {cameraId}");
                return false;
            }

            var imageFileName = Path.Combine(datasetFolder, $"datarow-{entryNumber:D5}-cam{cameraId}.jpg");
            _stepStopwatch.Restart();
            Console.WriteLine($"[ENTRY {entryNumber}] Saving image from camera {cameraId} to {imageFileName}...");
            await File.WriteAllBytesAsync(imageFileName, imageData);
            _stepStopwatch.Stop();
            Console.WriteLine($"[ENTRY {entryNumber}] Saved camera {cameraId} image ({imageData.Length} bytes) in {_stepStopwatch.ElapsedMilliseconds}ms");
        }

        // Save joint configuration as JSON
        _stepStopwatch.Restart();
        var jsonFileName = Path.Combine(datasetFolder, $"datarow-{entryNumber:D5}.json");
        Console.WriteLine($"[ENTRY {entryNumber}] Saving joint configuration to {jsonFileName}...");
        var jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true
        };
        var jsonString = JsonSerializer.Serialize(jointConfigurations, jsonOptions);
        await File.WriteAllTextAsync(jsonFileName, jsonString);
        _stepStopwatch.Stop();
        Console.WriteLine($"[ENTRY {entryNumber}] Saved joint configuration ({jsonString.Length} chars) in {_stepStopwatch.ElapsedMilliseconds}ms");

        _totalStopwatch.Stop();
        Console.WriteLine($"[ENTRY {entryNumber}] COMPLETED in {_totalStopwatch.ElapsedMilliseconds}ms total time");
        return true;
    }
    
    public void Dispose()
    {
        foreach (var client in _httpClients)
        {
            client?.Dispose();
        }
    }
}

class Program
{
    static async Task<int> Main(string[] args)
    {
        if (args.Length < 1 || args.Length > 2)
        {
            Console.WriteLine("Usage: DatasetCreator <number_of_dataset_entries> [parallel_tasks]");
            Console.WriteLine("Example: DatasetCreator 100 10  # 100 entries with 10 parallel tasks");
            return 1;
        }

        if (!int.TryParse(args[0], out int datasetCount) || datasetCount <= 0)
        {
            Console.WriteLine("Error: Please provide a valid positive number for dataset entries.");
            return 1;
        }

        // Parse parallel tasks parameter (default to 1 for backward compatibility)
        int parallelTasks = 1;
        if (args.Length == 2)
        {
            if (!int.TryParse(args[1], out parallelTasks) || parallelTasks <= 0)
            {
                Console.WriteLine("Error: Parallel tasks must be a positive number. Using default (1).");
                parallelTasks = 1;
            }
        }

        // Use 10 instances (ports 5001-5010)
        var creator = new DatasetCreator(10);

        try
        {
            Console.WriteLine("Connecting to PyBullet API...");

            // Get joint information
            var joints = await creator.GetJointInfoAsync();
            if (joints == null)
            {
                Console.WriteLine("Failed to retrieve joint information from API. Make sure the PyBullet viewer is running.");
                return 1;
            }

            Console.WriteLine($"Found {joints.Count} joints:");
            foreach (var joint in joints)
            {
                Console.WriteLine($"  Joint {joint.Key}: {joint.Value.Name} (range: {joint.Value.Lower:F3} to {joint.Value.Upper:F3})");
            }

            // Create dataset folder with timestamp
            var timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
            var datasetFolder = Path.Combine("datasets", $"dataset_{timestamp}");
            Directory.CreateDirectory(datasetFolder);
            Console.WriteLine($"Created dataset folder: {datasetFolder}");

            // Create dataset entries in parallel
            int successCount = 0;
            var options = new ParallelOptions { MaxDegreeOfParallelism = parallelTasks };

            Console.WriteLine($"Creating {datasetCount} dataset entries with {parallelTasks} parallel tasks...");
            Console.WriteLine($"All requests will be sent to port 5000 and distributed by the load balancer");

            // Use SemaphoreSlim to limit concurrency and add small delays between batches
            var semaphore = new SemaphoreSlim(parallelTasks);
            var tasks = new List<Task>();

            for (int i = 1; i <= datasetCount; i++)
            {
                int entryNum = i; // Capture loop variable
                await semaphore.WaitAsync();

                tasks.Add(Task.Run(async () =>
                {
                    try
                    {
                        var success = await creator.CreateDatasetEntryAsync(entryNum, datasetFolder, joints);
                        if (success)
                        {
                            Interlocked.Increment(ref successCount);
                            Console.WriteLine($"Progress: {entryNum}/{datasetCount} entries completed");
                        }
                        else
                        {
                            Console.WriteLine($"Failed to create entry {entryNum}");
                        }
                    }
                    finally
                    {
                        semaphore.Release();

                        // Small delay between entries to help with load balancing
                        if (entryNum < datasetCount)
                        {
                            await Task.Delay(20); // Reduced delay since we're using semaphore
                        }
                    }
                }));
            }

            await Task.WhenAll(tasks);

            Console.WriteLine($"\nDataset creation completed!");
            Console.WriteLine($"Successfully created {successCount}/{datasetCount} entries");
            Console.WriteLine($"Dataset saved in: {datasetFolder}");

            return successCount == datasetCount ? 0 : 1;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Unexpected error: {ex.Message}");
            return 1;
        }
        finally
        {
            creator.Dispose();
        }
    }
}