using System.Text.Json;
using System.Text.Json.Serialization;

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

public class DatasetCreator
{
    private readonly HttpClient _httpClient;
    private readonly string _baseUrl;
    private readonly Random _random;
    
    public DatasetCreator(string host = "127.0.0.1", int port = 5000)
    {
        _httpClient = new HttpClient();
        _baseUrl = $"http://{host}:{port}";
        _random = new Random();
    }
    
    public async Task<Dictionary<string, JointInfo>?> GetJointInfoAsync()
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/joints");
            response.EnsureSuccessStatusCode();
            
            var jsonString = await response.Content.ReadAsStringAsync();
            return JsonSerializer.Deserialize<Dictionary<string, JointInfo>>(jsonString);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error fetching joint info: {ex.Message}");
            return null;
        }
    }
    
    public async Task<bool> SetJointPositionInstantAsync(int jointId, double position)
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/joint/{jointId}/{position:F6}/instant");
            response.EnsureSuccessStatusCode();
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error setting joint {jointId} to position {position}: {ex.Message}");
            return false;
        }
    }
    
    public async Task<byte[]?> GetCameraImageAsync(int cameraId)
    {
        try
        {
            var response = await _httpClient.GetAsync($"{_baseUrl}/api/camera/{cameraId}");
            response.EnsureSuccessStatusCode();
            
            return await response.Content.ReadAsByteArrayAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error capturing camera {cameraId}: {ex.Message}");
            return null;
        }
    }
    
    public List<JointConfiguration> GenerateRandomJointPositions(Dictionary<string, JointInfo> joints)
    {
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
        
        return configurations;
    }
    
    public async Task<bool> CreateDatasetEntryAsync(int entryNumber, string datasetFolder, Dictionary<string, JointInfo> joints)
    {
        Console.WriteLine($"Creating dataset entry {entryNumber}...");
        
        // Generate random joint positions
        var jointConfigurations = GenerateRandomJointPositions(joints);
        
        // Set all joint positions
        int jointIndex = 0;
        foreach (var config in jointConfigurations)
        {
            var success = await SetJointPositionInstantAsync(jointIndex, config.Position);
            if (!success)
            {
                Console.WriteLine($"Failed to set joint {jointIndex} ({config.Name}) to position {config.Position}");
                return false;
            }
            jointIndex++;
        }
        
        // Small delay to ensure joints are set
        await Task.Delay(100);
        
        // Capture images from all 3 cameras
        for (int cameraId = 1; cameraId <= 3; cameraId++)
        {
            var imageData = await GetCameraImageAsync(cameraId);
            if (imageData == null)
            {
                Console.WriteLine($"Failed to capture image from camera {cameraId}");
                return false;
            }
            
            var imageFileName = Path.Combine(datasetFolder, $"datarow-{entryNumber:D5}-cam{cameraId}.jpg");
            await File.WriteAllBytesAsync(imageFileName, imageData);
            Console.WriteLine($"Saved image: {imageFileName}");
        }
        
        // Save joint configuration as JSON
        var jsonFileName = Path.Combine(datasetFolder, $"datarow-{entryNumber:D5}.json");
        var jsonOptions = new JsonSerializerOptions
        {
            WriteIndented = true
        };
        var jsonString = JsonSerializer.Serialize(jointConfigurations, jsonOptions);
        await File.WriteAllTextAsync(jsonFileName, jsonString);
        Console.WriteLine($"Saved joint configuration: {jsonFileName}");
        
        return true;
    }
    
    public void Dispose()
    {
        _httpClient?.Dispose();
    }
}

class Program
{
    static async Task<int> Main(string[] args)
    {
        if (args.Length != 1)
        {
            Console.WriteLine("Usage: DatasetCreator <number_of_dataset_entries>");
            Console.WriteLine("Example: DatasetCreator 100");
            return 1;
        }
        
        if (!int.TryParse(args[0], out int datasetCount) || datasetCount <= 0)
        {
            Console.WriteLine("Error: Please provide a valid positive number for dataset entries.");
            return 1;
        }
        
        var creator = new DatasetCreator();
        
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
            
            // Create dataset entries
            int successCount = 0;
            for (int i = 1; i <= datasetCount; i++)
            {
                var success = await creator.CreateDatasetEntryAsync(i, datasetFolder, joints);
                if (success)
                {
                    successCount++;
                    Console.WriteLine($"Progress: {i}/{datasetCount} entries completed");
                }
                else
                {
                    Console.WriteLine($"Failed to create entry {i}");
                }
                
                // Small delay between entries to avoid overwhelming the API
                if (i < datasetCount)
                {
                    await Task.Delay(50);
                }
            }
            
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