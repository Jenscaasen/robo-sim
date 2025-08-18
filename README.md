# Robo-sim

Lightweight headless 2D robot simulator exposing a small HTTP API compatible with the Pi robot plan.

Run

1. Install .NET 8 SDK.
2. Restore and build:
   dotnet restore src
   dotnet build src
3. Run (HTTP):
   dotnet run --project src --urls "http://0.0.0.0:5332"
   Or to use HTTPS (dev cert required):
   dotnet dev-certs https --trust
   dotnet run --project src --urls "https://0.0.0.0:5332"

API examples

- GET /webcam/1/image
  curl -sS http://localhost:5332/webcam/1/image --output frame.jpg

- GET /state
  curl -sS http://localhost:5332/state | jq .

- GET servo current/target/load
  curl -sS http://localhost:5332/servo/1/currentStep
  curl -sS http://localhost:5332/servo/1/targetStep
  curl -sS http://localhost:5332/servo/1/load

- POST set target step
  curl -sS -X POST http://localhost:5332/servo/1/targetStep -H "Content-Type: application/json" -d '{"value":300}'

Notes

- 1 px = 1 mm mapping; world size 640x480.
- Servo mapping: 10 steps/deg, link lengths L1=150, L2=120, L3=80 mm.
- Gripper servo maps to width 0-50 mm from step range 0-500.