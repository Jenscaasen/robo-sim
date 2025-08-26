#!/usr/bin/env python3
"""
Debug script to compare manual HTTP requests vs simulator_client.py behavior.

Run this while the PyBullet HTTP server is running (the same server used by remotecontrol.http).
It will:
 - Send an explicit manual POST with httpx (json= and data= variants)
 - Send the same payload via the SimulatorClient internals (both using its helper and directly via its httpx client)
 - Print the raw request content bytes and the JSON responses so you can compare exactly what's sent
"""

import asyncio
import json
import httpx

from simulator_client import SimulatorClient

BASE_URL = "http://127.0.0.1:5000"
ENDPOINT_FAST = "/api/joints/fast"
ENDPOINT_INSTANT = "/api/joints/instant"

PAYLOAD = [
    {"id": 0, "pos": 0.1},
    {"id": 1, "pos": 0.9},
    {"id": 2, "pos": 0.6},
    {"id": 3, "pos": 0.2},
    {"id": 4, "pos": 0.0},
]


async def reset_sim():
    """Reset all joints to neutral using the instant reset endpoint."""
    print("== Reset: /api/reset/instant ==")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{BASE_URL}/api/reset/instant")
            print("Reset status:", resp.status_code)
            try:
                print("Reset response JSON:", resp.json())
            except Exception:
                print("Reset response text:", resp.text)
        except Exception as e:
            print("Reset request failed:", e)
    # Small delay to let simulator apply reset
    await asyncio.sleep(0.05)
 
 
async def send_manual_httpx_json():
    print("== Manual httpx POST (json=PAYLOAD) to /api/joints/fast ==")
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{BASE_URL}{ENDPOINT_FAST}", json=PAYLOAD,
                                 headers={"Accept": "application/json"})
        print("Status:", resp.status_code)
        try:
            print("Response JSON:", resp.json())
        except Exception:
            print("Response text:", resp.text)
        # Raw request content as seen by httpx
        req_content = resp.request.content
        try:
            print("Raw request content (decoded):", req_content.decode("utf-8"))
        except Exception:
            print("Raw request content (bytes):", req_content)
        print()
 
 
async def send_manual_httpx_data():
    print("== Manual httpx POST (data=json.dumps(PAYLOAD)) to /api/joints/fast ==")
    async with httpx.AsyncClient() as client:
        body = json.dumps(PAYLOAD)
        resp = await client.post(f"{BASE_URL}{ENDPOINT_FAST}", data=body,
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
        print("Status:", resp.status_code)
        try:
            print("Response JSON:", resp.json())
        except Exception:
            print("Response text:", resp.text)
        req_content = resp.request.content
        try:
            print("Raw request content (decoded):", req_content.decode("utf-8"))
        except Exception:
            print("Raw request content (bytes):", req_content)
        print()
 
 
async def send_via_simulator_client_direct_post():
    print("== Using SimulatorClient.client.post directly (json=PAYLOAD) to /api/joints/fast ==")
    sim = SimulatorClient(host="127.0.0.1", port=5000)
    try:
        resp = await sim.client.post(f"{sim.base_url}{ENDPOINT_FAST}", json=PAYLOAD,
                                     headers={"Accept": "application/json"})
        print("Status:", resp.status_code)
        try:
            print("Response JSON:", resp.json())
        except Exception:
            print("Response text:", resp.text)
        req_content = resp.request.content
        try:
            print("Raw request content (decoded):", req_content.decode("utf-8"))
        except Exception:
            print("Raw request content (bytes):", req_content)
    finally:
        await sim.close()
    print()
 
 
async def send_via_simulator_helper():
    print("== Using SimulatorClient.set_multiple_joints(payload) helper ==")
    sim = SimulatorClient(host="127.0.0.1", port=5000)
    try:
        # Call the helper which wraps the internal post logic
        result = await sim.set_multiple_joints(PAYLOAD)
        print("Result from helper (parsed JSON):", result)
    except Exception as e:
        print("Helper raised exception:", e)
    finally:
        await sim.close()
    print()
 
 
async def main():
    print("Payload to send:")
    print(json.dumps(PAYLOAD, indent=2))
    print("\nMake sure your simulator HTTP server is running before continuing.\n")
 
    # Reset before the entire suite
    await reset_sim()
 
    # Manual json variant
    await reset_sim()
    await send_manual_httpx_json()
 
    # Manual data variant
    await reset_sim()
    await send_manual_httpx_data()
 
    # Direct post via SimulatorClient's httpx client
    await reset_sim()
    await send_via_simulator_client_direct_post()
 
    # Helper wrapper call
    await reset_sim()
    await send_via_simulator_helper()
 
    print("Done. Compare raw request content and responses above to find differences.")
 
 
if __name__ == "__main__":
    asyncio.run(main())