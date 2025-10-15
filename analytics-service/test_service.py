"""
Example: Testing the Battery Analytics Service
Run this to verify the service is working correctly
"""

import httpx
import json
import asyncio
from datetime import datetime


async def test_service():
    """Test all endpoints of the analytics service"""
    
    base_url = "http://localhost:8000"
    
    print("=" * 70)
    print("Battery Analytics Service - Test Suite")
    print("=" * 70)
    print()
    
    async with httpx.AsyncClient() as client:
        
        # Test 1: Health Check
        print("Test 1: Health Check")
        print("-" * 70)
        try:
            response = await client.get(f"{base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Service Status: {data['status']}")
                print(f"✓ Version: {data['version']}")
                print(f"✓ Timestamp: {data['timestamp']}")
            else:
                print(f"✗ Health check failed: {response.status_code}")
        except Exception as e:
            print(f"✗ Error: {e}")
        print()
        
        # Test 2: List Alert Rules
        print("Test 2: List Alert Rules")
        print("-" * 70)
        try:
            response = await client.get(f"{base_url}/api/analytics/rules")
            if response.status_code == 200:
                data = response.json()
                print(f"✓ Found {data['count']} alert rules:")
                for rule in data['rules']:
                    print(f"  - {rule['name']} ({rule['severity']})")
            else:
                print(f"✗ Failed to list rules: {response.status_code}")
        except Exception as e:
            print(f"✗ Error: {e}")
        print()
        
        # Test 3: Battery Analytics (Basic)
        print("Test 3: Battery Analytics - Basic Query")
        print("-" * 70)
        try:
            query = {
                "limit": 10,
                "applyKalman": True,
                "applyZScore": True,
                "applyRules": True,
                "includeWeather": False
            }
            
            response = await client.post(
                f"{base_url}/api/analytics/battery",
                json=query,
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✓ Analyzed {data['summary']['totalDevices']} devices")
                print(f"  - Healthy: {data['summary']['healthyDevices']}")
                print(f"  - Warning: {data['summary']['warningDevices']}")
                print(f"  - Critical: {data['summary']['criticalDevices']}")
                print(f"  - Avg Capacity: {data['summary']['avgCapacity']:.1f}%")
                print(f"  - Total Alerts: {data['summary']['totalAlerts']}")
                print()
                
                print("Analytics Applied:")
                print(f"  - Kalman Filter: {data['analytics']['kalmanFilterApplied']}")
                print(f"  - Z-Score Analysis: {data['analytics']['zScoreAnalysisApplied']}")
                print(f"  - Rules Evaluated: {data['analytics']['rulesEvaluated']}")
                print(f"  - Anomalies Detected: {data['analytics']['anomaliesDetected']}")
                print()
                
                if data['devices']:
                    print("Sample Device (first result):")
                    device = data['devices'][0]
                    print(f"  Device ID: {device['device']}")
                    print(f"  Region: {device['region']}")
                    print(f"  Voltage: {device['voltage']}V → {device.get('filteredVoltage', 'N/A')}V (filtered)")
                    print(f"  Capacity: {device['capacity']}%")
                    print(f"  Temperature: {device['temperature']}°C")
                    print(f"  Health: {device['health']}")
                    print(f"  Z-Score: {device.get('voltageZScore', 'N/A')}")
                    print(f"  Alerts: {len(device['alerts'])}")
                    
                    if device['alerts']:
                        print("  Top Alert:")
                        alert = device['alerts'][0]
                        print(f"    - Type: {alert['alertType']}")
                        print(f"    - Severity: {alert['severity']}")
                        print(f"    - Message: {alert['message']}")
                
            else:
                print(f"✗ Analytics query failed: {response.status_code}")
                print(f"Response: {response.text}")
        except Exception as e:
            print(f"✗ Error: {e}")
        print()
        
        # Test 4: Battery Analytics with Weather
        print("Test 4: Battery Analytics - With Weather Correlation")
        print("-" * 70)
        try:
            query = {
                "region": "Asia-Pacific",
                "limit": 5,
                "applyKalman": True,
                "applyZScore": True,
                "applyRules": True,
                "includeWeather": True
            }
            
            response = await client.post(
                f"{base_url}/api/analytics/battery",
                json=query,
                timeout=30.0
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"✓ Weather Enriched: {data['analytics']['weatherEnriched']}")
                
                if data.get('regionalWeather'):
                    print()
                    print("Regional Weather:")
                    for region, weather in data['regionalWeather'].items():
                        print(f"  {region}:")
                        print(f"    Temperature: {weather['temperature']}°C")
                        print(f"    Conditions: {weather['conditions']}")
                        if weather.get('humidity'):
                            print(f"    Humidity: {weather['humidity']}%")
                
                # Check for weather-related alerts
                weather_alerts = []
                for device in data['devices']:
                    for alert in device['alerts']:
                        if 'WEATHER' in alert['alertType']:
                            weather_alerts.append(alert)
                
                if weather_alerts:
                    print()
                    print(f"✓ Found {len(weather_alerts)} weather-related alerts")
                    for alert in weather_alerts[:3]:  # Show first 3
                        print(f"  - {alert['message']}")
                else:
                    print()
                    print("✓ No weather-related alerts (devices operating normally)")
                
            else:
                print(f"✗ Weather query failed: {response.status_code}")
        except Exception as e:
            print(f"✗ Error: {e}")
        print()
        
        # Test 5: Regional Query
        print("Test 5: Battery Analytics - Regional Filter")
        print("-" * 70)
        try:
            for region in ["Asia-Pacific", "Europe", "North America"]:
                query = {
                    "region": region,
                    "limit": 20,
                    "applyKalman": False,
                    "applyZScore": False,
                    "applyRules": False
                }
                
                response = await client.post(
                    f"{base_url}/api/analytics/battery",
                    json=query,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    count = data['summary']['totalDevices']
                    avg_cap = data['summary']['avgCapacity']
                    print(f"✓ {region}: {count} devices, avg capacity {avg_cap:.1f}%")
                else:
                    print(f"✗ {region}: Query failed")
        except Exception as e:
            print(f"✗ Error: {e}")
        print()
    
    print("=" * 70)
    print("Test Suite Complete!")
    print("=" * 70)


if __name__ == "__main__":
    print()
    print("Make sure the service is running: python main.py")
    print()
    
    asyncio.run(test_service())
