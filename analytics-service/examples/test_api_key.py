"""
Quick test to verify OpenWeatherMap API key is working
"""

import asyncio
import httpx
import os
from dotenv import load_dotenv

# Load environment variables
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

async def test_api_key():
    api_key = os.getenv("WEATHER_API_KEY")
    
    print("=" * 70)
    print("OpenWeatherMap API Key Test")
    print("=" * 70)
    
    if not api_key:
        print("\n❌ No API key found in environment variables")
        return
    
    print(f"\n✓ API Key found: {api_key[:8]}...{api_key[-4:]}")
    print(f"✓ API Key length: {len(api_key)} characters")
    
    # Test the API with a simple request
    print("\nTesting API connection with 'London' query...")
    
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": "London",
        "appid": api_key,
        "units": "metric"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            print(f"\nRequest URL: {url}")
            print(f"Request params: q=London, appid={api_key[:8]}..., units=metric")
            
            response = await client.get(url, params=params, timeout=10.0)
            
            print(f"\nResponse Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print("\n✅ SUCCESS! API key is working!")
                print("\nWeather Data:")
                print(f"  Location: {data['name']}, {data['sys']['country']}")
                print(f"  Temperature: {data['main']['temp']}°C")
                print(f"  Humidity: {data['main']['humidity']}%")
                print(f"  Conditions: {data['weather'][0]['main']}")
                print(f"  Description: {data['weather'][0]['description']}")
            elif response.status_code == 401:
                print("\n❌ UNAUTHORIZED (401)")
                print("   This means the API key is invalid or not activated yet.")
                print("\n   Possible reasons:")
                print("   1. API key hasn't been activated yet (can take a few minutes)")
                print("   2. API key was entered incorrectly")
                print("   3. API key has been revoked or expired")
                print("\n   Response body:")
                print(f"   {response.text}")
            elif response.status_code == 429:
                print("\n❌ RATE LIMIT EXCEEDED (429)")
                print("   You've exceeded the API rate limit")
            else:
                print(f"\n❌ ERROR: HTTP {response.status_code}")
                print(f"   Response: {response.text}")
    
    except httpx.TimeoutException:
        print("\n❌ Request timed out")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_api_key())
