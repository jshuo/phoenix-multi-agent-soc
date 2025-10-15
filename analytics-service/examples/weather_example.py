"""
Weather Service Example
Demonstrates how to use the Weather Service to fetch weather data for different regions.
"""

import asyncio
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(env_path)

# Add parent directory to path to import analytics module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analytics.weather import (
    WeatherService,
    get_location_for_region,
    get_regional_weather,
    REGION_LOCATIONS
)


async def example_single_location():
    """Example: Fetch weather for a single location"""
    print("=" * 70)
    print("Example 1: Fetch Weather for a Single Location")
    print("=" * 70)
    
    weather_service = WeatherService()
    location = "New York"
    
    print(f"\nFetching weather for {location}...")
    weather_data = await weather_service.get_current_weather(location)
    
    print(f"\n📍 Location: {weather_data['location']}")
    print(f"🌡️  Temperature: {weather_data['temperature']}°C")
    print(f"💧 Humidity: {weather_data['humidity']}%")
    print(f"🌧️  Precipitation: {weather_data['precipitation']} mm")
    print(f"💨 Wind Speed: {weather_data['windSpeed']} km/h")
    print(f"☁️  Conditions: {weather_data['conditions']}")
    print(f"⏰ Timestamp: {weather_data['timestamp']}")


async def example_multiple_locations():
    """Example: Fetch weather for multiple locations"""
    print("\n" + "=" * 70)
    print("Example 2: Fetch Weather for Multiple Locations")
    print("=" * 70)
    
    weather_service = WeatherService()
    locations = ["London", "Tokyo", "Sydney", "Paris"]
    
    print(f"\nFetching weather for {len(locations)} cities...")
    
    for location in locations:
        weather_data = await weather_service.get_current_weather(location)
        print(f"\n📍 {weather_data['location']}: {weather_data['temperature']}°C, {weather_data['conditions']}")


async def example_regional_weather():
    """Example: Fetch weather for multiple regions"""
    print("\n" + "=" * 70)
    print("Example 3: Fetch Weather for Multiple Regions")
    print("=" * 70)
    
    regions = ["North America", "Europe", "Asia-Pacific", "South America"]
    
    print(f"\nFetching weather for {len(regions)} regions...")
    print("\nRegion Mappings:")
    for region in regions:
        location = get_location_for_region(region)
        print(f"  {region} → {location}")
    
    regional_weather = await get_regional_weather(regions)
    
    print("\nRegional Weather Data:")
    print("-" * 70)
    for region, weather_data in regional_weather.items():
        print(f"\n🌍 {region}")
        print(f"   Location: {weather_data['location']}")
        print(f"   Temperature: {weather_data['temperature']}°C")
        print(f"   Conditions: {weather_data['conditions']}")


async def example_all_regions():
    """Example: Fetch weather for all available regions"""
    print("\n" + "=" * 70)
    print("Example 4: Fetch Weather for All Available Regions")
    print("=" * 70)
    
    all_regions = list(REGION_LOCATIONS.keys())
    print(f"\nFetching weather for all {len(all_regions)} regions...")
    
    regional_weather = await get_regional_weather(all_regions)
    
    print("\nGlobal Weather Summary:")
    print("-" * 70)
    print(f"{'Region':<20} {'Location':<25} {'Temp (°C)':<12} {'Conditions'}")
    print("-" * 70)
    
    for region, weather_data in regional_weather.items():
        temp = weather_data['temperature']
        location = weather_data['location']
        conditions = weather_data['conditions']
        print(f"{region:<20} {location:<25} {temp:<12.1f} {conditions}")


async def example_caching_demonstration():
    """Example: Demonstrate weather data caching"""
    print("\n" + "=" * 70)
    print("Example 5: Demonstrate Weather Data Caching")
    print("=" * 70)
    
    weather_service = WeatherService()
    location = "Berlin"
    
    print(f"\nFetching weather for {location} (first call - will fetch from API or fallback)...")
    import time
    start_time = time.time()
    weather_data_1 = await weather_service.get_current_weather(location)
    time_1 = time.time() - start_time
    
    print(f"✓ First call completed in {time_1:.4f} seconds")
    print(f"  Temperature: {weather_data_1['temperature']}°C")
    
    print(f"\nFetching weather for {location} again (second call - should use cache)...")
    start_time = time.time()
    weather_data_2 = await weather_service.get_current_weather(location)
    time_2 = time.time() - start_time
    
    print(f"✓ Second call completed in {time_2:.4f} seconds")
    print(f"  Temperature: {weather_data_2['temperature']}°C")
    
    print(f"\n📊 Performance comparison:")
    print(f"   First call:  {time_1:.4f}s")
    print(f"   Second call: {time_2:.4f}s (cached)")
    if time_2 < time_1:
        speedup = time_1 / time_2
        print(f"   Speedup: {speedup:.2f}x faster with cache!")


async def example_weather_for_analytics():
    """Example: Fetch weather data suitable for analytics correlation"""
    print("\n" + "=" * 70)
    print("Example 6: Weather Data for Analytics Correlation")
    print("=" * 70)
    
    print("\nFetching weather data for datacenter regions...")
    
    datacenter_regions = {
        "us-east-1": "Virginia,US",
        "eu-west-1": "Dublin,IE",
        "ap-southeast-1": "Singapore,SG",
        "us-west-2": "Oregon,US"
    }
    
    weather_service = WeatherService()
    
    print("\n📊 Weather Data for Correlation with System Metrics:")
    print("-" * 70)
    
    for region_code, location in datacenter_regions.items():
        weather_data = await weather_service.get_current_weather(location)
        
        print(f"\n🏢 Region: {region_code}")
        print(f"   Location: {weather_data['location']}")
        print(f"   Temperature: {weather_data['temperature']}°C")
        print(f"   Humidity: {weather_data['humidity']}%")
        print(f"   Pressure: {weather_data['pressure']} hPa")
        
        # Demonstrate how this data could be used for analytics
        temp = weather_data['temperature']
        if temp is not None and temp > 30:
            print(f"   ⚠️  High temperature detected - may correlate with cooling load")
        elif temp is not None and temp < 10:
            print(f"   ℹ️  Low temperature - reduced cooling requirements expected")
        else:
            print(f"   ✓ Normal temperature range")


async def main():
    """Run all weather service examples"""
    print("\n" + "=" * 70)
    print("WEATHER SERVICE EXAMPLES")
    print("=" * 70)
    
    # Check if API key is configured
    api_key = os.getenv("WEATHER_API_KEY")
    if api_key:
        print(f"\n✓ Weather API key configured: {api_key[:8]}...{api_key[-4:]}")
        print("  Using real weather data from OpenWeatherMap API")
    else:
        print("\n⚠️  Weather API key not configured - using fallback data")
        print("  To use real weather data, set the WEATHER_API_KEY in .env file")
    
    print("=" * 70)
    
    try:
        # Run all examples
        await example_single_location()
        await example_multiple_locations()
        await example_regional_weather()
        await example_all_regions()
        await example_caching_demonstration()
        await example_weather_for_analytics()
        
        print("\n" + "=" * 70)
        print("✓ All examples completed successfully!")
        print("=" * 70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
