"""
Weather Service Integration
Fetch and cache external weather data
"""

import httpx
import os
from typing import Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Weather cache to minimize API calls
_weather_cache = {}
CACHE_TTL = timedelta(minutes=30)


class WeatherService:
    """
    Weather service for fetching external weather data.
    Supports OpenWeatherMap API (can be extended for other providers).
    """
    
    def __init__(self):
        self.api_key = os.getenv("WEATHER_API_KEY")
        self.api_url = os.getenv("WEATHER_API_URL", "https://api.openweathermap.org/data/2.5")
        self.timeout = 10.0
    
    async def get_current_weather(self, location: str) -> dict:
        """
        Fetch current weather data for a location.
        
        Args:
            location: City name, coordinates, or region
        
        Returns:
            Dictionary with weather data
        """
        # Check cache first
        cache_key = f"{location}-{datetime.now().strftime('%Y-%m-%d-%H')}"
        if cache_key in _weather_cache:
            cached_data, cached_time = _weather_cache[cache_key]
            if datetime.now() - cached_time < CACHE_TTL:
                logger.debug(f"Returning cached weather for {location}")
                return cached_data
        
        # Check if API key is configured
        if not self.api_key:
            logger.warning("Weather API key not configured, returning fallback data")
            return self._get_fallback_weather(location)
        
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.api_url}/weather"
                params = {
                    "q": location,
                    "appid": self.api_key,
                    "units": "metric"  # Celsius
                }
                
                response = await client.get(url, params=params, timeout=self.timeout)
                
                if response.status_code == 200:
                    data = response.json()
                    weather_data = self._parse_openweather_response(data, location)
                    
                    # Cache the result
                    _weather_cache[cache_key] = (weather_data, datetime.now())
                    
                    logger.info(f"Weather fetched for {location}: {weather_data['temperature']}°C")
                    return weather_data
                else:
                    logger.error(f"Weather API returned status {response.status_code}")
                    return self._get_fallback_weather(location)
                    
        except httpx.TimeoutException:
            logger.error(f"Weather API timeout for {location}")
            return self._get_fallback_weather(location)
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return self._get_fallback_weather(location)
    
    def _parse_openweather_response(self, data: dict, location: str) -> dict:
        """Parse OpenWeatherMap API response"""
        return {
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "precipitation": data.get("rain", {}).get("1h", 0),
            "windSpeed": data["wind"]["speed"] * 3.6,  # m/s to km/h
            "conditions": data["weather"][0]["main"]
        }
    
    def _get_fallback_weather(self, location: str) -> dict:
        """Return fallback weather data when API is unavailable"""
        return {
            "location": location,
            "timestamp": datetime.now().isoformat(),
            "temperature": 22.0,
            "humidity": None,
            "pressure": None,
            "precipitation": None,
            "windSpeed": None,
            "conditions": "Unknown (API unavailable)"
        }
    
    async def get_historical_weather(
        self, 
        location: str, 
        start_date: datetime, 
        end_date: datetime
    ) -> list:
        """
        Fetch historical weather data (requires premium API plan).
        
        Args:
            location: City name or coordinates
            start_date: Start date for historical data
            end_date: End date for historical data
        
        Returns:
            List of weather data points
        """
        # Note: Historical weather requires OpenWeather OneCall API (paid)
        # This is a placeholder for future implementation
        logger.warning("Historical weather API not yet implemented")
        return [self._get_fallback_weather(location)]


# Region to location mapping
REGION_LOCATIONS = {
    "North America": "New York,US",
    "Asia-Pacific": "Singapore,SG",
    "Europe": "Frankfurt,DE",
    "South America": "Sao Paulo,BR",
    "Africa": "Lagos,NG",
    "Middle East": "Dubai,AE"
}


def get_location_for_region(region: str) -> str:
    """
    Map region name to a representative city for weather lookup.
    
    Args:
        region: Region name
    
    Returns:
        City name suitable for weather API
    """
    return REGION_LOCATIONS.get(region, region)


async def get_regional_weather(regions: list) -> dict:
    """
    Fetch weather for multiple regions.
    
    Args:
        regions: List of region names
    
    Returns:
        Dictionary mapping regions to weather data
    """
    weather_service = WeatherService()
    regional_weather = {}
    
    for region in regions:
        location = get_location_for_region(region)
        weather_data = await weather_service.get_current_weather(location)
        regional_weather[region] = weather_data
    
    return regional_weather
