"""
Tests for Weather Service Integration
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from analytics.weather import (
    WeatherService,
    get_location_for_region,
    get_regional_weather,
    REGION_LOCATIONS
)


class TestWeatherService:
    """Test WeatherService class"""
    
    @pytest.fixture
    def weather_service(self):
        """Create a WeatherService instance"""
        return WeatherService()
    
    @pytest.mark.asyncio
    async def test_get_current_weather_fallback(self, weather_service):
        """Test fallback weather data when API key is not configured"""
        # Without API key, should return fallback data
        location = "New York"
        weather_data = await weather_service.get_current_weather(location)
        
        assert weather_data is not None
        assert weather_data["location"] == location
        assert "temperature" in weather_data
        assert weather_data["temperature"] == 22.0
        assert "timestamp" in weather_data
        assert "Unknown" in weather_data["conditions"]
    
    @pytest.mark.asyncio
    async def test_get_current_weather_caching(self, weather_service):
        """Test that weather data is cached properly"""
        location = "London"
        
        # First call
        weather_data_1 = await weather_service.get_current_weather(location)
        
        # Second call (should use cache)
        weather_data_2 = await weather_service.get_current_weather(location)
        
        # Both should return same data
        assert weather_data_1["location"] == weather_data_2["location"]
        assert weather_data_1["temperature"] == weather_data_2["temperature"]
    
    def test_fallback_weather_structure(self, weather_service):
        """Test fallback weather data structure"""
        location = "Tokyo"
        fallback_data = weather_service._get_fallback_weather(location)
        
        # Check all required fields are present
        assert "location" in fallback_data
        assert "timestamp" in fallback_data
        assert "temperature" in fallback_data
        assert "humidity" in fallback_data
        assert "pressure" in fallback_data
        assert "precipitation" in fallback_data
        assert "windSpeed" in fallback_data
        assert "conditions" in fallback_data
        
        # Check location matches
        assert fallback_data["location"] == location
    
    def test_parse_openweather_response(self, weather_service):
        """Test parsing of OpenWeatherMap API response"""
        mock_response = {
            "main": {
                "temp": 25.5,
                "humidity": 65,
                "pressure": 1013
            },
            "wind": {
                "speed": 5.5  # m/s
            },
            "weather": [
                {"main": "Clear"}
            ],
            "rain": {
                "1h": 0.5
            }
        }
        
        location = "Paris"
        parsed_data = weather_service._parse_openweather_response(mock_response, location)
        
        assert parsed_data["location"] == location
        assert parsed_data["temperature"] == 25.5
        assert parsed_data["humidity"] == 65
        assert parsed_data["pressure"] == 1013
        assert parsed_data["precipitation"] == 0.5
        assert parsed_data["windSpeed"] == pytest.approx(19.8, rel=0.1)  # 5.5 m/s * 3.6
        assert parsed_data["conditions"] == "Clear"
    
    def test_parse_openweather_response_no_rain(self, weather_service):
        """Test parsing when no rain data is present"""
        mock_response = {
            "main": {
                "temp": 20.0,
                "humidity": 50,
                "pressure": 1015
            },
            "wind": {
                "speed": 3.0
            },
            "weather": [
                {"main": "Sunny"}
            ]
        }
        
        location = "Madrid"
        parsed_data = weather_service._parse_openweather_response(mock_response, location)
        
        assert parsed_data["precipitation"] == 0
        assert parsed_data["conditions"] == "Sunny"
    
    @pytest.mark.asyncio
    async def test_get_historical_weather(self, weather_service):
        """Test historical weather (placeholder implementation)"""
        location = "Berlin"
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        historical_data = await weather_service.get_historical_weather(
            location, start_date, end_date
        )
        
        # Currently returns fallback data
        assert isinstance(historical_data, list)
        assert len(historical_data) > 0
        assert historical_data[0]["location"] == location


class TestRegionMapping:
    """Test region to location mapping functions"""
    
    def test_get_location_for_region_known(self):
        """Test mapping known regions"""
        assert get_location_for_region("North America") == "New York,US"
        assert get_location_for_region("Asia-Pacific") == "Singapore,SG"
        assert get_location_for_region("Europe") == "Frankfurt,DE"
        assert get_location_for_region("South America") == "Sao Paulo,BR"
        assert get_location_for_region("Africa") == "Lagos,NG"
        assert get_location_for_region("Middle East") == "Dubai,AE"
    
    def test_get_location_for_region_unknown(self):
        """Test mapping unknown region returns the region itself"""
        unknown_region = "Antarctica"
        assert get_location_for_region(unknown_region) == unknown_region
    
    def test_region_locations_coverage(self):
        """Test that all expected regions are mapped"""
        expected_regions = [
            "North America",
            "Asia-Pacific",
            "Europe",
            "South America",
            "Africa",
            "Middle East"
        ]
        
        for region in expected_regions:
            assert region in REGION_LOCATIONS
            assert REGION_LOCATIONS[region] is not None


class TestRegionalWeather:
    """Test regional weather fetching"""
    
    @pytest.mark.asyncio
    async def test_get_regional_weather_single_region(self):
        """Test fetching weather for a single region"""
        regions = ["Europe"]
        regional_weather = await get_regional_weather(regions)
        
        assert "Europe" in regional_weather
        assert regional_weather["Europe"]["location"] == "Frankfurt,DE"
        assert "temperature" in regional_weather["Europe"]
    
    @pytest.mark.asyncio
    async def test_get_regional_weather_multiple_regions(self):
        """Test fetching weather for multiple regions"""
        regions = ["North America", "Asia-Pacific", "Europe"]
        regional_weather = await get_regional_weather(regions)
        
        assert len(regional_weather) == 3
        assert "North America" in regional_weather
        assert "Asia-Pacific" in regional_weather
        assert "Europe" in regional_weather
        
        # Check each region has valid data
        for region in regions:
            assert "temperature" in regional_weather[region]
            assert "timestamp" in regional_weather[region]
    
    @pytest.mark.asyncio
    async def test_get_regional_weather_empty_list(self):
        """Test fetching weather for empty region list"""
        regions = []
        regional_weather = await get_regional_weather(regions)
        
        assert regional_weather == {}
    
    @pytest.mark.asyncio
    async def test_get_regional_weather_unknown_region(self):
        """Test fetching weather for unknown region"""
        regions = ["Unknown Region"]
        regional_weather = await get_regional_weather(regions)
        
        assert "Unknown Region" in regional_weather
        assert regional_weather["Unknown Region"]["location"] == "Unknown Region"


class TestWeatherDataStructure:
    """Test weather data structure and validation"""
    
    @pytest.mark.asyncio
    async def test_weather_data_has_required_fields(self):
        """Test that weather data contains all required fields"""
        weather_service = WeatherService()
        weather_data = await weather_service.get_current_weather("Test City")
        
        required_fields = [
            "location",
            "timestamp",
            "temperature",
            "humidity",
            "pressure",
            "precipitation",
            "windSpeed",
            "conditions"
        ]
        
        for field in required_fields:
            assert field in weather_data, f"Missing required field: {field}"
    
    @pytest.mark.asyncio
    async def test_weather_timestamp_format(self):
        """Test that timestamp is in ISO format"""
        weather_service = WeatherService()
        weather_data = await weather_service.get_current_weather("Test City")
        
        timestamp = weather_data["timestamp"]
        # Should be able to parse as ISO format
        parsed_time = datetime.fromisoformat(timestamp)
        assert isinstance(parsed_time, datetime)
    
    @pytest.mark.asyncio
    async def test_weather_temperature_is_numeric(self):
        """Test that temperature is a numeric value"""
        weather_service = WeatherService()
        weather_data = await weather_service.get_current_weather("Test City")
        
        temperature = weather_data["temperature"]
        assert isinstance(temperature, (int, float))


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
