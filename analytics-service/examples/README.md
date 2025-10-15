# Weather Service Examples

This directory contains examples demonstrating the Weather Service functionality.

## Setup

### 1. Get a Free OpenWeatherMap API Key

The weather examples require an API key from OpenWeatherMap (it's free!):

1. Go to: https://home.openweathermap.org/users/sign_up
2. Create a free account
3. After signup, go to: https://home.openweathermap.org/api_keys
4. Copy your API key

### 2. Configure the API Key

Edit the `.env` file in this directory and replace `YOUR_API_KEY_HERE` with your actual API key:

```bash
WEATHER_API_KEY=your_actual_api_key_here
```

### 3. Run the Examples

```bash
# From the analytics-service directory
cd /Users/jmh_cheng/workspace/phoenix-multi-agent-soc/analytics-service

# Run the weather example
python examples/weather_example.py
```

## What the Examples Demonstrate

The `weather_example.py` file includes 6 examples:

1. **Single Location Weather** - Fetch weather for one city
2. **Multiple Locations** - Fetch weather for multiple cities  
3. **Regional Weather** - Fetch weather for defined regions with automatic location mapping
4. **All Regions** - Fetch weather for all 6 available regions
5. **Caching Demonstration** - Shows how the weather service caches data
6. **Analytics Correlation** - Demonstrates correlation with system metrics

## Fallback Mode

If the API key is not configured or invalid, the examples will use fallback data with a default temperature of 22°C. This allows testing the integration without requiring a valid API key.

## Notes

- OpenWeatherMap free tier allows 1,000 API calls per day
- API key activation may take a few minutes after signup
- Weather data is cached for 30 minutes to reduce API calls
