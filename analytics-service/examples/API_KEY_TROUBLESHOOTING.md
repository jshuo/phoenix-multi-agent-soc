# OpenWeatherMap API Key Troubleshooting

## Current Issue: 401 Unauthorized Error

The API key `9e370e21261fcedd37045581a12cbc55` is being rejected by OpenWeatherMap with a 401 error.

## Solutions

### Solution 1: Wait for API Key Activation (Most Common)
New API keys can take **2-10 minutes** to activate after creation.

**Steps:**
1. Wait 5-10 minutes after creating the API key
2. Run the test again: `python examples/test_api_key.py`
3. If successful, run the full example: `python examples/weather_example.py`

### Solution 2: Verify API Key is Correct

1. Go to: https://home.openweathermap.org/api_keys
2. Log in to your OpenWeatherMap account
3. Copy the API key **exactly** (no spaces before/after)
4. Update the `.env` file:
   ```
   WEATHER_API_KEY=your_exact_api_key_here
   ```
5. Run the test: `python examples/test_api_key.py`

### Solution 3: Generate a New API Key

If the current key doesn't work after 10 minutes:

1. Go to: https://home.openweathermap.org/api_keys
2. Create a new API key with a name like "Analytics Service"
3. Wait 5-10 minutes for activation
4. Copy the new key to `.env`
5. Test again

### Solution 4: Check API Key Type

Make sure you're using the **free tier API key** for:
- Current Weather Data API (v2.5)
- NOT the One Call API 3.0 (requires subscription)

## Testing Commands

```bash
# Quick API key test
cd /Users/jmh_cheng/workspace/phoenix-multi-agent-soc/analytics-service
python examples/test_api_key.py

# Full weather example
python examples/weather_example.py
```

## Expected Output (When Working)

```
✅ SUCCESS! API key is working!

Weather Data:
  Location: London, GB
  Temperature: 15.2°C
  Humidity: 72%
  Conditions: Clouds
  Description: overcast clouds
```

## Additional Resources

- OpenWeatherMap FAQ: https://openweathermap.org/faq#error401
- API Documentation: https://openweathermap.org/current
- Support: https://home.openweathermap.org/questions

## Current Status

- ✓ API key format is correct (32 characters)
- ✓ API key is being loaded from .env file
- ✓ Request is being sent correctly
- ❌ API key is not yet activated or is invalid

**Recommended Action:** Wait 5-10 minutes and test again, or generate a new API key.
