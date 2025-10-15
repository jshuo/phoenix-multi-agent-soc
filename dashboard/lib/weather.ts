/**
 * Weather Service Integration
 * Fetch and cache external weather data
 */

interface WeatherData {
  location: string;
  timestamp: string;
  temperature: number;
  humidity: number | null;
  pressure: number | null;
  precipitation: number | null;
  windSpeed: number | null;
  conditions: string;
}

interface CacheEntry {
  data: WeatherData;
  timestamp: Date;
}

// Weather cache to minimize API calls
const weatherCache = new Map<string, CacheEntry>();
const CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes

/**
 * Region to location mapping
 */
export const REGION_LOCATIONS: Record<string, string> = {
  "North America": "New York,US",
  "Asia-Pacific": "Singapore,SG",
  "Europe": "Frankfurt,DE",
  "South America": "Sao Paulo,BR",
  "Africa": "Lagos,NG",
  "Middle East": "Dubai,AE",
};

/**
 * Region to city display name mapping
 */
export const REGION_CITY_NAMES: Record<string, string> = {
  "North America": "New York",
  "Asia-Pacific": "Singapore",
  "Europe": "Frankfurt",
  "South America": "São Paulo",
  "Africa": "Lagos",
  "Middle East": "Dubai",
};

/**
 * Weather service for fetching external weather data.
 * Supports OpenWeatherMap API.
 */
export class WeatherService {
  private apiKey: string | undefined;
  private apiUrl: string;
  private timeout: number;

  constructor() {
    this.apiKey = process.env.WEATHER_API_KEY;
    this.apiUrl = process.env.WEATHER_API_URL || "https://api.openweathermap.org/data/2.5";
    this.timeout = 10000; // 10 seconds
  }

  /**
   * Fetch current weather data for a location.
   */
  async getCurrentWeather(location: string): Promise<WeatherData> {
    // Check cache first
    const now = new Date();
    const cacheKey = `${location}-${now.getFullYear()}-${now.getMonth()}-${now.getDate()}-${now.getHours()}`;
    
    const cached = weatherCache.get(cacheKey);
    if (cached && (now.getTime() - cached.timestamp.getTime()) < CACHE_TTL_MS) {
      console.log(`Returning cached weather for ${location}`);
      return cached.data;
    }

    // Check if API key is configured
    if (!this.apiKey) {
      console.warn("Weather API key not configured, returning fallback data");
      return this.getFallbackWeather(location);
    }

    try {
      const url = new URL(`${this.apiUrl}/weather`);
      url.searchParams.set("q", location);
      url.searchParams.set("appid", this.apiKey);
      url.searchParams.set("units", "metric"); // Celsius

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), this.timeout);

      const response = await fetch(url.toString(), {
        signal: controller.signal,
      });
      clearTimeout(timeoutId);

      if (response.ok) {
        const data = await response.json();
        const weatherData = this.parseOpenWeatherResponse(data, location);

        // Cache the result
        weatherCache.set(cacheKey, {
          data: weatherData,
          timestamp: new Date(),
        });

        console.log(`Weather fetched for ${location}: ${weatherData.temperature}°C`);
        return weatherData;
      } else {
        console.error(`Weather API returned status ${response.status}`);
        return this.getFallbackWeather(location);
      }
    } catch (error) {
      if (error instanceof Error && error.name === "AbortError") {
        console.error(`Weather API timeout for ${location}`);
      } else {
        console.error(`Error fetching weather:`, error);
      }
      return this.getFallbackWeather(location);
    }
  }

  /**
   * Parse OpenWeatherMap API response
   */
  private parseOpenWeatherResponse(data: any, location: string): WeatherData {
    return {
      location,
      timestamp: new Date().toISOString(),
      temperature: data.main.temp,
      humidity: data.main.humidity,
      pressure: data.main.pressure,
      precipitation: data.rain?.["1h"] || 0,
      windSpeed: data.wind.speed * 3.6, // m/s to km/h
      conditions: data.weather[0].main,
    };
  }

  /**
   * Return fallback weather data when API is unavailable
   */
  private getFallbackWeather(location: string): WeatherData {
    return {
      location,
      timestamp: new Date().toISOString(),
      temperature: 22.0,
      humidity: null,
      pressure: null,
      precipitation: null,
      windSpeed: null,
      conditions: "Unknown (API unavailable)",
    };
  }
}

/**
 * Map region name to a representative city for weather lookup.
 */
export function getLocationForRegion(region: string): string {
  return REGION_LOCATIONS[region] || region;
}

/**
 * Get city display name for a region.
 */
export function getCityNameForRegion(region: string): string {
  return REGION_CITY_NAMES[region] || region;
}

/**
 * Fetch weather for multiple regions.
 */
export async function getRegionalWeather(regions: string[]): Promise<Record<string, WeatherData>> {
  const weatherService = new WeatherService();
  const regionalWeather: Record<string, WeatherData> = {};

  for (const region of regions) {
    const location = getLocationForRegion(region);
    const weatherData = await weatherService.getCurrentWeather(location);
    regionalWeather[region] = weatherData;
  }

  return regionalWeather;
}

/**
 * Get current weather for a single region
 */
export async function getWeatherForRegion(region: string): Promise<WeatherData> {
  const weatherService = new WeatherService();
  const location = getLocationForRegion(region);
  return weatherService.getCurrentWeather(location);
}
