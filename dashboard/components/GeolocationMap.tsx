'use client';

import React, { useEffect, useState } from 'react';
import dynamic from 'next/dynamic';
import { MapPin, Truck, AlertTriangle, Navigation } from 'lucide-react';

// Dynamically import map components to avoid SSR issues
const MapContainer = dynamic(
  () => import('react-leaflet').then((mod) => mod.MapContainer),
  { ssr: false }
);

const TileLayer = dynamic(
  () => import('react-leaflet').then((mod) => mod.TileLayer),
  { ssr: false }
);

const Marker = dynamic(
  () => import('react-leaflet').then((mod) => mod.Marker),
  { ssr: false }
);

const Popup = dynamic(
  () => import('react-leaflet').then((mod) => mod.Popup),
  { ssr: false }
);

const Polyline = dynamic(
  () => import('react-leaflet').then((mod) => mod.Polyline),
  { ssr: false }
);

// Define cargo types and their colors
const CARGO_COLORS = {
  PERISH: '#ef4444', // red
  ELECT: '#3b82f6',  // blue
  HAZ: '#f59e0b',    // amber
  BULK: '#10b981',   // emerald
  FRAG: '#8b5cf6'    // violet
};

// Global shipment locations with realistic coordinates
const GLOBAL_LOCATIONS = {
  'TPE': { name: 'Taipei', lat: 25.0330, lng: 121.5654, country: 'Taiwan' },
  'LAX': { name: 'Los Angeles', lat: 34.0522, lng: -118.2437, country: 'USA' },
  'AMS': { name: 'Amsterdam', lat: 52.3676, lng: 4.9041, country: 'Netherlands' },
  'NRT': { name: 'Tokyo Narita', lat: 35.7648, lng: 140.3864, country: 'Japan' },
  'JFK': { name: 'New York JFK', lat: 40.6413, lng: -73.7781, country: 'USA' },
  'FRA': { name: 'Frankfurt', lat: 50.1109, lng: 8.6821, country: 'Germany' },
  'SIN': { name: 'Singapore', lat: 1.3521, lng: 103.8198, country: 'Singapore' },
  'SYD': { name: 'Sydney', lat: -33.9399, lng: 151.1753, country: 'Australia' },
  'DXB': { name: 'Dubai', lat: 25.2528, lng: 55.3644, country: 'UAE' },
  'LHR': { name: 'London Heathrow', lat: 51.4700, lng: -0.4543, country: 'UK' }
};

// Generate route paths between locations
const generateRoutePath = (origin: string, destination: string) => {
  const start = GLOBAL_LOCATIONS[origin as keyof typeof GLOBAL_LOCATIONS];
  const end = GLOBAL_LOCATIONS[destination as keyof typeof GLOBAL_LOCATIONS];
  
  if (!start || !end) return [];
  
  // Create a curved path for better visualization
  const midLat = (start.lat + end.lat) / 2;
  const midLng = (start.lng + end.lng) / 2;
  
  // Add some curve to the route
  const curveLat = midLat + (Math.random() - 0.5) * 10;
  const curveLng = midLng + (Math.random() - 0.5) * 20;
  
  return [
    [start.lat, start.lng],
    [curveLat, curveLng],
    [end.lat, end.lng]
  ];
};

// Mock shipment data with realistic tracking
const generateMockShipments = () => {
  const routes = [
    'TPE→LAX', 'TPE→AMS', 'TPE→NRT', 'AMS→JFK', 'NRT→FRA', 'SIN→SYD',
    'TPE→DXB', 'LAX→LHR', 'SIN→AMS', 'NRT→LAX'
  ];
  
  const cargoTypes = ['PERISH', 'ELECT', 'HAZ', 'BULK', 'FRAG'];
  const statuses = ['In Transit', 'At Port', 'Customs', 'Delivered', 'Delayed'];
  
  return Array.from({ length: 25 }, (_, i) => {
    const route = routes[Math.floor(Math.random() * routes.length)];
    const [origin, destination] = route.split('→');
    const cargo = cargoTypes[Math.floor(Math.random() * cargoTypes.length)];
    const status = statuses[Math.floor(Math.random() * statuses.length)];
    
    // Calculate current position along route (0-1)
    const progress = Math.random();
    const originCoords = GLOBAL_LOCATIONS[origin as keyof typeof GLOBAL_LOCATIONS];
    const destCoords = GLOBAL_LOCATIONS[destination as keyof typeof GLOBAL_LOCATIONS];
    
    const currentLat = originCoords.lat + (destCoords.lat - originCoords.lat) * progress;
    const currentLng = originCoords.lng + (destCoords.lng - originCoords.lng) * progress;
    
    return {
      id: `ARV-${99200 + i}`,
      route,
      origin,
      destination,
      cargo,
      status,
      currentPosition: [currentLat, currentLng],
      routePath: generateRoutePath(origin, destination),
      temperature: 20 + Math.random() * 15,
      humidity: 40 + Math.random() * 40,
      batteryLevel: 60 + Math.random() * 40,
      lastUpdate: new Date(Date.now() - Math.random() * 3600000).toISOString(),
      estimatedArrival: new Date(Date.now() + Math.random() * 7 * 24 * 3600000).toISOString(),
      riskScore: Math.random(),
      alerts: Math.random() > 0.7 ? ['Temperature Excursion'] : []
    };
  });
};

interface GeolocationMapProps {
  className?: string;
  height?: string;
  selectedShipment?: string | null;
  onShipmentSelect?: (shipmentId: string) => void;
}

const GeolocationMap: React.FC<GeolocationMapProps> = ({
  className = '',
  height = '500px',
  selectedShipment,
  onShipmentSelect
}) => {
  const [shipments, setShipments] = useState<any[]>([]);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [selectedRoute, setSelectedRoute] = useState<string | null>(null);
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
    setMapLoaded(true);
    
    // Generate shipments only on client side to avoid hydration mismatch
    const initialShipments = generateMockShipments();
    setShipments(initialShipments);
  }, []);

  useEffect(() => {
    if (!isClient) return;
    
    // Update shipment positions every 10 seconds
    const interval = setInterval(() => {
      setShipments(prev => prev.map(shipment => {
        const progress = Math.min(1, Math.random() * 0.1 + 0.9); // Simulate movement
        const originCoords = GLOBAL_LOCATIONS[shipment.origin as keyof typeof GLOBAL_LOCATIONS];
        const destCoords = GLOBAL_LOCATIONS[shipment.destination as keyof typeof GLOBAL_LOCATIONS];
        
        const newLat = originCoords.lat + (destCoords.lat - originCoords.lat) * progress;
        const newLng = originCoords.lng + (destCoords.lng - originCoords.lng) * progress;
        
        return {
          ...shipment,
          currentPosition: [newLat, newLng],
          lastUpdate: new Date().toISOString()
        };
      }));
    }, 10000);

    return () => clearInterval(interval);
  }, [isClient]);

  // Custom icon creation function
  const createCustomIcon = (cargo: string, hasAlert: boolean = false) => {
    if (typeof window === 'undefined') return null;
    
    const L = require('leaflet');
    const color = CARGO_COLORS[cargo as keyof typeof CARGO_COLORS] || '#6b7280';
    const alertBorder = hasAlert ? '#ef4444' : color;
    
    return L.divIcon({
      html: `
        <div style="
          width: 24px;
          height: 24px;
          background-color: ${color};
          border: 3px solid ${alertBorder};
          border-radius: 50%;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 0 2px 4px rgba(0,0,0,0.2);
          ${hasAlert ? 'animation: pulse 1s infinite;' : ''}
        ">
          <div style="
            width: 8px;
            height: 8px;
            background-color: white;
            border-radius: 50%;
          "></div>
        </div>
        <style>
          @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
            100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
          }
        </style>
      `,
      className: 'custom-marker',
      iconSize: [30, 30],
      iconAnchor: [15, 15]
    });
  };

  if (!mapLoaded || !isClient) {
    return (
      <div className={`${className} bg-gray-100 rounded-lg flex items-center justify-center`} style={{ height }}>
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-violet-600 mx-auto mb-2"></div>
          <p className="text-sm text-gray-600">Loading Global Tracking Map...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`relative ${className}`} style={{ height }}>
      <MapContainer
        center={[25.0, 120.0]} // Centered on Taiwan
        zoom={3}
        style={{ height: '100%', width: '100%' }}
        className="rounded-lg z-0"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        {/* Render route paths */}
        {shipments.map((shipment) => (
          <Polyline
            key={`route-${shipment.id}`}
            positions={shipment.routePath}
            color={CARGO_COLORS[shipment.cargo as keyof typeof CARGO_COLORS] || '#6b7280'}
            weight={selectedRoute === shipment.route ? 4 : 2}
            opacity={selectedRoute === shipment.route ? 0.8 : 0.3}
            dashArray={shipment.status === 'Delayed' ? '10, 10' : undefined}
          />
        ))}
        
        {/* Render shipment markers */}
        {shipments.map((shipment) => (
          <Marker
            key={shipment.id}
            position={shipment.currentPosition}
            icon={createCustomIcon(shipment.cargo, shipment.alerts.length > 0)}
            eventHandlers={{
              click: () => {
                setSelectedRoute(shipment.route);
                onShipmentSelect?.(shipment.id);
              }
            }}
          >
            <Popup>
              <div className="min-w-64 p-2">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-lg">{shipment.id}</h3>
                  <span 
                    className="px-2 py-1 rounded text-xs font-medium"
                    style={{ 
                      backgroundColor: CARGO_COLORS[shipment.cargo as keyof typeof CARGO_COLORS] + '20',
                      color: CARGO_COLORS[shipment.cargo as keyof typeof CARGO_COLORS]
                    }}
                  >
                    {shipment.cargo}
                  </span>
                </div>
                
                <div className="space-y-2 text-sm">
                  <div className="flex items-center gap-2">
                    <Navigation className="h-4 w-4 text-gray-500" />
                    <span>{shipment.route}</span>
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <Truck className="h-4 w-4 text-gray-500" />
                    <span className={`px-2 py-1 rounded text-xs ${
                      shipment.status === 'Delayed' ? 'bg-red-100 text-red-700' :
                      shipment.status === 'In Transit' ? 'bg-blue-100 text-blue-700' :
                      shipment.status === 'Delivered' ? 'bg-green-100 text-green-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      {shipment.status}
                    </span>
                  </div>
                  
                  {shipment.alerts.length > 0 && (
                    <div className="flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4 text-red-500" />
                      <span className="text-red-600 text-xs">{shipment.alerts.join(', ')}</span>
                    </div>
                  )}
                  
                  <div className="grid grid-cols-2 gap-4 mt-3 pt-2 border-t">
                    <div>
                      <div className="text-xs text-gray-500">Temperature</div>
                      <div className="font-medium">{shipment.temperature.toFixed(1)}°C</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-500">Battery</div>
                      <div className="font-medium">{shipment.batteryLevel.toFixed(0)}%</div>
                    </div>
                  </div>
                  
                  <div className="text-xs text-gray-500 mt-2">
                    Last Update: {new Date(shipment.lastUpdate).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
      
      {/* Map Legend */}
      <div className="absolute top-4 right-4 bg-white rounded-lg shadow-lg p-3 z-10">
        <h4 className="font-semibold text-sm mb-2">Cargo Types</h4>
        <div className="space-y-1">
          {Object.entries(CARGO_COLORS).map(([cargo, color]) => (
            <div key={cargo} className="flex items-center gap-2 text-xs">
              <div 
                className="w-3 h-3 rounded-full"
                style={{ backgroundColor: color }}
              ></div>
              <span>{cargo}</span>
            </div>
          ))}
        </div>
        <div className="mt-3 pt-2 border-t">
          <div className="flex items-center gap-2 text-xs">
            <div className="w-3 h-3 rounded-full border-2 border-red-500 bg-red-500 animate-pulse"></div>
            <span>Alert</span>
          </div>
        </div>
      </div>
      
      {/* Status Panel */}
      <div className="absolute bottom-4 left-4 bg-white rounded-lg shadow-lg p-3 z-10">
        <div className="text-sm">
          <div className="font-semibold mb-1">Live Tracking</div>
          <div className="text-gray-600">
            {shipments.length} Active Shipments
          </div>
          <div className="text-gray-600">
            {shipments.filter(s => s.alerts.length > 0).length} Alerts
          </div>
        </div>
      </div>
    </div>
  );
};

export default GeolocationMap;