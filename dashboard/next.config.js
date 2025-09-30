/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    // The experimental.appDir is no longer needed in Next.js 14+
  },
  // Ensure we can use external modules without issues
  transpilePackages: ['recharts', 'lucide-react']
};

module.exports = nextConfig;
